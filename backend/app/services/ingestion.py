from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pymupdf as fitz

from app.core.config import Settings
from app.services.retrieval import RetrievalService, tokenize

LOGGER = logging.getLogger(__name__)

CHUNK_SIZE_WORDS = 220
CHUNK_OVERLAP_WORDS = 40
MIN_CHUNK_WORDS = 40
MIN_PDF_BYTES = 10_000
MIN_TOTAL_CHARS = 1_500

ARXIV_ID_RE = re.compile(r"arXiv:\s*(\d{4}\.\d{4,5})(?:v\d+)?", re.I)
DOI_RE = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+", re.I)
KNOWN_SECTIONS = {
    "abstract": "Abstract", "introduction": "Introduction", "background": "Background",
    "related work": "Related Work", "preliminaries": "Preliminaries", "method": "Methods",
    "methods": "Methods", "methodology": "Methodology", "approach": "Approach",
    "proposed method": "Proposed Method", "proposed approach": "Proposed Approach", "model": "Model",
    "experiments": "Experiments", "experimental setup": "Experimental Setup",
    "experimental results": "Experimental Results", "evaluation": "Evaluation", "results": "Results",
    "main results": "Results", "results and discussion": "Results and Discussion", "analysis": "Analysis",
    "ablation": "Ablation", "ablation study": "Ablation Study", "discussion": "Discussion",
    "conclusion": "Conclusion", "conclusions": "Conclusion", "conclusion and future work": "Conclusion",
    "limitations": "Limitations", "future work": "Future Work", "ethics": "Ethics",
    "broader impact": "Broader Impact", "acknowledgments": "Acknowledgments",
    "acknowledgements": "Acknowledgments", "references": "References", "appendix": "Appendix",
}
MAX_HEADING_CHARS = 70


def detect_section(line: str) -> str | None:
    raw = re.sub(r"\s+", " ", line.strip())
    if not raw or raw[-1] in ".,;:" or len(raw) > MAX_HEADING_CHARS:
        return None
    normalized = re.sub(r"[^a-zA-Z0-9 ]", "", raw.lower()).strip()
    if normalized in KNOWN_SECTIONS:
        return KNOWN_SECTIONS[normalized]
    match = re.match(r"^(\d+(?:\.\d+)*)\s+(.+)$", normalized)
    if match and match.group(2).strip() in KNOWN_SECTIONS:
        return KNOWN_SECTIONS[match.group(2).strip()]
    return None


def clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"-\n(?=\w)", "", text)
    text = re.sub(r"\n+", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def parse_pdf(pdf_path: str) -> list[dict[str, Any]]:
    pages: list[dict[str, Any]] = []
    current_section = "Unknown"
    with fitz.open(pdf_path) as doc:
        for page_idx, page in enumerate(doc):
            lines = [line.strip() for line in page.get_text("text").splitlines() if line.strip()]
            segments: list[tuple[str, list[str]]] = []
            section, buffer = current_section, []
            for line in lines:
                detected = detect_section(line)
                if detected:
                    if buffer:
                        segments.append((section, buffer))
                    section, buffer = detected, []
                else:
                    buffer.append(line)
            if buffer:
                segments.append((section, buffer))
            current_section = section
            for section_name, segment_lines in segments:
                text = clean_text("\n".join(segment_lines))
                if len(text) >= 80:
                    pages.append({"page": page_idx + 1, "section": section_name, "text": text})
    return pages


def word_chunks(text: str, size: int = CHUNK_SIZE_WORDS, overlap: int = CHUNK_OVERLAP_WORDS) -> list[str]:
    if size <= 0 or overlap < 0 or overlap >= size:
        raise ValueError("chunk size must be positive and overlap must be smaller than size")
    words = text.split()
    chunks: list[str] = []
    start = 0
    while start < len(words):
        end = min(start + size, len(words))
        chunk = " ".join(words[start:end]).strip()
        if len(chunk.split()) >= MIN_CHUNK_WORDS:
            chunks.append(chunk)
        if end == len(words):
            break
        start = max(end - overlap, start + 1)
    return chunks


def _norm_title(title: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", (title or "").lower())).strip()


class IngestionService:
    """Transactional PDF ingestion with startup repair and duplicate recovery."""

    def __init__(self, settings: Settings, retrieval: RetrievalService) -> None:
        self.settings = settings
        self.retrieval = retrieval
        self.pdfs_dir = settings.artifacts_dir.parent / "pdfs"
        self.pdfs_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        with self._lock:
            self._reconcile_library()

    def _existing_pdf_hashes(self) -> set[str]:
        result: set[str] = set()
        for path in self.pdfs_dir.glob("*.pdf"):
            try:
                result.add(hashlib.sha256(path.read_bytes()).hexdigest())
            except OSError:
                LOGGER.warning("Could not hash %s", path)
        return result

    def _paper_map(self) -> dict[str, dict[str, Any]]:
        return {
            str(p.get("paperId") or p.get("paper_id")): p
            for p in self.retrieval.papers_meta
            if p.get("paperId") or p.get("paper_id")
        }

    def _existing_titles(self) -> set[str]:
        return {_norm_title(p.get("title") or "") for p in self.retrieval.papers_meta if p.get("title")}

    def _find_paper(self, paper_id: str) -> dict[str, Any] | None:
        return self._paper_map().get(str(paper_id))

    def _pdf_for_hash(self, sha: str) -> Path | None:
        for path in self.pdfs_dir.glob("*.pdf"):
            try:
                if hashlib.sha256(path.read_bytes()).hexdigest() == sha:
                    return path
            except OSError:
                continue
        return None

    def _paper_id_for_hash_from_chunks(self, sha: str) -> str | None:
        pdf = self._pdf_for_hash(sha)
        if pdf is None:
            return None
        stem = pdf.stem
        for chunk in self.retrieval.chunks:
            pid = str(chunk.get("paper_id") or "")
            if pid and (pid == stem or stem.startswith(pid) or pid.startswith(stem)):
                return pid
        return None

    def _write_papers_only(self, papers: list[dict[str, Any]]) -> None:
        path = self.settings.artifacts_dir / "papers.json"
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(papers, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(path)

    def _write_chunks_only(self, chunks: list[dict[str, Any]]) -> None:
        path = self.settings.artifacts_dir / "chunks.json"
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(path)

    def _reconcile_library(self) -> None:
        """Make papers.json agree with chunks.json before serving the API."""
        by_id = self._paper_map()
        changed = False
        for paper in self.retrieval.papers_meta:
            paper_id = str(paper.get("paperId") or paper.get("paper_id") or "")
            if not paper_id:
                continue
            pdf = Path(str(paper.get("pdf_path") or "")) if paper.get("pdf_path") else self.pdfs_dir / f"{paper_id}.pdf"
            if not pdf.is_file():
                matches = list(self.pdfs_dir.glob(f"{paper_id}*.pdf"))
                pdf = matches[0] if matches else None
            if pdf is not None:
                pdf_url = f"/papers/{paper_id}/pdf"
                if paper.get("pdf_path") != str(pdf.resolve()) or paper.get("pdf_url") != pdf_url:
                    paper["pdf_path"] = str(pdf.resolve())
                    paper["pdf_url"] = pdf_url
                    changed = True
                for chunk in self.retrieval.chunks:
                    if str(chunk.get("paper_id")) == paper_id and chunk.get("pdf_url") != pdf_url:
                        chunk["pdf_url"] = pdf_url
                        changed = True
        missing_ids = sorted({str(c.get("paper_id")) for c in self.retrieval.chunks if c.get("paper_id")} - set(by_id))
        if not missing_ids:
            if changed:
                self._write_papers_only(self.retrieval.papers_meta)
                self._write_chunks_only(self.retrieval.chunks)
            return

        for paper_id in missing_ids:
            chunks = [c for c in self.retrieval.chunks if str(c.get("paper_id")) == paper_id]
            if not chunks:
                continue
            first = chunks[0]
            pdf = self.pdfs_dir / f"{paper_id}.pdf"
            if not pdf.is_file():
                matches = list(self.pdfs_dir.glob(f"{paper_id}*.pdf"))
                pdf = matches[0] if matches else None
            paper = {
                "paperId": paper_id,
                "corpusId": first.get("corpus_id"),
                "title": str(first.get("title") or paper_id),
                "abstract": "",
                "doi": None,
                "year": first.get("year"),
                "authors": [{"name": str(a)} for a in (first.get("authors") or []) if str(a).strip()],
                "url": str(first.get("source_url") or ""),
                "openAccessPdf": {"url": ""},
                "citationCount": first.get("citation_count") or 0,
                "venue": first.get("venue") or "Uploaded PDF",
                "fieldsOfStudy": ["Computer Science"],
                "search_query": "reconciled",
            }
            if pdf is not None:
                paper["pdf_path"] = str(pdf.resolve())
                paper["pdf_url"] = f"/papers/{paper_id}/pdf"
                for chunk in chunks:
                    chunk["pdf_url"] = paper["pdf_url"]
            self.retrieval.papers_meta.append(paper)
            changed = True
            LOGGER.warning("Reconciled orphaned paper into library: %s", paper_id)

        if changed:
            self._write_papers_only(self.retrieval.papers_meta)
            self._write_chunks_only(self.retrieval.chunks)

    def _rebuild_bm25(self) -> None:
        from rank_bm25 import BM25Okapi
        self.retrieval.chunk_texts = [c["text"] for c in self.retrieval.chunks]
        self.retrieval.id_to_index = {c["chunk_id"]: i for i, c in enumerate(self.retrieval.chunks)}
        self.retrieval.paper_ids = {str(c.get("paper_id")) for c in self.retrieval.chunks if c.get("paper_id")}
        self.retrieval.bm25 = BM25Okapi([tokenize(t) for t in self.retrieval.chunk_texts])

    def _duplicate_response(self, paper: dict[str, Any], reason: str, source_url: str = "") -> dict[str, Any]:
        pid = str(paper.get("paperId") or paper.get("paper_id") or "")
        if source_url and paper.get("url") != source_url:
            paper["url"] = source_url
            self._write_papers_only(self.retrieval.papers_meta)
            for chunk in self.retrieval.chunks:
                if str(chunk.get("paper_id")) == pid:
                    chunk["source_url"] = source_url
            self._write_chunks_only(self.retrieval.chunks)
        return {
            "ok": True,
            "message": f'Already indexed: "{paper.get("title") or "Untitled paper"}". {reason}',
            "paper_id": pid or None,
            "title": str(paper.get("title") or "Untitled paper"),
            "n_chunks": sum(1 for c in self.retrieval.chunks if str(c.get("paper_id")) == pid),
            "already_indexed": True,
            "repaired": False,
        }

    def _recover_duplicate(self, sha: str, source_url: str = "") -> dict[str, Any] | None:
        existing_pdf = self._pdf_for_hash(sha)
        if existing_pdf is None:
            return None

        # First choice: a metadata record already owns this exact file.
        for paper in self.retrieval.papers_meta:
            pdf_path = str(paper.get("pdf_path") or "")
            if pdf_path and Path(pdf_path).is_file():
                try:
                    if hashlib.sha256(Path(pdf_path).read_bytes()).hexdigest() == sha:
                        return self._duplicate_response(paper, "No new chunks were created.", source_url)
                except OSError:
                    pass

        # Second choice: chunks identify the orphaned PDF. Rebuild the missing paper row.
        paper_id = self._paper_id_for_hash_from_chunks(sha)
        if not paper_id:
            return None
        chunks = [c for c in self.retrieval.chunks if str(c.get("paper_id")) == paper_id]
        first = chunks[0] if chunks else {}
        paper = {
            "paperId": paper_id,
            "corpusId": first.get("corpus_id"),
            "title": str(first.get("title") or existing_pdf.stem),
            "abstract": "",
            "doi": None,
            "year": first.get("year"),
            "authors": [{"name": str(a)} for a in (first.get("authors") or []) if str(a).strip()],
            "url": str(first.get("source_url") or ""),
            "openAccessPdf": {"url": ""},
            "citationCount": first.get("citation_count") or 0,
            "venue": first.get("venue") or "Uploaded PDF",
            "fieldsOfStudy": ["Computer Science"],
            "search_query": "recovered",
            "pdf_path": str(existing_pdf.resolve()),
            "pdf_url": f"/papers/{paper_id}/pdf",
        }
        for chunk in chunks:
            chunk["pdf_url"] = paper["pdf_url"]
        self.retrieval.papers_meta.append(paper)
        self._write_papers_only(self.retrieval.papers_meta)
        return {
            "ok": True,
            "message": f'Repaired library record for already-indexed paper: "{paper["title"]}".',
            "paper_id": paper_id,
            "title": paper["title"],
            "n_chunks": len(chunks),
            "already_indexed": True,
            "repaired": True,
        }

    def _persist(self, chunks: list[dict[str, Any]], papers: list[dict[str, Any]], embeddings: np.ndarray) -> None:
        artifacts = self.settings.artifacts_dir
        artifacts.mkdir(parents=True, exist_ok=True)
        manifest_path = artifacts / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
        manifest.update({
            "embedding_model": self.settings.embedding_model,
            "reranker_model": self.settings.reranker_model,
            "num_chunks": len(chunks),
        })
        payloads = {
            "chunks.json": json.dumps(chunks, ensure_ascii=False, indent=2),
            "papers.json": json.dumps(papers, ensure_ascii=False, indent=2),
            "manifest.json": json.dumps(manifest, ensure_ascii=False, indent=2),
        }
        stage = artifacts / ".ingest_stage"
        backup = artifacts / ".ingest_backup"
        stage.mkdir(exist_ok=True)
        backup.mkdir(exist_ok=True)
        targets = [artifacts / name for name in (*payloads.keys(), "embeddings.npy")]
        try:
            for name, text in payloads.items():
                (stage / name).write_text(text, encoding="utf-8")
            np.save(stage / "embeddings.npy", embeddings.astype("float32"), allow_pickle=False)
            for target in targets:
                if target.exists():
                    target.replace(backup / target.name)
            for target in targets:
                (stage / target.name).replace(target)
        except Exception:
            for target in targets:
                target.unlink(missing_ok=True)
            for old in backup.iterdir():
                old.replace(artifacts / old.name)
            raise
        finally:
            for directory in (stage, backup):
                if directory.exists():
                    for child in directory.iterdir():
                        child.unlink(missing_ok=True)
                    directory.rmdir()

    def ingest(self, data: bytes, filename: str, source_url: str = "") -> dict[str, Any]:
        with self._lock:
            return self._ingest_locked(data, filename, source_url.strip())

    def _ingest_locked(self, data: bytes, filename: str, source_url: str = "") -> dict[str, Any]:
        if not data or not filename.lower().endswith(".pdf"):
            return {"ok": False, "message": "Only PDF files are accepted."}
        if len(data) < MIN_PDF_BYTES:
            return {"ok": False, "message": "File is too small to be a real paper."}

        sha = hashlib.sha256(data).hexdigest()
        if sha in self._existing_pdf_hashes():
            recovered = self._recover_duplicate(sha, source_url)
            if recovered:
                return recovered
            return {
                "ok": False,
                "message": "This PDF already exists on disk, but its index is inconsistent. Restart the backend to run automatic reconciliation before uploading it again.",
            }

        tmp = self.pdfs_dir / f".__tmp_{sha[:16]}.pdf"
        final_path: Path | None = None
        added_ids: list[str] = []
        old_chunks = list(self.retrieval.chunks)
        old_papers = list(self.retrieval.papers_meta)
        old_embeddings: np.ndarray | None = None
        try:
            tmp.write_bytes(data)
            pages = parse_pdf(str(tmp))
            if sum(len(p["text"]) for p in pages) < MIN_TOTAL_CHARS:
                return {"ok": False, "message": "Not enough extractable text (scanned/image PDF?)."}

            head = "\n".join(p["text"] for p in pages[:2])
            arxiv_match = ARXIV_ID_RE.search(head)
            arxiv_id = arxiv_match.group(1) if arxiv_match else None
            doi_match = DOI_RE.search(head)
            doi = doi_match.group(0).rstrip(".,;)") if doi_match else None

            existing_ids = {str(p.get("paperId") or p.get("paper_id")) for p in self.retrieval.papers_meta}
            existing_dois = {str(p.get("doi")).lower() for p in self.retrieval.papers_meta if p.get("doi")}
            if arxiv_id and arxiv_id in existing_ids:
                return self._duplicate_response(self._find_paper(arxiv_id) or {"paperId": arxiv_id, "title": arxiv_id}, "Matched by arXiv ID.", source_url)
            if doi and doi.lower() in existing_dois:
                paper = next(p for p in self.retrieval.papers_meta if str(p.get("doi")).lower() == doi.lower())
                return self._duplicate_response(paper, "Matched by DOI.", source_url)

            with fitz.open(str(tmp)) as doc:
                meta = doc.metadata or {}
            title = (meta.get("title") or "").strip()
            if len(title) < 12 or ".dvi" in title.lower() or title.lower().endswith((".tex", ".doc")):
                title = next((line.strip() for line in head.splitlines() if 20 <= len(line.strip()) <= 160 and not line.strip().lower().startswith("arxiv")), "")
            title = title or Path(filename).stem or "Untitled upload"
            normalized = _norm_title(title)
            if normalized and normalized in self._existing_titles():
                paper = next(p for p in self.retrieval.papers_meta if _norm_title(p.get("title") or "") == normalized)
                return self._duplicate_response(paper, "Matched by normalized title.", source_url)

            paper_id = arxiv_id or f"up{sha[:12]}"
            year_match = re.search(r"D:(\d{4})", meta.get("creationDate") or "")
            year = int(year_match.group(1)) if year_match else date.today().year
            authors = [{"name": a.strip()} for a in re.split(r"[;,]", meta.get("author") or "") if a.strip()]
            abstract = next((p["text"] for p in pages if p["section"] == "Abstract"), "")
            paper = {
                "paperId": paper_id,
                "corpusId": None,
                "title": title,
                "abstract": abstract[:1500],
                "doi": doi,
                "year": year,
                "authors": authors,
                "url": source_url or (f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else ""),
                "openAccessPdf": {"url": ""},
                "citationCount": 0,
                "venue": "Uploaded PDF",
                "fieldsOfStudy": ["Computer Science"],
                "search_query": "upload",
            }

            new_chunks: list[dict[str, Any]] = []
            for seg_i, page_info in enumerate(pages):
                for local_idx, text in enumerate(word_chunks(page_info["text"])):
                    new_chunks.append({
                        "chunk_id": f"{paper_id}_p{page_info['page']}_s{seg_i}_c{local_idx}",
                        "paper_id": paper_id,
                        "corpus_id": None,
                        "title": title,
                        "authors": [a["name"] for a in authors],
                        "year": year,
                        "page": page_info["page"],
                        "section": page_info["section"],
                        "venue": "Uploaded PDF",
                        "citation_count": 0,
                        "source_url": paper["url"],
                        "pdf_url": f"/papers/{paper_id}/pdf",
                        "text": text,
                    })
            if len(new_chunks) < 3:
                return {"ok": False, "message": "Extracted text was too short to index."}

            embeddings_path = self.settings.artifacts_dir / "embeddings.npy"
            if embeddings_path.is_file():
                old_embeddings = np.load(embeddings_path, allow_pickle=False).astype("float32")
                if old_embeddings.ndim != 2 or old_embeddings.shape[0] != len(old_chunks):
                    raise ValueError("Existing embeddings.npy is inconsistent with chunks.json; ingestion aborted.")
                new_embeddings = self.retrieval.embedder.encode(
                    [c["text"] for c in new_chunks], batch_size=64, show_progress_bar=False,
                    normalize_embeddings=True, convert_to_numpy=True,
                ).astype("float32")
                all_embeddings = np.vstack([old_embeddings, new_embeddings])
            else:
                all_embeddings = self.retrieval.embedder.encode(
                    [c["text"] for c in old_chunks + new_chunks], batch_size=64, show_progress_bar=False,
                    normalize_embeddings=True, convert_to_numpy=True,
                ).astype("float32")

            final_path = self.pdfs_dir / f"{re.sub(r'[^a-zA-Z0-9._-]+', '_', paper_id)[:80]}.pdf"
            if final_path.exists():
                final_path = self.pdfs_dir / f"{re.sub(r'[^a-zA-Z0-9._-]+', '_', paper_id)[:60]}_{sha[:8]}.pdf"
            tmp.replace(final_path)
            paper["pdf_path"] = str(final_path.resolve())
            paper["pdf_url"] = f"/papers/{paper_id}/pdf"
            combined_chunks = old_chunks + new_chunks
            combined_papers = old_papers + [paper]

            # Add vectors first, then atomically persist the correlated artifacts.
            # If persistence fails, the Chroma records are explicitly rolled back.
            new_ids = [c["chunk_id"] for c in new_chunks]
            self.retrieval.collection.add(
                ids=new_ids,
                documents=[c["text"] for c in new_chunks],
                embeddings=all_embeddings[-len(new_chunks):].tolist(),
                metadatas=[self.retrieval._chroma_metadata(c) for c in new_chunks],
            )
            added_ids = new_ids
            self._persist(combined_chunks, combined_papers, all_embeddings)

            self.retrieval.chunks = combined_chunks
            self.retrieval.papers_meta = combined_papers
            self._rebuild_bm25()
            return {
                "ok": True,
                "message": f'Indexed "{title}" — {len(new_chunks)} chunks added to the library and vector store.',
                "paper_id": paper_id,
                "title": title,
                "n_chunks": len(new_chunks),
                "already_indexed": False,
                "repaired": False,
            }
        except Exception as exc:
            LOGGER.exception("Ingestion failed")
            if added_ids:
                try:
                    self.retrieval.collection.delete(ids=added_ids)
                except Exception:
                    LOGGER.exception("Failed to rollback Chroma records")
            if final_path is not None:
                final_path.unlink(missing_ok=True)
            # Best effort restore in-memory state. Artifact rollback is handled by _persist.
            self.retrieval.chunks = old_chunks
            self.retrieval.papers_meta = old_papers
            if old_embeddings is not None:
                try:
                    np.save(self.settings.artifacts_dir / "embeddings.npy", old_embeddings, allow_pickle=False)
                except Exception:
                    LOGGER.exception("Failed to restore embeddings")
            self._rebuild_bm25()
            return {"ok": False, "message": f"Ingestion failed safely: {exc}"}
        finally:
            tmp.unlink(missing_ok=True)
