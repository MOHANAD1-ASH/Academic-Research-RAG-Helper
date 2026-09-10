from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from app.core.config import Settings

LOGGER = logging.getLogger(__name__)


@dataclass
class RetrievedChunk:
    chunk: dict[str, Any]
    text: str
    index: int
    hybrid_score: float = 0.0
    rerank_score: float = 0.0


def tokenize(text: str) -> list[str]:
    """Small, deterministic tokenizer shared by BM25 and ingestion."""
    return re.findall(r"(?u)\b\w+\b", text.lower())


class RetrievalService:
    """Hybrid BM25 + dense retrieval + RRF + cross-encoder reranking."""

    bm25_top_k = 30
    dense_top_k = 30
    hybrid_top_k = 40
    rerank_top_k = 10

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.chunks = self._load_chunks(settings.artifacts_dir / "chunks.json")
        self.chunk_texts = [chunk["text"] for chunk in self.chunks]
        self.id_to_index = {chunk["chunk_id"]: i for i, chunk in enumerate(self.chunks)}
        self.papers_meta = self._load_papers(settings.artifacts_dir / "papers.json")
        self.paper_ids = {str(chunk.get("paper_id")) for chunk in self.chunks if chunk.get("paper_id")}
        self._validate_manifest()

        # Heavy ML/vector-store imports are intentionally lazy so API/unit tests can
        # import the application without installing the whole inference stack.
        from rank_bm25 import BM25Okapi
        from sentence_transformers import CrossEncoder, SentenceTransformer

        LOGGER.info("Loading BM25 index for %s chunks", len(self.chunks))
        self.bm25 = BM25Okapi([tokenize(text) for text in self.chunk_texts])

        LOGGER.info("Loading embedding model: %s", settings.embedding_model)
        self.embedder = SentenceTransformer(settings.embedding_model)
        LOGGER.info("Loading reranker model: %s", settings.reranker_model)
        self.reranker = CrossEncoder(settings.reranker_model)

        self._load_chroma()
        LOGGER.info("Retriever ready with %s Chroma records", self.collection.count())

    @staticmethod
    def _load_json(path: Path, default: Any) -> Any:
        if not path.is_file():
            return default
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON artifact: {path}") from exc

    @classmethod
    def _load_chunks(cls, path: Path) -> list[dict[str, Any]]:
        chunks = cls._load_json(path, None)
        if chunks is None:
            raise FileNotFoundError(f"Missing notebook artifact: {path}")
        if not isinstance(chunks, list) or not chunks:
            raise ValueError("chunks.json must contain a non-empty JSON list")

        required = {"chunk_id", "text", "title", "page", "paper_id", "year"}
        seen: set[str] = set()
        for position, chunk in enumerate(chunks):
            if not isinstance(chunk, dict):
                raise ValueError(f"chunks.json item {position} is not an object")
            missing = required.difference(chunk)
            if missing:
                raise ValueError(f"chunks.json item {position} is missing: {sorted(missing)}")
            chunk_id = str(chunk["chunk_id"])
            if not chunk_id or chunk_id in seen:
                raise ValueError(f"Duplicate/empty chunk_id in chunks.json: {chunk_id!r}")
            if not str(chunk["text"]).strip():
                raise ValueError(f"Empty text for chunk {chunk_id!r}")
            seen.add(chunk_id)
        return chunks

    @classmethod
    def _load_papers(cls, path: Path) -> list[dict[str, Any]]:
        raw = cls._load_json(path, [])
        if not isinstance(raw, list):
            raise ValueError("papers.json must contain a JSON list")
        return [item for item in raw if isinstance(item, dict)]

    def _validate_manifest(self) -> None:
        path = self.settings.artifacts_dir / "manifest.json"
        manifest = self._load_json(path, None)
        if manifest is None:
            raise FileNotFoundError(f"Missing notebook artifact: {path}")
        expected = {
            "embedding_model": self.settings.embedding_model,
            "reranker_model": self.settings.reranker_model,
            "num_chunks": len(self.chunks),
        }
        mismatches = [
            f"{key}={manifest.get(key)!r} (expected {value!r})"
            for key, value in expected.items()
            if manifest.get(key) != value
        ]
        if mismatches:
            raise ValueError("Notebook artifact configuration mismatch: " + "; ".join(mismatches))

    def _load_chroma(self) -> None:
        from chromadb import PersistentClient
        from chromadb.config import Settings as ChromaSettings

        self.settings.vector_store_dir.mkdir(parents=True, exist_ok=True)
        self.client = PersistentClient(
            path=str(self.settings.vector_store_dir),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            name=self.settings.chroma_collection,
            metadata={"hnsw:space": "cosine"},
        )

        expected_ids = set(self.id_to_index)
        actual_ids = set(self.collection.get(include=[]).get("ids", []))
        if actual_ids == expected_ids:
            return

        embeddings_path = self.settings.artifacts_dir / "embeddings.npy"
        if not embeddings_path.is_file():
            raise FileNotFoundError(f"Missing notebook artifact: {embeddings_path}")
        embeddings = np.load(embeddings_path, allow_pickle=False)
        if embeddings.ndim != 2 or embeddings.shape[0] != len(self.chunks):
            raise ValueError("embeddings.npy row count must match chunks.json")

        if actual_ids:
            LOGGER.warning("Chroma collection is out of sync; rebuilding it")
            self.client.delete_collection(self.settings.chroma_collection)
            self.collection = self.client.get_or_create_collection(
                name=self.settings.chroma_collection,
                metadata={"hnsw:space": "cosine"},
            )

        ids = [chunk["chunk_id"] for chunk in self.chunks]
        metadatas = [self._chroma_metadata(chunk) for chunk in self.chunks]
        batch_size = 5_000
        for start in range(0, len(ids), batch_size):
            end = start + batch_size
            self.collection.add(
                ids=ids[start:end],
                documents=self.chunk_texts[start:end],
                embeddings=embeddings[start:end].astype("float32").tolist(),
                metadatas=metadatas[start:end],
            )

    @staticmethod
    def _chroma_metadata(chunk: dict[str, Any]) -> dict[str, Any]:
        return {
            "paper_id": str(chunk.get("paper_id") or ""),
            "corpus_id": str(chunk.get("corpus_id") or ""),
            "title": str(chunk.get("title") or "Untitled"),
            "page": int(chunk.get("page") or 0),
            "year": int(chunk["year"]) if chunk.get("year") else 0,
            "section": str(chunk.get("section") or "Unknown"),
            "authors": ", ".join(chunk.get("authors", [])[:5]),
            "source_url": str(chunk.get("source_url") or ""),
            "pdf_url": str(chunk.get("pdf_url") or ""),
        }

    def bm25_search(self, query: str, top_k: int | None = None, paper_id: str | None = None) -> list[RetrievedChunk]:
        allowed = [
            i for i, chunk in enumerate(self.chunks)
            if paper_id is None or str(chunk.get("paper_id")) == str(paper_id)
        ]
        limit = min(top_k or self.bm25_top_k, len(allowed))
        if limit <= 0:
            return []
        scores = self.bm25.get_scores(tokenize(query))
        indices = sorted(allowed, key=lambda i: float(scores[i]), reverse=True)[:limit]
        return [
            RetrievedChunk(self.chunks[int(i)], self.chunk_texts[int(i)], int(i))
            for i in indices
        ]

    def dense_search(self, query: str, top_k: int | None = None, paper_id: str | None = None) -> list[RetrievedChunk]:
        limit = min(top_k or self.dense_top_k, len(self.chunks))
        if limit <= 0:
            return []
        query_embedding = self.embedder.encode(
            [query], normalize_embeddings=True, convert_to_numpy=True
        )[0].astype("float32")
        kwargs = {
            "query_embeddings": [query_embedding.tolist()],
            "n_results": limit,
            "include": ["distances"],
        }
        if paper_id is not None:
            if str(paper_id) not in self.paper_ids:
                return []
            kwargs["where"] = {"paper_id": str(paper_id)}
        result = self.collection.query(**kwargs)
        ids = (result.get("ids") or [[]])[0]
        out = []
        for chunk_id in ids:
            index = self.id_to_index.get(chunk_id)
            if index is not None:
                out.append(RetrievedChunk(self.chunks[index], self.chunk_texts[index], index))
        return out

    def hybrid_search(self, query: str, top_k: int | None = None, rrf_k: int = 60, paper_id: str | None = None) -> list[RetrievedChunk]:
        if rrf_k <= 0:
            raise ValueError("rrf_k must be positive")
        fused: dict[int, float] = {}
        information: dict[int, RetrievedChunk] = {}
        for rank, item in enumerate(self.bm25_search(query, paper_id=paper_id), start=1):
            fused[item.index] = fused.get(item.index, 0.0) + 1.0 / (rrf_k + rank)
            information[item.index] = item
        for rank, item in enumerate(self.dense_search(query, paper_id=paper_id), start=1):
            fused[item.index] = fused.get(item.index, 0.0) + 1.0 / (rrf_k + rank)
            information[item.index] = item
        results = [
            RetrievedChunk(information[index].chunk, information[index].text, index, hybrid_score=score)
            for index, score in fused.items()
        ]
        return sorted(results, key=lambda item: item.hybrid_score, reverse=True)[: top_k or self.hybrid_top_k]

    def rerank(self, query: str, candidates: list[RetrievedChunk]) -> list[RetrievedChunk]:
        if not candidates:
            return []
        scores = self.reranker.predict(
            [(query, candidate.text) for candidate in candidates],
            show_progress_bar=False,
        )
        ranked = [
            RetrievedChunk(
                candidate.chunk,
                candidate.text,
                candidate.index,
                candidate.hybrid_score,
                float(score),
            )
            for candidate, score in zip(candidates, scores)
        ]
        return sorted(ranked, key=lambda item: item.rerank_score, reverse=True)[: self.rerank_top_k]

    def retrieve(self, query: str, paper_id: str | None = None) -> list[RetrievedChunk]:
        query = query.strip()
        if not query:
            return []
        if paper_id is not None and str(paper_id) not in self.paper_ids:
            raise KeyError(f"Unknown paper_id: {paper_id}")
        return self.rerank(query, self.hybrid_search(query, paper_id=paper_id))
