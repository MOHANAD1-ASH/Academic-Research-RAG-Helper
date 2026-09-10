"""corpus.py — loads the notebook artifacts ONCE at startup. All services share it."""
from __future__ import annotations
import json, os, re
from pathlib import Path
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer


def _tokenize(text: str) -> list[str]:
    return re.findall(r"(?u)\b\w+\b", text.lower())


class Corpus:
    def __init__(self):
        base = Path(__file__).resolve().parents[2] / "data"   # backend/data
        env = os.getenv("ACADEMIC_RAG_ARTIFACTS")
        self.art_dir = None
        for c in ([Path(env)] if env else []) + [
                base / "artifacts", base / "vector_store", base]:
            if (c / "chunks.json").is_file() and (c / "papers.json").is_file():
                self.art_dir = c.resolve(); break
        if self.art_dir is None:
            raise FileNotFoundError(
                "papers.json/chunks.json not found under backend/data — "
                "copy them from the notebook artifacts folder.")
        self.pdfs_dir = base / "pdfs"
        self.chroma_dir = base / "vector_store" if (base / "vector_store").exists() \
            else self.art_dir.parent / "chroma"
        self.pdfs_dir.mkdir(parents=True, exist_ok=True)

        self.papers: list[dict] = json.loads(
            (self.art_dir / "papers.json").read_text(encoding="utf-8"))
        self.chunks: list[dict] = json.loads(
            (self.art_dir / "chunks.json").read_text(encoding="utf-8"))
        self.texts: list[str] = [c["text"] for c in self.chunks]
        self.bm25 = BM25Okapi([_tokenize(t) for t in self.texts])

        self.embedding_model = "BAAI/bge-m3"
        rc = self.art_dir / "run_config.json"
        if rc.is_file():
            try:
                self.embedding_model = json.loads(rc.read_text()).get(
                    "embedding_model", self.embedding_model)
            except Exception:
                pass
        self._embedder = self._reranker = self._collection = None

    @property
    def embedder(self):
        if self._embedder is None:
            self._embedder = SentenceTransformer(self.embedding_model)
        return self._embedder

    @property
    def reranker(self):
        if self._reranker is None:
            self._reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        return self._reranker

    @property
    def collection(self):
        if self._collection is None:
            from chromadb import PersistentClient
            from chromadb.config import Settings
            client = PersistentClient(path=str(self.chroma_dir),
                                      settings=Settings(anonymized_telemetry=False))
            self._collection = client.get_or_create_collection(
                "academic_papers", metadata={"hnsw:space": "cosine"})
        return self._collection

    def paper_by_id(self, pid: str) -> dict | None:
        return next((p for p in self.papers if p.get("paperId") == pid), None)


corpus = Corpus()   # main.py imports this at startup