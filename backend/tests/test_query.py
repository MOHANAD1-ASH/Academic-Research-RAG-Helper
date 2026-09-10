from contextlib import asynccontextmanager

from fastapi.testclient import TestClient

from app.api.routes.query import get_services
from app.main import create_app
from app.services.retrieval import RetrievedChunk


class FakeRetrievalService:
    def retrieve(self, question: str, paper_id: str | None = None) -> list[RetrievedChunk]:
        return [RetrievedChunk(
            chunk={
                "title": "Test Paper", "year": 2024, "page": 3,
                "source_url": "https://example.org/paper",
                "text": "Evidence from the fixed test corpus.",
            },
            text="Evidence from the fixed test corpus.", index=0,
        )]


class FakeGenerationService:
    def generate_answer(self, question: str, ranked: list[RetrievedChunk]) -> tuple[str, str]:
        return "A grounded answer. [S1]", "grounded citation tags found"

    def source_objects(self, ranked: list[RetrievedChunk]) -> list[dict]:
        return [{"source": "S1", "paper_id": "p1", "title": "Test Paper", "authors": [], "year": 2024, "page": 3, "section": None, "url": "https://example.org/paper", "pdf_url": ""}]


@asynccontextmanager
async def empty_lifespan(app):
    yield


def make_client() -> TestClient:
    app = create_app(lifespan_context=empty_lifespan)
    app.dependency_overrides[get_services] = lambda: (FakeRetrievalService(), FakeGenerationService())
    return TestClient(app)


def test_query_returns_answer_and_sources() -> None:
    with make_client() as client:
        response = client.post("/query", json={"question": "What does the corpus say?"})
    assert response.status_code == 200
    assert response.json() == {
        "answer": "A grounded answer. [S1]",
        "sources": [{"source": "S1", "paper_id": "p1", "title": "Test Paper", "authors": [], "year": 2024, "page": 3, "section": None, "url": "https://example.org/paper", "pdf_url": ""}],
    }


def test_query_rejects_blank_question() -> None:
    with make_client() as client:
        response = client.post("/query", json={"question": "   "})
    assert response.status_code == 422


def test_query_returns_503_when_retrieval_fails() -> None:
    class BrokenRetrieval:
        def retrieve(self, question: str, paper_id: str | None = None):
            raise RuntimeError("boom")

    app = create_app(lifespan_context=empty_lifespan)
    app.dependency_overrides[get_services] = lambda: (BrokenRetrieval(), FakeGenerationService())
    with TestClient(app) as client:
        response = client.post("/query", json={"question": "test"})
    assert response.status_code == 503


def test_health_reports_uninitialized_state() -> None:
    app = create_app(lifespan_context=empty_lifespan)
    with TestClient(app) as client:
        response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {
        "status": "starting",
        "retrieval": "not_ready",
        "generation": "not_ready",
        "ingestion": "not_ready",
    }
