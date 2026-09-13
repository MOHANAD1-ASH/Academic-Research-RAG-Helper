from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv

load_dotenv()


class ResearchAPIError(Exception):
    """User-facing backend error."""


def _base_url() -> str:
    return os.getenv("API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")


def api_url(value: str) -> str:
    """Turn backend-relative URLs into browser-usable absolute URLs."""
    value = str(value or "")
    if value.startswith("/"):
        return _base_url() + value
    return value


def _error_detail(response: httpx.Response) -> str:
    try:
        payload = response.json()
        detail = payload.get("detail") if isinstance(payload, dict) else None
        if detail:
            return str(detail)
    except ValueError:
        pass
    return f"Backend returned HTTP {response.status_code}."


def query_research_api(question: str, scope: str = "all", paper_id: str | None = None) -> tuple[str, list[dict[str, Any]]]:
    payload: dict[str, Any] = {"question": question, "scope": scope}
    if scope == "paper":
        payload["paper_id"] = paper_id
    try:
        response = httpx.post(f"{_base_url()}/query", json=payload, timeout=120.0)
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        raise ResearchAPIError(_error_detail(exc.response)) from exc
    except httpx.RequestError as exc:
        raise ResearchAPIError("Cannot reach the research backend. Start it and check API_BASE_URL.") from exc
    try:
        data: Any = response.json()
        answer, sources = data["answer"], data["sources"]
        if not isinstance(answer, str) or not isinstance(sources, list):
            raise ValueError
        return answer, [s for s in sources if isinstance(s, dict)]
    except (ValueError, KeyError, TypeError) as exc:
        raise ResearchAPIError("The backend returned an invalid /query response.") from exc


def list_papers() -> list[dict[str, Any]]:
    try:
        response = httpx.get(f"{_base_url()}/papers", timeout=20.0)
        response.raise_for_status()
        data = response.json()
        if not isinstance(data, list):
            raise ValueError
        return [item for item in data if isinstance(item, dict)]
    except httpx.HTTPStatusError as exc:
        raise ResearchAPIError(_error_detail(exc.response)) from exc
    except httpx.RequestError as exc:
        raise ResearchAPIError("Cannot reach the research backend.") from exc
    except (ValueError, TypeError) as exc:
        raise ResearchAPIError("The backend returned an invalid /papers response.") from exc


def ingest_pdf(path: str, source_url: str = "") -> dict[str, Any]:
    try:
        with open(path, "rb") as handle:
            response = httpx.post(
                f"{_base_url()}/papers/ingest",
                files={
                    "file": (Path(path).name, handle, "application/pdf"),
                    "source_url": (None, source_url.strip()),
                },
                timeout=600.0,
            )
        response.raise_for_status()
        data = response.json()
        if not isinstance(data, dict):
            raise ValueError
        return data
    except httpx.HTTPStatusError as exc:
        raise ResearchAPIError(_error_detail(exc.response)) from exc
    except httpx.TimeoutException as exc:
        raise ResearchAPIError("Ingestion timed out while the backend was processing the PDF. Check the backend console for the exact error.") from exc
    except httpx.RequestError as exc:
        raise ResearchAPIError(f"Cannot reach the research backend during ingestion ({type(exc).__name__}: {exc}).") from exc
    except (OSError, ValueError, TypeError) as exc:
        raise ResearchAPIError("The backend returned an invalid ingestion response.") from exc
