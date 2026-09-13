from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile, status
from fastapi.responses import FileResponse
from starlette.concurrency import run_in_threadpool

from app.schemas.papers import IngestResponse, PaperResponse
from app.core.config import get_settings
from app.services.ingestion import IngestionService

router = APIRouter(prefix="/papers", tags=["papers"])


def get_ingestion_service(request: Request) -> IngestionService:
    service = getattr(request.app.state, "ingestion_service", None)
    if service is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Ingestion service is not ready.")
    return service


@router.get("", response_model=list[PaperResponse])
def list_papers(request: Request) -> list[dict]:
    retrieval = getattr(request.app.state, "retrieval_service", None)
    if retrieval is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Retrieval service is not ready.")
    return retrieval.papers_meta


@router.get("/{paper_id}/pdf")
def get_paper_pdf(paper_id: str, request: Request):
    retrieval = getattr(request.app.state, "retrieval_service", None)
    if retrieval is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Retrieval service is not ready.")

    paper = next((p for p in retrieval.papers_meta if str(p.get("paperId") or p.get("paper_id")) == str(paper_id)), None)
    if paper is None:
        raise HTTPException(status_code=404, detail="The selected paper was not found in the indexed library.")

    pdf_path = paper.get("pdf_path")
    if not pdf_path:
        raise HTTPException(status_code=404, detail="A local PDF is not available for this paper.")

    path = Path(str(pdf_path)).resolve()
    pdf_root = (get_settings().artifacts_dir.parent / "pdfs").resolve()
    try:
        path.relative_to(pdf_root)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid paper file path.") from exc
    if not path.is_file():
        raise HTTPException(status_code=404, detail="The local PDF file is missing.")
    return FileResponse(path, media_type="application/pdf", filename=path.name)


@router.post("/ingest", response_model=IngestResponse)
async def ingest_paper(
    file: UploadFile = File(...),
    source_url: str = Form(default=""),
    ingestion: IngestionService = Depends(get_ingestion_service),
) -> IngestResponse:
    filename = file.filename or ""
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=415, detail="Only PDF files are accepted.")
    settings = get_settings()
    max_bytes = settings.max_upload_mb * 1024 * 1024
    data = await file.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise HTTPException(status_code=413, detail=f"PDF exceeds the {settings.max_upload_mb} MB upload limit.")
    # Ingestion is CPU-heavy (PDF parsing, embeddings, Chroma, BM25). Run it off
    # FastAPI's event loop so the server remains responsive while indexing.
    result = await run_in_threadpool(ingestion.ingest, data, filename or "upload.pdf", source_url.strip())
    if not result["ok"]:
        raise HTTPException(status_code=400, detail=result["message"])
    return IngestResponse(**result)
