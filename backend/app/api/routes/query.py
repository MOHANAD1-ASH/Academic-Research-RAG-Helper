from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Request, status

from app.schemas.query import QueryRequest, QueryResponse

LOGGER = logging.getLogger(__name__)
router = APIRouter(tags=["research"])


def get_services(request: Request):
    retrieval = getattr(request.app.state, "retrieval_service", None)
    generation = getattr(request.app.state, "generation_service", None)
    if retrieval is None or generation is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="The RAG services are not ready.")
    return retrieval, generation


@router.post("/query", response_model=QueryResponse)
def query_corpus(payload: QueryRequest, services=Depends(get_services)) -> QueryResponse:
    retrieval_service, generation_service = services
    try:
        ranked = retrieval_service.retrieve(payload.question, paper_id=payload.paper_id if payload.scope == "paper" else None)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="The selected paper was not found in the indexed library.") from exc
    except Exception as exc:
        LOGGER.exception("Retrieval failed")
        raise HTTPException(status_code=503, detail="Retrieval service is temporarily unavailable.") from exc

    if not ranked:
        return QueryResponse(answer="The corpus does not provide enough evidence.", sources=[])

    try:
        answer, diagnostic = generation_service.generate_answer(payload.question, ranked)
        LOGGER.info("Query completed: %s scope=%s paper_id=%s", diagnostic, payload.scope, payload.paper_id)
    except Exception as exc:
        LOGGER.exception("Generation failed; returning ranked evidence")
        answer = "LLM unavailable right now. The retrieved evidence is shown below."

    return QueryResponse(answer=answer, sources=generation_service.source_objects(ranked))
