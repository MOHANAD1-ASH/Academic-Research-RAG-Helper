from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes.papers import router as papers_router
from app.api.routes.query import router as query_router
from app.core.config import get_settings
from app.services.generation import GenerationService
from app.services.ingestion import IngestionService
from app.services.retrieval import RetrievalService
from app.utils.logging_config import configure_logging

LOGGER = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    configure_logging(settings.log_level)
    LOGGER.info("Starting %s", settings.app_name)
    retrieval = RetrievalService(settings)
    generation = GenerationService(settings)
    app.state.retrieval_service = retrieval
    app.state.generation_service = generation
    app.state.ingestion_service = IngestionService(settings, retrieval)
    try:
        yield
    finally:
        LOGGER.info("Stopping %s", settings.app_name)


def create_app(lifespan_context=lifespan) -> FastAPI:
    settings = get_settings()
    app = FastAPI(title=settings.app_name, version="1.0.0", lifespan=lifespan_context)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )

    @app.get("/health", tags=["health"])
    def health(request: Request) -> dict[str, str]:
        retrieval = getattr(request.app.state, "retrieval_service", None)
        generation = getattr(request.app.state, "generation_service", None)
        ingestion = getattr(request.app.state, "ingestion_service", None)
        ready = all(x is not None for x in (retrieval, generation, ingestion))
        return {
            "status": "ok" if ready else "starting",
            "retrieval": "ready" if retrieval else "not_ready",
            "generation": "ready" if generation else "not_ready",
            "ingestion": "ready" if ingestion else "not_ready",
        }

    app.include_router(query_router)
    app.include_router(papers_router)
    return app


app = create_app()
