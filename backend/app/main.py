from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import chat, documents, health, observability, settings
from app.core.config import get_settings
from app.core.logging import configure_logging
from app.services.observability import configure_langsmith


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    configure_langsmith()
    logging.getLogger(__name__).info("Starting AdvancedRAG AI backend")
    yield
    logging.getLogger(__name__).info("Stopping AdvancedRAG AI backend")


def create_app() -> FastAPI:
    app_settings = get_settings()
    app = FastAPI(
        title="AdvancedRAG AI",
        description="Production-grade document intelligence and Advanced RAG API.",
        version="1.0.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health.router, prefix="/api", tags=["health"])
    app.include_router(documents.router, prefix="/api/documents", tags=["documents"])
    app.include_router(chat.router, prefix="/api/chat", tags=["chat"])
    app.include_router(settings.router, prefix="/api/settings", tags=["settings"])
    app.include_router(observability.router, prefix="/api/observability", tags=["observability"])
    return app


app = create_app()
