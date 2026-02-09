"""FastAPI server — entry point for the REST API."""

from __future__ import annotations

from contextlib import asynccontextmanager
from collections.abc import AsyncGenerator

import structlog
from fastapi import FastAPI

from kgrag.api.routes import router, set_orchestrator
from kgrag.agents.orchestrator import Orchestrator
from kgrag.core.config import Settings

logger = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Startup / shutdown lifecycle for the FastAPI app."""
    settings = Settings()  # type: ignore[call-arg]
    orchestrator = Orchestrator(settings)
    await orchestrator.startup()
    set_orchestrator(orchestrator)
    logger.info("api.started")

    yield

    await orchestrator.shutdown()
    logger.info("api.shutdown")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="KG-RAG QA Agent",
        description="Ontology-informed GraphRAG QA Agent",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.include_router(router, prefix="/api/v1")
    return app


app = create_app()
