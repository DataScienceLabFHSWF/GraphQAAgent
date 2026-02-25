"""FastAPI server — entry point for the REST API.

Registers all routers:
- ``/api/v1/`` — core QA endpoints (``routes.py``)
- ``/api/v1/chat/`` — streaming chat + session management (``chat_routes.py``)
- ``/api/v1/explore/`` — KG / ontology browsing (``explorer_routes.py``)
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from collections.abc import AsyncGenerator

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from kgrag.api.routes import router, set_orchestrator
from kgrag.api.chat_routes import router as chat_router, set_session_manager
from kgrag.api.explorer_routes import router as explorer_router, set_connectors
from kgrag.agents.orchestrator import Orchestrator
from kgrag.chat.session import ChatSessionManager
from kgrag.core.config import Settings

logger = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Startup / shutdown lifecycle for the FastAPI app."""
    settings = Settings()  # type: ignore[call-arg]
    orchestrator = Orchestrator(settings)
    await orchestrator.startup()

    # Wire core QA
    set_orchestrator(orchestrator)

    # Wire chat (session management)
    session_manager = ChatSessionManager(orchestrator)
    set_session_manager(session_manager)

    # Wire explorer (read-only connectors)
    set_connectors(orchestrator.neo4j, orchestrator.fuseki)

    logger.info("api.started")

    yield

    await orchestrator.shutdown()
    logger.info("api.shutdown")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="KG-RAG QA Agent",
        description="Ontology-informed GraphRAG QA Agent with Chat, Explorer, and HITL support",
        version="0.1.0",
        lifespan=lifespan,
    )

    # CORS — allow Streamlit (default :8501) and local dev frontends
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:8501",  # Streamlit
            "http://localhost:3000",  # Next.js (if TypeScript frontend is used)
        ],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register routers
    app.include_router(router, prefix="/api/v1")
    app.include_router(chat_router, prefix="/api/v1")
    app.include_router(explorer_router, prefix="/api/v1")

    return app


app = create_app()
