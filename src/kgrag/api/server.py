"""FastAPI server — entry point for the REST API.

Registers all routers:
- ``/api/v1/`` — core QA endpoints (``routes.py``)
- ``/api/v1/chat/`` — streaming chat + session management (``chat_routes.py``)
- ``/api/v1/explore/`` — KG / ontology browsing (``explorer_routes.py``)
"""

from __future__ import annotations

import os
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

# ---------------------------------------------------------------------------
# CORS — configurable via the KGRAG_CORS_ORIGINS env var.
# Comma-separated list.  Falls back to sensible defaults for local dev.
# Set to "*" to allow all origins (NOT recommended for production).
# ---------------------------------------------------------------------------
_DEFAULT_CORS_ORIGINS = [
    "http://localhost:8501",   # Streamlit
    "http://localhost:3000",   # Next.js / gaia-tt dev server
    "http://localhost:5173",   # Vite dev server
]


def _cors_origins() -> list[str]:
    """Return allowed CORS origins from env or defaults."""
    raw = os.environ.get("KGRAG_CORS_ORIGINS", "")
    if not raw:
        return _DEFAULT_CORS_ORIGINS
    return [o.strip() for o in raw.split(",") if o.strip()]


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

    # CORS — configurable via KGRAG_CORS_ORIGINS env var
    origins = _cors_origins()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    logger.info("cors.configured", origins=origins)

    # Register routers
    app.include_router(router, prefix="/api/v1")
    app.include_router(chat_router, prefix="/api/v1")
    app.include_router(explorer_router, prefix="/api/v1")

    # HITL feedback & cross-service endpoints
    from kgrag.api.hitl_routes import router as hitl_router
    app.include_router(hitl_router, prefix="/api/v1")

    return app


app = create_app()
