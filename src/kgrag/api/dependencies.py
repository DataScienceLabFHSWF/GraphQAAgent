"""Lightweight helpers used by API modules for configuration and
cross-service communication.

Most of the project already uses ``kgrag.core.config.Settings``; this
module exposes a thin wrapper plus a `forward_to_kgbuilder` helper used by
HITL endpoints and the chat session manager.
"""

from __future__ import annotations

import httpx
import structlog

from kgrag.core.config import Settings

logger = structlog.get_logger(__name__)


def get_settings() -> Settings:
    """Return a fresh Settings instance (cached by Pydantic if desired)."""
    # Settings uses Pydantic's caching internally so calling multiple times
    # is cheap; we avoid ``@lru_cache`` here to keep test fixtures simple.
    return Settings()


def forward_to_kgbuilder(qa_results: list[dict]) -> dict:
    """POST low-confidence results to KGBuilder's HITL gap detector.

    The target URL is derived from the ``hitl.kgbuilder_api_url`` setting and
    should point at KGBuilder's ``/api/v1/hitl/gaps/detect`` endpoint.
    """
    settings = get_settings()
    base = settings.hitl.kgbuilder_api_url.rstrip("/")
    url = f"{base}/api/v1/hitl/gaps/detect"
    try:
        with httpx.Client(timeout=30.0) as client:
            resp = client.post(url, json={"qa_results": qa_results})
            resp.raise_for_status()
            return resp.json()
    except Exception as e:
        logger.warning("kgbuilder_forward_failed", url=url, error=str(e))
        return {"status": "error", "message": str(e)}
