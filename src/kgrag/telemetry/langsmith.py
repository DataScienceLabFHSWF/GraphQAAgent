"""Optional Langsmith tracing helper (port from KnowledgeGraphBuilder).

The logic is identical to KGB's but lives in GraphQAAgent so the project can
opt in independently.  Callers should ask for callbacks and pass them to
LangChain models via ``callbacks=``; if tracing is disabled or the package
is missing, ``None`` is returned and callers just continue.
"""
from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)


def _env_enabled(name: str) -> bool:
    return str(os.environ.get(name, "false")).lower() in ("1", "true", "yes")


def get_langsmith_callbacks() -> list[Any] | None:
    """Return a list with a LangsmithTracer instance or ``None``.

    Reads the same environment variables as the KGB helper:

    * ``LANGSMITH_TRACING`` (boolean-like flag)
    * ``LANGSMITH_API_KEY`` (optionally read by the tracer itself)
    * ``LANGSMITH_PROJECT`` (project name)
    * ``LANGSMITH_ENDPOINT`` (custom endpoint)

    The returned list may be passed directly into LangChain model/chain
    constructors using the ``callbacks`` keyword.  If tracing is disabled or
    the ``langsmith`` package cannot be imported, ``None`` is returned so
    callers simply ignore it.
    """
    if not _env_enabled("LANGSMITH_TRACING"):
        return None

    try:
        from langchain.callbacks.tracers import LangsmithTracer
    except Exception as e:  # pragma: no cover - defensive
        logger.warning("LangsmithTracer not available: %s", e)
        return None

    project = os.environ.get("LANGSMITH_PROJECT")
    endpoint = os.environ.get("LANGSMITH_ENDPOINT")

    try:
        tracer = LangsmithTracer(project=project) if project else LangsmithTracer()
        if endpoint:
            try:
                setattr(tracer, "_endpoint", endpoint)
            except Exception:
                pass
        logger.info("Langsmith tracing enabled (project=%s)", project)
        return [tracer]
    except Exception as e:  # pragma: no cover - defensive
        logger.exception("Failed to create LangsmithTracer: %s", e)
        return None
