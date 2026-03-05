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
            # ensure the tracer is returned in a list so callers can pass it
        # directly into LangChain constructors using ``callbacks=``
        return [tracer]
    except Exception as e:  # pragma: no cover - defensive
        logger.exception("Failed to create LangsmithTracer: %s", e)
        return None


def tracing_context(
    *,
    enabled: bool | None = None,
    metadata: dict[str, Any] | None = None,
    project: str | None = None,
) -> Any:
    """Return a context manager that enables LangSmith tracing.

    When tracing is disabled (either via ``LANGSMITH_TRACING`` or
    the ``enabled`` argument) or the ``langsmith`` package cannot be
    imported, this returns :func:`contextlib.nullcontext` so callers can
    simply ``with tracing_context(...):`` without guarding manually.

    ``metadata`` is passed through to :func:`langsmith.tracing_context`
    and is useful to attach things such as ``session_id`` or
    ``run_name``.  ``project`` overrides ``LANGSMITH_PROJECT`` if provided.
    """
    import contextlib

    if enabled is None:
        enabled = _env_enabled("LANGSMITH_TRACING")
    if not enabled:
        return contextlib.nullcontext()

    try:
        import langsmith as ls
    except Exception as e:  # pragma: no cover
        logger.warning("langsmith not importable: %s", e)
        return contextlib.nullcontext()

    project_name = project or os.environ.get("LANGSMITH_PROJECT")
    return ls.tracing_context(
        enabled=True,
        project_name=project_name,
        metadata=metadata or {},
    )


def tracing_context(
    *,
    project_name: str | None = None,
    tags: list[str] | None = None,
    metadata: dict[str, any] | None = None,
    enabled: bool | str | None = None,
    **kwargs: any,
):
    """Helper that returns a context manager for a LangSmith run.

    The context can be used to group multiple LangChain invocations under a
    single parent run.  It mirrors ``langsmith.tracing_context`` when
    tracing is enabled, and falls back to a no-op context otherwise.  The
    caller may pass metadata such as ``session_id`` or a custom run name.

    Example::

        with tracing_context(metadata={"session_id": sid}):
            result = chain.invoke(inputs)

    If ``LANGSMITH_TRACING`` is false or the package is unavailable, a
    ``contextlib.nullcontext`` is returned so user code can be written
    without additional conditionals.
    """
    # short-circuit when tracing is disabled at the environment level
    if not _env_enabled("LANGSMITH_TRACING"):
        from contextlib import nullcontext

        return nullcontext()

    try:
        import langsmith

        return langsmith.tracing_context(
            project_name=project_name,
            tags=tags,
            metadata=metadata,
            enabled=enabled,
            **kwargs,
        )
    except Exception:  # pragma: no cover - defensive
        # failed to import / create context, just return a no-op
        from contextlib import nullcontext

        return nullcontext()
