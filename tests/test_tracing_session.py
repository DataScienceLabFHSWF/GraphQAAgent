"""Tests around improved LangSmith tracing and session grouping."""
from __future__ import annotations

import pytest

from kgrag.telemetry.langsmith import get_langsmith_callbacks, tracing_context


class DummyCM:
    def __init__(self):
        self.entered = False

    def __enter__(self):
        self.entered = True

    def __exit__(self, exc_type, exc, tb):
        return False


def test_tracing_context_no_env(monkeypatch):
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    cm = tracing_context(metadata={"foo": "bar"})
    # should be a no-op context manager
    with cm:
        pass
    assert cm.__class__.__name__ == "nullcontext" or not hasattr(cm, "entered")


def test_tracing_context_with_dummy(monkeypatch):
    # if langsmith is missing we should still get a nullcontext
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setitem(__import__("sys").modules, "langsmith", None)
    cm = tracing_context(metadata={"foo": "bar"})
    with cm:
        pass
    assert cm.__class__.__name__ == "nullcontext"


@pytest.mark.parametrize("enabled", [True, False])
def test_get_langsmith_callbacks_respects_env(monkeypatch, enabled: bool) -> None:
    monkeypatch.setenv("LANGSMITH_TRACING", "true" if enabled else "false")
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)

    callbacks = get_langsmith_callbacks()

    if enabled:
        assert callbacks is None or isinstance(callbacks, list)
    else:
        assert callbacks is None


def test_tracing_context_proxies_to_langsmith(monkeypatch):
    """When langsmith is installed we should get its context manager."""
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    dummy_ctx = DummyCM()
    # create a fake langsmith module with tracing_context record
    class FakeLS:
        def tracing_context(self, **kw):
            assert kw.get("metadata") == {"session": "xyz"}
            return dummy_ctx

    monkeypatch.setitem(__import__("sys").modules, "langsmith", FakeLS())

    cm = tracing_context(metadata={"session": "xyz"}, project="test")
    with cm:
        # the dummy context records entry via attribute
        pass
    assert dummy_ctx.entered
