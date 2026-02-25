"""Tests for the chat history stores."""

from __future__ import annotations

import pytest

from kgrag.chat.history import InMemoryHistoryStore, JsonFileHistoryStore, SqliteHistoryStore


class BaseHistoryTests:
    """Mixin with generic tests that apply to any HistoryStore."""

    @pytest.fixture
    def store(self):
        raise NotImplementedError

    @pytest.mark.asyncio
    async def test_save_and_load(self, store) -> None:
        await store.save_turn("s1", "Hello", "Hi there")
        turns = await store.load_session("s1")
        assert len(turns) == 1
        assert turns[0]["user"] == "Hello"
        assert turns[0]["assistant"] == "Hi there"

    @pytest.mark.asyncio
    async def test_load_empty(self, store) -> None:
        turns = await store.load_session("nonexistent")
        assert turns == []

    @pytest.mark.asyncio
    async def test_delete_session(self, store) -> None:
        await store.save_turn("s1", "Q", "A")
        await store.delete_session("s1")
        turns = await store.load_session("s1")
        assert turns == []

    @pytest.mark.asyncio
    async def test_list_sessions(self, store) -> None:
        await store.save_turn("s1", "Q1", "A1")
        await store.save_turn("s2", "Q2", "A2")
        sessions = await store.list_sessions()
        ids = {s["session_id"] for s in sessions}
        assert ids == {"s1", "s2"}


class TestInMemoryHistoryStore(BaseHistoryTests):
    @pytest.fixture
    def store(self) -> InMemoryHistoryStore:
        return InMemoryHistoryStore()


class TestJsonFileHistoryStore(BaseHistoryTests):
    @pytest.fixture
    def store(self, tmp_path) -> Any:
        # use temporary directory so tests don't pollute workspace
        return JsonFileHistoryStore(base_dir=str(tmp_path))


class TestSqliteHistoryStore(BaseHistoryTests):
    @pytest.fixture
    def store(self, tmp_path) -> Any:
        db = tmp_path / "history.db"
        return SqliteHistoryStore(db_path=str(db))
