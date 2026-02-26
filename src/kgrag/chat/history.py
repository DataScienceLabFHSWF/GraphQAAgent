"""Persistent chat history storage.

Provides three backends behind the ``HistoryStore`` protocol:

* ``InMemoryHistoryStore`` — non-persistent, for tests and quick dev.
* ``JsonFileHistoryStore`` — persists sessions as JSON files.
* ``SqliteHistoryStore`` — persists turns in a local SQLite database.

All backends support ``session_ttl_seconds`` for TTL-based session expiry.
"""

from __future__ import annotations

from typing import Any, Protocol

import structlog
import time
import asyncio

logger = structlog.get_logger(__name__)


class HistoryStore(Protocol):
    """Protocol for chat history persistence backends."""

    async def save_turn(
        self,
        session_id: str,
        user_message: str,
        assistant_message: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Persist a single conversation turn."""
        ...

    async def load_session(self, session_id: str) -> list[dict[str, Any]]:
        """Load all turns for a session, ordered by timestamp."""
        ...

    async def delete_session(self, session_id: str) -> None:
        """Delete all turns for a session."""
        ...

    async def list_sessions(self) -> list[dict[str, Any]]:
        """Return metadata for all stored sessions."""
        ...


# ---------------------------------------------------------------------------
# In-memory implementation (default / testing)
# ---------------------------------------------------------------------------


class InMemoryHistoryStore:
    """Non-persistent history store — data lives only in process memory.

    Good for development and tests.  Replace with ``JsonFileHistoryStore``
    or ``SqliteHistoryStore`` for production.

    Parameters
    ----------
    session_ttl_seconds:
        Maximum age (in seconds) of the *most recent* turn in a session
        before the session is eligible for expiry.  ``0`` disables TTL
        (default).
    """

    def __init__(self, session_ttl_seconds: int = 0) -> None:
        self._sessions: dict[str, list[dict[str, Any]]] = {}
        self.session_ttl_seconds = session_ttl_seconds

    async def save_turn(
        self,
        session_id: str,
        user_message: str,
        assistant_message: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if session_id not in self._sessions:
            self._sessions[session_id] = []
        self._sessions[session_id].append(
            {
                "user": user_message,
                "assistant": assistant_message,
                "timestamp": time.time(),
                **(metadata or {}),
            }
        )

    async def load_session(self, session_id: str) -> list[dict[str, Any]]:
        return self._sessions.get(session_id, [])

    async def delete_session(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)

    async def list_sessions(self) -> list[dict[str, Any]]:
        return [
            {"session_id": sid, "turns": len(turns)}
            for sid, turns in self._sessions.items()
        ]

    async def expire_sessions(self) -> int:
        """Delete sessions whose most recent turn is older than the TTL.

        Returns the number of expired sessions removed.
        """
        if self.session_ttl_seconds <= 0:
            return 0
        cutoff = time.time() - self.session_ttl_seconds
        expired = [
            sid
            for sid, turns in self._sessions.items()
            if turns and turns[-1].get("timestamp", 0) < cutoff
        ]
        for sid in expired:
            del self._sessions[sid]
        if expired:
            logger.info("expired_sessions", count=len(expired), store="memory")
        return len(expired)


# ---------------------------------------------------------------------------
# JSON file store
# ---------------------------------------------------------------------------


import json
import os
from pathlib import Path

class JsonFileHistoryStore:
    """Persist sessions as JSON files under a configurable directory.

    Each session is stored in ``{base_dir}/{session_id}.json`` as a list of
    turn dicts.  File I/O is wrapped in ``asyncio.to_thread`` to avoid
    blocking the event loop.

    Parameters
    ----------
    session_ttl_seconds:
        Maximum age (in seconds) of the *most recent* turn before the
        session file is eligible for expiry/deletion.  ``0`` disables TTL.
    """

    def __init__(self, base_dir: str = "data/chat_sessions", session_ttl_seconds: int = 0) -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.session_ttl_seconds = session_ttl_seconds

    async def save_turn(
        self,
        session_id: str,
        user_message: str,
        assistant_message: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        path = self.base_dir / f"{session_id}.json"
        turn = {
            "user": user_message,
            "assistant": assistant_message,
            "timestamp": time.time(),
            **(metadata or {}),
        }

        def _write():
            if path.exists():
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            else:
                data = []
            data.append(turn)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f)

        await asyncio.to_thread(_write)

    async def load_session(self, session_id: str) -> list[dict[str, Any]]:
        path = self.base_dir / f"{session_id}.json"
        if not path.exists():
            return []

        def _read():
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)

        return await asyncio.to_thread(_read)

    async def delete_session(self, session_id: str) -> None:
        path = self.base_dir / f"{session_id}.json"
        if path.exists():
            try:
                await asyncio.to_thread(path.unlink)
            except Exception:
                pass

    async def list_sessions(self) -> list[dict[str, Any]]:
        sessions = []
        for file in self.base_dir.glob("*.json"):
            try:
                data = await asyncio.to_thread(lambda p=file: json.load(open(p, "r")))
                sessions.append({"session_id": file.stem, "turns": len(data)})
            except Exception:
                continue
        return sessions

    async def expire_sessions(self) -> int:
        """Delete session files whose most recent turn is older than the TTL."""
        if self.session_ttl_seconds <= 0:
            return 0
        cutoff = time.time() - self.session_ttl_seconds
        expired = 0
        for file in self.base_dir.glob("*.json"):
            try:
                data = await asyncio.to_thread(lambda p=file: json.load(open(p, "r")))
                if data and data[-1].get("timestamp", 0) < cutoff:
                    await asyncio.to_thread(file.unlink)
                    expired += 1
            except Exception:
                continue
        if expired:
            logger.info("expired_sessions", count=expired, store="json")
        return expired


# ---------------------------------------------------------------------------
# SQLite store
# ---------------------------------------------------------------------------


import aiosqlite

class SqliteHistoryStore:
    """Persist sessions in a local SQLite database.

    The database is created automatically if it doesn't exist.

    Parameters
    ----------
    session_ttl_seconds:
        Maximum age (in seconds) of the *most recent* turn in a session
        before the session is eligible for expiry.  ``0`` disables TTL.
    """

    def __init__(self, db_path: str = "data/chat_history.db", session_ttl_seconds: int = 0) -> None:
        self.db_path = db_path
        self.session_ttl_seconds = session_ttl_seconds
        # ensure parent directory exists
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

    async def _ensure_schema(self) -> None:
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS turns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    user_msg TEXT NOT NULL,
                    asst_msg TEXT NOT NULL,
                    confidence REAL DEFAULT 0,
                    created_at REAL NOT NULL
                )
                """
            )
            await db.execute("CREATE INDEX IF NOT EXISTS idx_turns_session ON turns(session_id)")
            await db.commit()

    async def save_turn(
        self,
        session_id: str,
        user_message: str,
        assistant_message: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        await self._ensure_schema()
        confidence = metadata.get("confidence", 0.0) if metadata else 0.0
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "INSERT INTO turns (session_id, user_msg, asst_msg, confidence, created_at) VALUES (?,?,?,?,?)",
                (session_id, user_message, assistant_message, confidence, time.time()),
            )
            await db.commit()

    async def load_session(self, session_id: str) -> list[dict[str, Any]]:
        await self._ensure_schema()
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                "SELECT user_msg AS user, asst_msg AS assistant, confidence, created_at AS timestamp FROM turns WHERE session_id = ? ORDER BY created_at",
                (session_id,)
            )
            rows = await cursor.fetchall()
            cols = [c[0] for c in cursor.description]
        return [dict(zip(cols, row)) for row in rows]

    async def delete_session(self, session_id: str) -> None:
        await self._ensure_schema()
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "DELETE FROM turns WHERE session_id = ?",
                (session_id,)
            )
            await db.commit()

    async def list_sessions(self) -> list[dict[str, Any]]:
        await self._ensure_schema()
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                "SELECT session_id, COUNT(*) AS turns FROM turns GROUP BY session_id"
            )
            rows = await cursor.fetchall()
            cols = [c[0] for c in cursor.description]
        return [dict(zip(cols, row)) for row in rows]

    async def expire_sessions(self) -> int:
        """Delete sessions whose most recent turn is older than the TTL.

        Uses a single SQL statement for efficiency.
        """
        if self.session_ttl_seconds <= 0:
            return 0
        cutoff = time.time() - self.session_ttl_seconds
        await self._ensure_schema()
        async with aiosqlite.connect(self.db_path) as db:
            # Find sessions where the newest turn is older than the cutoff
            cursor = await db.execute(
                "SELECT session_id FROM turns "
                "GROUP BY session_id HAVING MAX(created_at) < ?",
                (cutoff,),
            )
            rows = await cursor.fetchall()
            expired_ids = [r[0] for r in rows]
            if expired_ids:
                placeholders = ",".join("?" for _ in expired_ids)
                await db.execute(
                    f"DELETE FROM turns WHERE session_id IN ({placeholders})",
                    expired_ids,
                )
                await db.commit()
                logger.info("expired_sessions", count=len(expired_ids), store="sqlite")
        return len(expired_ids) if expired_ids else 0
