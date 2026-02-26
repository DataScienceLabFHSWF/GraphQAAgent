"""KG versioning via temporal properties and change event logging.

Every mutation to the knowledge graph is tracked as a ``ChangeEvent`` node
in Neo4j.  Entities and relations gain temporal properties so the graph
can be queried "as of" any point in time.

Schema additions to Neo4j
-------------------------
Nodes (new labels / properties):

.. code-block:: cypher

    // Temporal properties added to existing Entity / Relation nodes:
    // _version       : INT       — monotonically increasing version counter
    // _valid_from    : DATETIME  — when this version became active
    // _valid_to      : DATETIME  — NULL if current, set on supersession
    // _modified_by   : STRING    — user / system identifier
    // _change_event_id : STRING  — links to the ChangeEvent node

    // New node type for audit trail:
    CREATE (ce:ChangeEvent {
        id:            STRING,    // ce_<uuid>
        timestamp:     DATETIME,
        author:        STRING,    // who made the change
        change_type:   STRING,    // "create" | "update" | "delete" | "merge"
        target_type:   STRING,    // "entity" | "relation" | "property"
        target_id:     STRING,    // ID of the affected entity/relation
        before_snapshot: STRING,  // JSON of previous state (NULL for creates)
        after_snapshot:  STRING,  // JSON of new state (NULL for deletes)
        proposal_id:   STRING,    // links to the ChangeProposal if HITL-driven
        status:        STRING,    // "applied" | "rolled_back"
    })
"""

from __future__ import annotations

import json as json_mod
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Enums and data classes
# ---------------------------------------------------------------------------


class ChangeType(str, Enum):
    CREATE = "create"
    UPDATE = "update"
    DELETE = "delete"
    MERGE = "merge"


class ChangeStatus(str, Enum):
    APPLIED = "applied"
    ROLLED_BACK = "rolled_back"


@dataclass
class ChangeEvent:
    """Immutable record of a single KG mutation."""

    id: str = field(default_factory=lambda: f"ce_{uuid.uuid4().hex[:12]}")
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    author: str = "system"
    change_type: ChangeType = ChangeType.UPDATE
    target_type: str = "entity"  # "entity" | "relation" | "property"
    target_id: str = ""
    before_snapshot: dict[str, Any] | None = None
    after_snapshot: dict[str, Any] | None = None
    proposal_id: str | None = None
    status: ChangeStatus = ChangeStatus.APPLIED


# ---------------------------------------------------------------------------
# KG Versioning service
# ---------------------------------------------------------------------------


class KGVersioningService:
    """Manage temporal versioning of the knowledge graph.

    All mutation methods execute within Neo4j transactions so that the
    audit trail (``ChangeEvent`` nodes) and the temporal properties on
    entity/relation nodes are always consistent.
    """

    def __init__(self, neo4j_connector: Any) -> None:
        """
        Parameters
        ----------
        neo4j_connector:
            Instance of :class:`kgrag.connectors.neo4j.Neo4jConnector`.
        """
        self._neo4j = neo4j_connector

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _db(self) -> str:
        """Return the configured Neo4j database name."""
        return self._neo4j._config.database

    @staticmethod
    def _snap_json(data: dict[str, Any] | None) -> str | None:
        """Serialise a snapshot dict to JSON (or None)."""
        return json_mod.dumps(data, default=str) if data else None

    # ------------------------------------------------------------------
    # apply_change
    # ------------------------------------------------------------------

    async def apply_change(
        self,
        change_type: ChangeType,
        target_type: str,
        target_id: str,
        new_data: dict[str, Any],
        *,
        author: str = "system",
        proposal_id: str | None = None,
    ) -> ChangeEvent:
        """Apply a change to the KG and create an audit record.

        1. Read current state of the target (``before_snapshot``).
        2. Set ``_valid_to = now()`` on the current version.
        3. Create a new version with ``_valid_from = now()``,
           ``_version = prev + 1``.
        4. Create a ``ChangeEvent`` node linked to the target.
        """
        event = ChangeEvent(
            author=author,
            change_type=change_type,
            target_type=target_type,
            target_id=target_id,
            after_snapshot=new_data,
            proposal_id=proposal_id,
        )

        async with self._neo4j.driver.session(database=self._db()) as session:
            async with session.begin_transaction() as tx:
                # 1. Read + supersede current version
                read_q = (
                    "MATCH (n {id: $tid}) WHERE n._valid_to IS NULL "
                    "RETURN properties(n) AS props, elementId(n) AS eid"
                )
                res = await tx.run(read_q, tid=target_id)
                record = await res.single()

                if record and change_type != ChangeType.CREATE:
                    before = dict(record["props"])
                    event.before_snapshot = before
                    old_version = before.get("_version", 0)

                    # Close the old version
                    await tx.run(
                        "MATCH (n {id: $tid}) WHERE n._valid_to IS NULL "
                        "SET n._valid_to = datetime()",
                        tid=target_id,
                    )
                else:
                    old_version = 0

                # 2. Create / update the live node
                if change_type == ChangeType.DELETE:
                    # Soft-delete: we already closed the old version above
                    pass
                else:
                    # Build properties for the new version node
                    props = dict(new_data)
                    props["id"] = target_id
                    props["_version"] = old_version + 1
                    props["_valid_from"] = "datetime()"  # set below via Cypher
                    props["_modified_by"] = author
                    props["_change_event_id"] = event.id
                    # Remove meta keys that we set via Cypher SET
                    props.pop("_valid_from", None)

                    create_q = (
                        "CREATE (n2 {id: $tid}) "
                        "SET n2 += $props, "
                        "    n2._valid_from = datetime(), "
                        "    n2._version = $ver, "
                        "    n2._modified_by = $author, "
                        "    n2._change_event_id = $ceid"
                    )
                    await tx.run(
                        create_q,
                        tid=target_id,
                        props=new_data,
                        ver=old_version + 1,
                        author=author,
                        ceid=event.id,
                    )

                # 3. Create the ChangeEvent audit node
                ce_q = (
                    "CREATE (ce:ChangeEvent {"
                    "  id: $ceid, timestamp: datetime(), "
                    "  author: $author, change_type: $ctype, "
                    "  target_type: $ttype, target_id: $tid, "
                    "  before_snapshot: $before, after_snapshot: $after, "
                    "  proposal_id: $pid, status: $status"
                    "})"
                )
                await tx.run(
                    ce_q,
                    ceid=event.id,
                    author=author,
                    ctype=change_type.value,
                    ttype=target_type,
                    tid=target_id,
                    before=self._snap_json(event.before_snapshot),
                    after=self._snap_json(event.after_snapshot),
                    pid=proposal_id or "",
                    status=ChangeStatus.APPLIED.value,
                )

                await tx.commit()

        logger.info(
            "kg_versioning.apply_change",
            change_event_id=event.id,
            change_type=change_type.value,
            target_id=target_id,
        )
        return event

    # ------------------------------------------------------------------
    # query_as_of
    # ------------------------------------------------------------------

    async def query_as_of(
        self,
        entity_id: str,
        timestamp: datetime | None = None,
    ) -> dict[str, Any] | None:
        """Query the state of an entity at a specific point in time.

        If ``timestamp`` is ``None``, returns the current (live) version.
        """
        if timestamp is None:
            query = (
                "MATCH (n {id: $eid}) WHERE n._valid_to IS NULL "
                "RETURN properties(n) AS props"
            )
            params: dict[str, Any] = {"eid": entity_id}
        else:
            query = (
                "MATCH (n {id: $eid}) "
                "WHERE n._valid_from <= datetime($ts) "
                "  AND (n._valid_to IS NULL OR n._valid_to > datetime($ts)) "
                "RETURN properties(n) AS props"
            )
            params = {"eid": entity_id, "ts": timestamp.isoformat()}

        async with self._neo4j.driver.session(database=self._db()) as session:
            result = await session.run(query, **params)
            record = await result.single()

        return dict(record["props"]) if record else None

    # ------------------------------------------------------------------
    # get_entity_history
    # ------------------------------------------------------------------

    async def get_entity_history(self, entity_id: str) -> list[dict[str, Any]]:
        """Return all versions of an entity, ordered by ``_valid_from``."""
        query = (
            "MATCH (n {id: $eid}) "
            "RETURN properties(n) AS props "
            "ORDER BY n._valid_from"
        )
        async with self._neo4j.driver.session(database=self._db()) as session:
            result = await session.run(query, eid=entity_id)
            records = await result.data()
        return [dict(r["props"]) for r in records]

    # ------------------------------------------------------------------
    # rollback_change
    # ------------------------------------------------------------------

    async def rollback_change(self, change_event_id: str) -> None:
        """Rollback a change — restore ``before_snapshot`` and mark the event.

        1. Find the ChangeEvent node.
        2. Set ``_valid_to = now()`` on the post-change version.
        3. Create a new version from ``before_snapshot`` with
           ``_valid_from = now()``.
        4. Mark the ChangeEvent as ``rolled_back``.
        """
        async with self._neo4j.driver.session(database=self._db()) as session:
            async with session.begin_transaction() as tx:
                # 1. Find the ChangeEvent
                ce_q = (
                    "MATCH (ce:ChangeEvent {id: $ceid}) "
                    "RETURN ce.target_id AS tid, ce.before_snapshot AS before_snap, "
                    "       ce.status AS status"
                )
                res = await tx.run(ce_q, ceid=change_event_id)
                record = await res.single()

                if not record:
                    raise ValueError(f"ChangeEvent {change_event_id} not found")
                if record["status"] == ChangeStatus.ROLLED_BACK.value:
                    raise ValueError(f"ChangeEvent {change_event_id} already rolled back")

                target_id = record["tid"]
                before_json = record["before_snap"]
                before_data = json_mod.loads(before_json) if before_json else {}

                # 2. Close the post-change version
                await tx.run(
                    "MATCH (n {id: $tid}) WHERE n._valid_to IS NULL "
                    "SET n._valid_to = datetime()",
                    tid=target_id,
                )

                # 3. Restore before_snapshot as a new live version
                if before_data:
                    restore_q = (
                        "CREATE (n {id: $tid}) "
                        "SET n += $props, "
                        "    n._valid_from = datetime(), "
                        "    n._modified_by = 'rollback', "
                        "    n._change_event_id = $ceid"
                    )
                    await tx.run(
                        restore_q,
                        tid=target_id,
                        props=before_data,
                        ceid=change_event_id,
                    )

                # 4. Mark the ChangeEvent as rolled_back
                await tx.run(
                    "MATCH (ce:ChangeEvent {id: $ceid}) "
                    "SET ce.status = 'rolled_back'",
                    ceid=change_event_id,
                )

                await tx.commit()

        logger.info("kg_versioning.rollback", change_event_id=change_event_id)

    # ------------------------------------------------------------------
    # get_change_log
    # ------------------------------------------------------------------

    async def get_change_log(
        self,
        *,
        limit: int = 50,
        target_id: str | None = None,
        author: str | None = None,
    ) -> list[ChangeEvent]:
        """Return recent change events, optionally filtered."""
        clauses = ["MATCH (ce:ChangeEvent)"]
        where_parts: list[str] = []
        params: dict[str, Any] = {"limit": limit}

        if target_id:
            where_parts.append("ce.target_id = $tid")
            params["tid"] = target_id
        if author:
            where_parts.append("ce.author = $auth")
            params["auth"] = author

        if where_parts:
            clauses.append("WHERE " + " AND ".join(where_parts))
        clauses.append("RETURN ce ORDER BY ce.timestamp DESC LIMIT $limit")
        query = " ".join(clauses)

        async with self._neo4j.driver.session(database=self._db()) as session:
            result = await session.run(query, **params)
            records = await result.data()

        events: list[ChangeEvent] = []
        for rec in records:
            ce = rec["ce"]
            events.append(
                ChangeEvent(
                    id=ce.get("id", ""),
                    author=ce.get("author", "system"),
                    change_type=ChangeType(ce.get("change_type", "update")),
                    target_type=ce.get("target_type", "entity"),
                    target_id=ce.get("target_id", ""),
                    before_snapshot=(
                        json_mod.loads(ce["before_snapshot"])
                        if ce.get("before_snapshot")
                        else None
                    ),
                    after_snapshot=(
                        json_mod.loads(ce["after_snapshot"])
                        if ce.get("after_snapshot")
                        else None
                    ),
                    proposal_id=ce.get("proposal_id") or None,
                    status=ChangeStatus(ce.get("status", "applied")),
                )
            )
        return events
