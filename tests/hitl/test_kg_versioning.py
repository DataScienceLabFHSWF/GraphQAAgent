"""Tests for HITL KG versioning models."""

from __future__ import annotations

from kgrag.hitl.kg_versioning import ChangeEvent, ChangeStatus, ChangeType


class TestChangeEvent:
    """Validate ChangeEvent dataclass."""

    def test_defaults(self) -> None:
        event = ChangeEvent()
        assert event.id.startswith("ce_")
        assert event.change_type == ChangeType.UPDATE
        assert event.status == ChangeStatus.APPLIED
        assert event.author == "system"

    def test_custom(self) -> None:
        event = ChangeEvent(
            author="expert-1",
            change_type=ChangeType.CREATE,
            target_type="entity",
            target_id="ent_abc123",
        )
        assert event.author == "expert-1"
        assert event.target_id == "ent_abc123"
