"""Change proposal workflow for HITL-driven KG corrections.

Experts review and approve changes before they are applied to the live
knowledge graph.  Each proposal goes through a lifecycle::

    proposed → validated → accepted → applied
                 ↓
             rejected

Proposals are stored in-memory (dict).  For production persistence,
swap for SQLite or Neo4j ``ChangeProposal`` nodes.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Enums and models
# ---------------------------------------------------------------------------


class ProposalStatus(str, Enum):
    PROPOSED = "proposed"
    VALIDATED = "validated"  # passed SHACL + ontology checks
    ACCEPTED = "accepted"  # expert-approved
    APPLIED = "applied"  # written to KG
    REJECTED = "rejected"


class ProposalType(str, Enum):
    ADD_ENTITY = "add_entity"
    UPDATE_ENTITY = "update_entity"
    DELETE_ENTITY = "delete_entity"
    ADD_RELATION = "add_relation"
    UPDATE_RELATION = "update_relation"
    DELETE_RELATION = "delete_relation"
    ADD_PROPERTY = "add_property"
    UPDATE_PROPERTY = "update_property"


@dataclass
class ChangeProposal:
    """A proposed change to the knowledge graph."""

    id: str = field(default_factory=lambda: f"cp_{uuid.uuid4().hex[:12]}")
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    author: str = "anonymous"
    proposal_type: ProposalType = ProposalType.UPDATE_ENTITY
    status: ProposalStatus = ProposalStatus.PROPOSED

    # What to change
    target_id: str | None = None  # entity / relation ID (None for creates)
    target_type: str = "entity"
    proposed_data: dict[str, Any] = field(default_factory=dict)

    # Context
    trigger_question: str | None = None  # question that triggered this proposal
    trigger_confidence: float | None = None  # confidence of the answer that triggered it
    rationale: str = ""

    # Validation
    shacl_valid: bool | None = None
    validation_errors: list[str] = field(default_factory=list)

    # Review
    reviewer: str | None = None
    review_comment: str | None = None


# ---------------------------------------------------------------------------
# Proposal service
# ---------------------------------------------------------------------------


class ChangeProposalService:
    """Manage the lifecycle of change proposals.

    Validation checks basic field requirements and optionally runs SHACL
    validation when a Fuseki connector is available.  Accepted proposals
    are applied via :class:`KGVersioningService`.
    """

    def __init__(self, versioning_service: Any, fuseki_connector: Any = None) -> None:
        """
        Parameters
        ----------
        versioning_service:
            :class:`KGVersioningService` for applying accepted changes.
        fuseki_connector:
            Optional :class:`FusekiConnector` for SHACL validation.
        """
        self._versioning = versioning_service
        self._fuseki = fuseki_connector
        self._proposals: dict[str, ChangeProposal] = {}  # in-memory store

    def create_proposal(
        self,
        proposal_type: ProposalType,
        proposed_data: dict[str, Any],
        *,
        target_id: str | None = None,
        author: str = "anonymous",
        trigger_question: str | None = None,
        trigger_confidence: float | None = None,
        rationale: str = "",
    ) -> ChangeProposal:
        """Create a new change proposal and store it in-memory."""
        proposal = ChangeProposal(
            author=author,
            proposal_type=proposal_type,
            target_id=target_id,
            proposed_data=proposed_data,
            trigger_question=trigger_question,
            trigger_confidence=trigger_confidence,
            rationale=rationale,
        )
        self._proposals[proposal.id] = proposal
        logger.info(
            "hitl.proposal_created",
            proposal_id=proposal.id,
            proposal_type=proposal_type.value,
        )
        return proposal

    async def validate_proposal(self, proposal_id: str) -> ChangeProposal:
        """Validate a proposal against schema rules and (optionally) SHACL.

        Checks:
        1. Required fields present (``proposed_data`` is non-empty, target_id
           is set for updates/deletes).
        2. If a Fuseki connector is available, attempt SHACL validation via
           the ``N10sIntegration`` helper.
        3. Advance status to ``VALIDATED`` if all checks pass.
        """
        proposal = self._proposals.get(proposal_id)
        if not proposal:
            raise ValueError(f"Proposal {proposal_id} not found")

        errors: list[str] = []

        # Basic field validation
        if not proposal.proposed_data:
            errors.append("proposed_data must not be empty")

        needs_target = proposal.proposal_type in {
            ProposalType.UPDATE_ENTITY,
            ProposalType.DELETE_ENTITY,
            ProposalType.UPDATE_RELATION,
            ProposalType.DELETE_RELATION,
            ProposalType.UPDATE_PROPERTY,
        }
        if needs_target and not proposal.target_id:
            errors.append("target_id is required for update/delete proposals")

        # Optional SHACL validation (best-effort)
        if not errors and self._fuseki is not None:
            try:
                from kgrag.hitl.n10s_integration import N10sIntegration

                n10s = N10sIntegration(self._versioning._neo4j)
                if await n10s.check_n10s_available():
                    import json as _j

                    rdf_stub = _j.dumps(proposal.proposed_data)
                    results = await n10s.validate_with_shacl(rdf_stub)
                    for r in results:
                        if r.get("resultSeverity") == "Violation":
                            errors.append(r.get("resultMessage", "SHACL violation"))
                    proposal.shacl_valid = len(results) == 0 or all(
                        r.get("resultSeverity") != "Violation" for r in results
                    )
            except Exception as exc:
                logger.debug("shacl_validation_skipped", error=str(exc))
                proposal.shacl_valid = None  # indeterminate

        proposal.validation_errors = errors
        if errors:
            logger.info("hitl.proposal_validation_failed", proposal_id=proposal_id, errors=errors)
        else:
            proposal.status = ProposalStatus.VALIDATED
            logger.info("hitl.proposal_validated", proposal_id=proposal_id)

        return proposal

    async def accept_proposal(
        self,
        proposal_id: str,
        *,
        reviewer: str = "anonymous",
        comment: str = "",
    ) -> ChangeProposal:
        """Expert-approve a validated proposal."""
        proposal = self._proposals.get(proposal_id)
        if not proposal:
            raise ValueError(f"Proposal {proposal_id} not found")
        proposal.status = ProposalStatus.ACCEPTED
        proposal.reviewer = reviewer
        proposal.review_comment = comment
        logger.info("hitl.proposal_accepted", proposal_id=proposal_id)
        return proposal

    async def apply_proposal(self, proposal_id: str) -> None:
        """Apply an accepted proposal to the live KG via KGVersioningService.

        Delegates to ``KGVersioningService.apply_change`` and advances the
        proposal status to ``APPLIED``.
        """
        proposal = self._proposals.get(proposal_id)
        if not proposal:
            raise ValueError(f"Proposal {proposal_id} not found")
        if proposal.status != ProposalStatus.ACCEPTED:
            raise ValueError(
                f"Proposal {proposal_id} is '{proposal.status.value}', expected 'accepted'"
            )

        # Map ProposalType to a ChangeType for the versioning service
        from kgrag.hitl.kg_versioning import ChangeType

        _TYPE_MAP = {
            ProposalType.ADD_ENTITY: ChangeType.CREATE,
            ProposalType.UPDATE_ENTITY: ChangeType.UPDATE,
            ProposalType.DELETE_ENTITY: ChangeType.DELETE,
            ProposalType.ADD_RELATION: ChangeType.CREATE,
            ProposalType.UPDATE_RELATION: ChangeType.UPDATE,
            ProposalType.DELETE_RELATION: ChangeType.DELETE,
            ProposalType.ADD_PROPERTY: ChangeType.UPDATE,
            ProposalType.UPDATE_PROPERTY: ChangeType.UPDATE,
        }
        change_type = _TYPE_MAP.get(proposal.proposal_type, ChangeType.UPDATE)
        target_type = "relation" if "relation" in proposal.proposal_type.value else "entity"

        await self._versioning.apply_change(
            change_type=change_type,
            target_type=target_type,
            target_id=proposal.target_id or proposal.id,
            new_data=proposal.proposed_data,
            author=proposal.author,
            proposal_id=proposal.id,
        )
        proposal.status = ProposalStatus.APPLIED
        logger.info("hitl.proposal_applied", proposal_id=proposal_id)

    async def reject_proposal(
        self,
        proposal_id: str,
        *,
        reviewer: str = "anonymous",
        comment: str = "",
    ) -> ChangeProposal:
        """Reject a proposal."""
        proposal = self._proposals.get(proposal_id)
        if not proposal:
            raise ValueError(f"Proposal {proposal_id} not found")
        proposal.status = ProposalStatus.REJECTED
        proposal.reviewer = reviewer
        proposal.review_comment = comment
        logger.info("hitl.proposal_rejected", proposal_id=proposal_id)
        return proposal

    def list_proposals(
        self,
        *,
        status: ProposalStatus | None = None,
    ) -> list[ChangeProposal]:
        """List proposals, optionally filtered by status."""
        proposals = list(self._proposals.values())
        if status:
            proposals = [p for p in proposals if p.status == status]
        return sorted(proposals, key=lambda p: p.created_at, reverse=True)
