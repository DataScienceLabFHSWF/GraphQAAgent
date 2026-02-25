"""Change proposal workflow for HITL-driven KG corrections.

Experts review and approve changes before they are applied to the live
knowledge graph.  Each proposal goes through a lifecycle::

    proposed → validated → accepted → applied
                 ↓
             rejected

Delegated implementation tasks
------------------------------
* TODO: Persist proposals (SQLite or Neo4j `ChangeProposal` nodes).
* TODO: Implement SHACL validation via n10s before accepting.
* TODO: Add a review UI in Streamlit (Phase D extension).
* TODO: Wire accepted proposals to ``KGVersioningService.apply_change``.
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

    All methods are stubs.
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
        """Create a new change proposal.

        TODO (delegate): Persist to storage.
        """
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
        """Validate a proposal against SHACL shapes via n10s / Fuseki.

        Steps:
        1. Convert ``proposed_data`` to RDF triples.
        2. Run SHACL validation against the domain ontology.
        3. Update ``shacl_valid`` and ``validation_errors``.
        4. If valid, advance status to ``VALIDATED``.

        TODO (delegate): Implement SHACL validation.  If n10s is available::

            // In Neo4j with n10s plugin:
            CALL n10s.validation.shacl.validate(
                $rdf_triples, 'Turtle'
            ) YIELD focusNode, resultMessage, ...

        Alternative: Use pyshacl library directly on the Fuseki endpoint.
        """
        raise NotImplementedError("ChangeProposalService.validate_proposal")

    async def accept_proposal(
        self,
        proposal_id: str,
        *,
        reviewer: str = "anonymous",
        comment: str = "",
    ) -> ChangeProposal:
        """Expert-approve a validated proposal.

        TODO (delegate): Update status and persist.
        """
        proposal = self._proposals.get(proposal_id)
        if not proposal:
            raise ValueError(f"Proposal {proposal_id} not found")
        proposal.status = ProposalStatus.ACCEPTED
        proposal.reviewer = reviewer
        proposal.review_comment = comment
        logger.info("hitl.proposal_accepted", proposal_id=proposal_id)
        return proposal

    async def apply_proposal(self, proposal_id: str) -> None:
        """Apply an accepted proposal to the live KG.

        TODO (delegate): Call ``KGVersioningService.apply_change`` and
        advance status to ``APPLIED``.
        """
        raise NotImplementedError("ChangeProposalService.apply_proposal")

    async def reject_proposal(
        self,
        proposal_id: str,
        *,
        reviewer: str = "anonymous",
        comment: str = "",
    ) -> ChangeProposal:
        """Reject a proposal.

        TODO (delegate): Update status and persist.
        """
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
