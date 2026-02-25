"""Tests for HITL change proposal models and service."""

from __future__ import annotations

from kgrag.hitl.change_proposals import (
    ChangeProposal,
    ChangeProposalService,
    ProposalStatus,
    ProposalType,
)


class TestChangeProposal:
    """Validate ChangeProposal dataclass."""

    def test_defaults(self) -> None:
        p = ChangeProposal()
        assert p.id.startswith("cp_")
        assert p.status == ProposalStatus.PROPOSED
        assert p.shacl_valid is None

    def test_custom(self) -> None:
        p = ChangeProposal(
            author="expert",
            proposal_type=ProposalType.ADD_ENTITY,
            proposed_data={"label": "New Facility", "entity_type": "Facility"},
            trigger_question="What facilities exist?",
        )
        assert p.proposal_type == ProposalType.ADD_ENTITY
        assert "label" in p.proposed_data


class TestChangeProposalService:
    """Test the in-memory proposal workflow."""

    def test_create_proposal(self) -> None:
        svc = ChangeProposalService(versioning_service=None)
        proposal = svc.create_proposal(
            proposal_type=ProposalType.UPDATE_ENTITY,
            proposed_data={"description": "Updated"},
            target_id="ent_abc",
            author="tester",
        )
        assert proposal.status == ProposalStatus.PROPOSED
        assert len(svc.list_proposals()) == 1

    def test_list_by_status(self) -> None:
        svc = ChangeProposalService(versioning_service=None)
        svc.create_proposal(ProposalType.ADD_ENTITY, {"label": "A"})
        svc.create_proposal(ProposalType.ADD_ENTITY, {"label": "B"})
        assert len(svc.list_proposals(status=ProposalStatus.PROPOSED)) == 2
        assert len(svc.list_proposals(status=ProposalStatus.ACCEPTED)) == 0
