import pytest

from deerflow.community.knowledgeops.knowledge_release import (
    publish_approved_proposal,
    rollback_knowledge_version,
)
from deerflow.community.knowledgeops.knowledge_versions import (
    create_knowledge_history,
    load_knowledge_history,
)
from deerflow.community.knowledgeops.proposal_review import (
    review_update_proposal,
)
from deerflow.community.knowledgeops.proposal_store import (
    load_update_proposal,
    save_update_proposal,
)
from deerflow.community.knowledgeops.proposals import (
    create_update_proposal,
)

ORIGINAL = "Python 3.13 is the latest feature release."
UPDATED = "Check the current stable Python version."


def make_proposal(original=ORIGINAL, proposed=UPDATED):
    audit = {
        "claim": original,
        "investigation_status": "INSUFFICIENT_EVIDENCE",
        "evidence": [],
        "review_items": [],
    }

    return create_update_proposal(
        original_knowledge=original,
        proposed_knowledge=proposed,
        reason="Knowledge may be outdated.",
        audit_report=audit,
    )


def test_publish_and_rollback(tmp_path):
    """Approved changes create V2; rollback creates V3."""

    knowledge_dir = tmp_path / "knowledge"
    proposal_dir = tmp_path / "proposals"

    history = create_knowledge_history(ORIGINAL, knowledge_dir)
    knowledge_id = history["knowledge_id"]

    proposal = make_proposal()
    proposal_id = proposal["proposal_id"]

    save_update_proposal(proposal, proposal_dir)

    # Publishing before approval must fail.
    with pytest.raises(ValueError):
        publish_approved_proposal(
            knowledge_id,
            proposal_id,
            knowledge_dir,
            proposal_dir,
        )

    review_update_proposal(
        proposal_id,
        "APPROVE",
        "local-reviewer-demo",
        "Test approval.",
        proposal_dir,
    )

    history = publish_approved_proposal(
        knowledge_id,
        proposal_id,
        knowledge_dir,
        proposal_dir,
    )

    assert history["current_version"] == 2
    assert history["versions"][-1]["content"] == UPDATED

    saved_proposal = load_update_proposal(
        proposal_id,
        proposal_dir,
    )
    assert saved_proposal["published"] is True

    # Publishing the same proposal twice must fail.
    with pytest.raises(ValueError):
        publish_approved_proposal(
            knowledge_id,
            proposal_id,
            knowledge_dir,
            proposal_dir,
        )

    history = rollback_knowledge_version(
        knowledge_id,
        1,
        "local-reviewer-demo",
        "Restore V1 for testing.",
        knowledge_dir,
    )

    assert history["current_version"] == 3
    assert history["versions"][-1]["action"] == "ROLLBACK"
    assert history["versions"][-1]["content"] == ORIGINAL

    loaded = load_knowledge_history(
        knowledge_id,
        knowledge_dir,
    )
    assert len(loaded["versions"]) == 3


def test_rejected_proposal_cannot_publish(tmp_path):
    """Rejected proposals cannot change knowledge."""

    knowledge_dir = tmp_path / "knowledge"
    proposal_dir = tmp_path / "proposals"

    history = create_knowledge_history(ORIGINAL, knowledge_dir)
    proposal = make_proposal()

    save_update_proposal(proposal, proposal_dir)

    reviewed = review_update_proposal(
        proposal["proposal_id"],
        "REJECT",
        "local-reviewer-demo",
        "Evidence is insufficient.",
        proposal_dir,
    )

    assert reviewed["status"] == "REJECTED"
    assert reviewed["human_approved"] is False

    with pytest.raises(ValueError):
        publish_approved_proposal(
            history["knowledge_id"],
            proposal["proposal_id"],
            knowledge_dir,
            proposal_dir,
        )

    # A second review must also be blocked.
    with pytest.raises(ValueError):
        review_update_proposal(
            proposal["proposal_id"],
            "APPROVE",
            "local-reviewer-demo",
            "Attempted second review.",
            proposal_dir,
        )


def test_outdated_proposal_cannot_publish(tmp_path):
    """A proposal cannot overwrite a newer knowledge version."""

    knowledge_dir = tmp_path / "knowledge"
    proposal_dir = tmp_path / "proposals"

    history = create_knowledge_history(ORIGINAL, knowledge_dir)
    knowledge_id = history["knowledge_id"]

    first = make_proposal()
    second = make_proposal(proposed="Use a verified supported Python version.")

    for proposal in (first, second):
        save_update_proposal(proposal, proposal_dir)
        review_update_proposal(
            proposal["proposal_id"],
            "APPROVE",
            "local-reviewer-demo",
            "Workflow test.",
            proposal_dir,
        )

    # The first proposal creates V2.
    publish_approved_proposal(
        knowledge_id,
        first["proposal_id"],
        knowledge_dir,
        proposal_dir,
    )

    # The second proposal was based on old V1 content.
    with pytest.raises(ValueError, match="Knowledge has changed"):
        publish_approved_proposal(
            knowledge_id,
            second["proposal_id"],
            knowledge_dir,
            proposal_dir,
        )

    loaded = load_knowledge_history(
        knowledge_id,
        knowledge_dir,
    )

    assert loaded["current_version"] == 2
    assert loaded["versions"][-1]["content"] == UPDATED


def test_invalid_rollback_is_blocked(tmp_path):
    """An initial version cannot roll back to itself."""

    history = create_knowledge_history(ORIGINAL, tmp_path)

    with pytest.raises(ValueError):
        rollback_knowledge_version(
            history["knowledge_id"],
            1,
            "local-reviewer-demo",
            "Invalid rollback test.",
            tmp_path,
        )
