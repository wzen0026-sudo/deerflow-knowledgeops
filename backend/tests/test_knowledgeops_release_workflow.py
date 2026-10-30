"""Tests for the KnowledgeOps release workflow."""

import asyncio
import json
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from deerflow.community.knowledgeops import release_workflow


@pytest.fixture
def demo(tmp_path, monkeypatch):
    knowledge_id = str(uuid4())
    proposal_id = str(uuid4())

    history = {
        "current_version": 1,
        "versions": [{"version": 1, "content": "Old knowledge"}],
    }

    proposal = {
        "status": "APPROVED",
        "human_approved": True,
        "published": False,
        "reviewer": "test-reviewer",
        "reviewed_at_utc": "2026-10-29T10:00:00+00:00",
        "original_knowledge": "Old knowledge",
        "proposed_knowledge": "New knowledge",
    }

    submit = AsyncMock(
        return_value={
            "status": "SUBMITTED",
            "knowledge_id": knowledge_id,
            "version": 2,
            "track_id": "insert_test_001",
        }
    )

    monkeypatch.setattr(
        release_workflow,
        "load_knowledge_history",
        lambda *args: history,
    )
    monkeypatch.setattr(
        release_workflow,
        "load_update_proposal",
        lambda *args: proposal,
    )
    monkeypatch.setattr(
        release_workflow,
        "submit_version_for_indexing",
        submit,
    )

    return {
        "knowledge_id": knowledge_id,
        "proposal_id": proposal_id,
        "history": history,
        "proposal": proposal,
        "submit": submit,
        "folder": tmp_path,
    }


def run_release(demo):
    folder = demo["folder"]

    return asyncio.run(
        release_workflow.start_release_indexing(
            knowledge_id=demo["knowledge_id"],
            proposal_id=demo["proposal_id"],
            knowledge_dir=folder / "knowledge",
            proposal_dir=folder / "proposals",
            journal_dir=folder / "journals",
        )
    )


def journal_path(demo):
    return demo["folder"] / "journals" / f"{demo['proposal_id']}.json"


def test_approved_release_submits_once(demo):
    result = run_release(demo)

    assert result["status"] == "SUBMITTED"
    assert result["version"] == 2

    journal = json.loads(journal_path(demo).read_text(encoding="utf-8"))

    assert journal["status"] == "SUBMITTED"
    assert journal["receipt"]["track_id"] == "insert_test_001"

    demo["submit"].assert_awaited_once()

    print("Approved submission: PASSED")


def test_duplicate_submission_is_blocked(demo):
    run_release(demo)

    with pytest.raises(RuntimeError, match="journal already exists"):
        run_release(demo)

    assert demo["submit"].await_count == 1

    print("Duplicate submission: BLOCKED")


def test_unapproved_proposal_is_blocked(demo):
    demo["proposal"]["status"] = "PENDING_REVIEW"
    demo["proposal"]["human_approved"] = False

    with pytest.raises(ValueError, match="not approved"):
        run_release(demo)

    demo["submit"].assert_not_awaited()
    assert not journal_path(demo).exists()

    print("Unapproved proposal: BLOCKED")


def test_submission_failure_is_recorded(demo):
    demo["submit"].side_effect = ConnectionError("Simulated network failure")

    with pytest.raises(RuntimeError, match="uncertain"):
        run_release(demo)

    journal = json.loads(journal_path(demo).read_text(encoding="utf-8"))

    assert journal["status"] == "SUBMISSION_UNKNOWN"
    assert journal["receipt"] is None

    demo["submit"].assert_awaited_once()

    print("Network failure: RECORDED")
