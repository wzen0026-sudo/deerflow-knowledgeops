"""Tests for the unified KnowledgeOps release orchestrator."""

import asyncio
import json
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from deerflow.community.knowledgeops import release_orchestrator


@pytest.fixture
def demo(tmp_path, monkeypatch):
    proposal_id = str(uuid4())
    knowledge_id = str(uuid4())

    journal = {
        "proposal_id": proposal_id,
        "knowledge_id": knowledge_id,
        "status": "SUBMITTED",
        "old_track_id": "track_old",
        "receipt": {
            "track_id": "track_new",
        },
    }

    journal_path = tmp_path / f"{proposal_id}.json"
    journal_path.write_text(json.dumps(journal), encoding="utf-8")

    index = AsyncMock(return_value={"status": "INDEXED"})
    retire = AsyncMock(return_value={"status": "DELETION_STARTED"})
    finalize = AsyncMock(return_value={"status": "PUBLISHED"})
    verify = AsyncMock(return_value={"status": "VERIFIED"})

    monkeypatch.setattr(
        release_orchestrator,
        "advance_release_indexing",
        index,
    )
    monkeypatch.setattr(
        release_orchestrator,
        "request_old_version_retirement",
        retire,
    )
    monkeypatch.setattr(
        release_orchestrator,
        "finalize_release_after_retirement",
        finalize,
    )
    monkeypatch.setattr(
        release_orchestrator,
        "verify_published_release",
        verify,
    )

    return {
        "proposal_id": proposal_id,
        "knowledge_id": knowledge_id,
        "journal_path": journal_path,
        "folder": tmp_path,
        "index": index,
        "retire": retire,
        "finalize": finalize,
        "verify": verify,
    }


def set_status(demo, status):
    path = demo["journal_path"]
    journal = json.loads(path.read_text(encoding="utf-8"))
    journal["status"] = status
    path.write_text(json.dumps(journal), encoding="utf-8")


def run_pipeline(demo, *, allow_retirement=False, old_track_id=None):
    return asyncio.run(
        release_orchestrator.advance_release_pipeline(
            proposal_id=demo["proposal_id"],
            journal_dir=demo["folder"],
            knowledge_dir=demo["folder"] / "knowledge",
            proposal_dir=demo["folder"] / "proposals",
            old_track_id=old_track_id,
            allow_retirement=allow_retirement,
        )
    )


@pytest.mark.parametrize("status", ["SUBMITTED", "INDEXING"])
def test_indexing_advances_to_retirement(demo, status):
    set_status(demo, status)

    result = run_pipeline(demo)

    assert result["status"] == "READY_FOR_RETIREMENT"
    demo["index"].assert_awaited_once()
    demo["retire"].assert_not_awaited()


def test_processing_does_not_delete(demo):
    demo["index"].return_value = {
        "status": "INDEXING",
        "remote_status": "PROCESSING",
    }

    result = run_pipeline(demo)

    assert result["status"] == "INDEXING"
    demo["retire"].assert_not_awaited()


def test_retirement_requires_approval(demo):
    set_status(demo, "INDEXED")

    result = run_pipeline(demo)

    assert result["status"] == "RETIREMENT_APPROVAL_REQUIRED"
    demo["retire"].assert_not_awaited()


def test_retirement_requires_old_track_id(demo):
    set_status(demo, "INDEXED")

    with pytest.raises(ValueError, match="tracking ID"):
        run_pipeline(demo, allow_retirement=True)

    demo["retire"].assert_not_awaited()


def test_authorized_retirement(demo):
    set_status(demo, "INDEXED")

    result = run_pipeline(
        demo,
        allow_retirement=True,
        old_track_id="track_old",
    )

    assert result["status"] == "DELETION_STARTED"
    demo["retire"].assert_awaited_once_with(
        proposal_id=demo["proposal_id"],
        old_track_id="track_old",
        journal_dir=demo["folder"],
        knowledge_dir=demo["folder"] / "knowledge",
        proposal_dir=demo["folder"] / "proposals",
    )


def test_finalization(demo):
    set_status(demo, "DELETION_STARTED")

    result = run_pipeline(demo)

    assert result["status"] == "PUBLISHED"
    demo["finalize"].assert_awaited_once()
    demo["retire"].assert_not_awaited()


def test_published_release_is_verified(demo):
    set_status(demo, "PUBLISHED")

    result = run_pipeline(demo)

    assert result["status"] == "VERIFIED"
    demo["verify"].assert_awaited_once()
    demo["retire"].assert_not_awaited()


@pytest.mark.parametrize(
    "status",
    [
        "SUBMISSION_UNKNOWN",
        "DELETION_UNKNOWN",
        "LOCAL_COMMIT_UNKNOWN",
        "INDEX_FAILED",
    ],
)
def test_uncertain_states_are_blocked(demo, status):
    set_status(demo, status)

    result = run_pipeline(
        demo,
        allow_retirement=True,
        old_track_id="track_old",
    )

    assert result["status"] == "NEEDS_RECONCILIATION"

    demo["index"].assert_not_awaited()
    demo["retire"].assert_not_awaited()
    demo["finalize"].assert_not_awaited()
    demo["verify"].assert_not_awaited()


def test_concurrent_release_is_blocked(demo):
    """Only one release may advance for the same knowledge."""

    entered = asyncio.Event()
    release_first = asyncio.Event()

    async def slow_index_check(**kwargs):
        entered.set()
        await release_first.wait()
        return {"status": "INDEXING"}

    demo["index"].side_effect = slow_index_check

    async def scenario():
        args = {
            "proposal_id": demo["proposal_id"],
            "journal_dir": demo["folder"],
            "knowledge_dir": demo["folder"] / "knowledge",
            "proposal_dir": demo["folder"] / "proposals",
        }

        # First release acquires the lock and pauses.
        first = asyncio.create_task(release_orchestrator.advance_release_pipeline(**args))

        await asyncio.wait_for(entered.wait(), timeout=5)

        try:
            # Second release must not execute concurrently.
            second = await release_orchestrator.advance_release_pipeline(**args)

            assert second["status"] == "RELEASE_BUSY"
            assert demo["index"].await_count == 1

        finally:
            release_first.set()

        first_result = await first
        assert first_result["status"] == "INDEXING"

        # The lock should now be available again.
        third = await release_orchestrator.advance_release_pipeline(**args)

        assert third["status"] == "INDEXING"
        assert demo["index"].await_count == 2

    asyncio.run(scenario())
