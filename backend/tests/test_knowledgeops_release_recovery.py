"""Tests for interrupted KnowledgeOps publication recovery."""

import asyncio
import json
from functools import partial
from unittest.mock import AsyncMock
from uuid import uuid4

import httpx
import pytest

from deerflow.community.knowledgeops import release_recovery


@pytest.fixture
def demo(tmp_path, monkeypatch):
    knowledge_id = str(uuid4())
    proposal_id = str(uuid4())

    journal_dir = tmp_path / "journals"
    knowledge_dir = tmp_path / "knowledge"
    proposal_dir = tmp_path / "proposals"

    for folder in (journal_dir, knowledge_dir, proposal_dir):
        folder.mkdir()

    journal = {
        "knowledge_id": knowledge_id,
        "proposal_id": proposal_id,
        "old_version": 1,
        "new_version": 2,
        "status": "LOCAL_COMMIT_UNKNOWN",
        "old_track_id": "insert_old_test",
        "receipt": {
            "status": "SUBMITTED",
            "knowledge_id": knowledge_id,
            "version": 2,
            "track_id": "insert_new_test",
        },
    }

    journal_path = journal_dir / f"{proposal_id}.json"
    journal_path.write_text(json.dumps(journal), encoding="utf-8")

    history = {
        "current_version": 2,
        "versions": [
            {"version": 1, "content": "Old knowledge"},
            {
                "version": 2,
                "content": "New knowledge",
                "proposal_id": proposal_id,
            },
        ],
    }

    proposal = {
        "status": "APPROVED",
        "human_approved": True,
        "published": True,
        "published_version": 2,
        "knowledge_id": knowledge_id,
        "original_knowledge": "Old knowledge",
        "proposed_knowledge": "New knowledge",
    }

    state = {
        "knowledge_id": knowledge_id,
        "proposal_id": proposal_id,
        "journal_dir": journal_dir,
        "knowledge_dir": knowledge_dir,
        "proposal_dir": proposal_dir,
        "journal_path": journal_path,
        "history": history,
        "proposal": proposal,
    }

    monkeypatch.setattr(
        release_recovery,
        "load_knowledge_history",
        lambda *args: state["history"],
    )
    monkeypatch.setattr(
        release_recovery,
        "load_update_proposal",
        lambda *args: state["proposal"],
    )

    verify = AsyncMock(return_value={"status": "VERIFIED"})

    monkeypatch.setattr(
        release_recovery,
        "verify_published_release",
        verify,
    )

    state["verify"] = verify

    return state


def run_recovery(demo):
    return asyncio.run(
        release_recovery.recover_completed_local_release(
            proposal_id=demo["proposal_id"],
            journal_dir=demo["journal_dir"],
            knowledge_dir=demo["knowledge_dir"],
            proposal_dir=demo["proposal_dir"],
        )
    )


def saved_journal(demo):
    return json.loads(demo["journal_path"].read_text(encoding="utf-8"))


def test_completed_commit_is_recovered(demo):
    result = run_recovery(demo)

    assert result["status"] == "RECOVERED"
    assert result["current_version"] == 2
    assert saved_journal(demo)["status"] == "PUBLISHED"

    demo["verify"].assert_awaited_once()

    # Repeated recovery must not publish another version.
    second = run_recovery(demo)
    assert second["status"] == "ALREADY_PUBLISHED"

    demo["verify"].assert_awaited_once()


def test_partial_proposal_commit_is_blocked(demo):
    demo["proposal"]["published"] = False

    result = run_recovery(demo)

    assert result["status"] == "PARTIAL_LOCAL_COMMIT"
    assert saved_journal(demo)["status"] == "LOCAL_COMMIT_UNKNOWN"
    demo["verify"].assert_not_awaited()


def test_partial_history_commit_is_blocked(demo):
    demo["history"]["current_version"] = 1
    demo["history"]["versions"].pop()

    result = run_recovery(demo)

    assert result["status"] == "PARTIAL_LOCAL_COMMIT"
    demo["verify"].assert_not_awaited()


def test_remote_verification_failure_blocks_recovery(demo):
    demo["verify"].side_effect = RuntimeError("Old document is still indexed")

    with pytest.raises(RuntimeError):
        run_recovery(demo)

    assert saved_journal(demo)["status"] == "LOCAL_COMMIT_UNKNOWN"


def test_unexpected_state_requires_review(demo):
    journal = saved_journal(demo)
    journal["status"] = "DELETION_UNKNOWN"

    demo["journal_path"].write_text(
        json.dumps(journal),
        encoding="utf-8",
    )

    result = run_recovery(demo)

    assert result["status"] == "NEEDS_MANUAL_REVIEW"
    demo["verify"].assert_not_awaited()


@pytest.mark.parametrize(
    ("old_present", "new_valid", "expected"),
    [
        (False, True, "REMOTE_CUTOVER_OBSERVED"),
        (True, True, "OLD_DOCUMENT_STILL_PRESENT"),
        (False, False, "NEEDS_MANUAL_REVIEW"),
        (True, False, "NEEDS_MANUAL_REVIEW"),
    ],
)
def test_uncertain_deletion_inspection(demo, monkeypatch, old_present, new_valid, expected):
    journal = saved_journal(demo)
    journal.update(
        {
            "status": "DELETION_UNKNOWN",
            "old_document_id": "doc-old",
            "new_document_id": "doc-new",
        }
    )

    demo["journal_path"].write_text(json.dumps(journal), encoding="utf-8")

    knowledge_id = demo["knowledge_id"]

    old_documents = (
        [
            {
                "id": "doc-old",
                "status": "processed",
                "file_path": f"knowledgeops-{knowledge_id}-v1.txt",
            }
        ]
        if old_present
        else []
    )

    new_documents = [
        {
            "id": "doc-new",
            "status": "processed" if new_valid else "failed",
            "file_path": f"knowledgeops-{knowledge_id}-v2.txt",
        }
    ]

    def handler(request):
        assert request.method == "GET"

        track_id = request.url.path.rsplit("/", 1)[-1]

        if track_id == "insert_old_test":
            documents = old_documents
        elif track_id == "insert_new_test":
            documents = new_documents
        else:
            raise AssertionError("Unexpected tracking ID")

        return httpx.Response(
            200,
            json={"track_id": track_id, "documents": documents},
        )

    monkeypatch.setenv("LIGHTRAG_API_KEY", "test-key")
    monkeypatch.setenv("LIGHTRAG_BASE_URL", "http://127.0.0.1:9621")

    real_client = httpx.AsyncClient

    monkeypatch.setattr(
        release_recovery.httpx,
        "AsyncClient",
        partial(real_client, transport=httpx.MockTransport(handler)),
    )

    result = asyncio.run(
        release_recovery.inspect_uncertain_deletion(
            proposal_id=demo["proposal_id"],
            journal_dir=demo["journal_dir"],
        )
    )

    assert result["status"] == expected

    # Inspection must never modify the release journal.
    assert saved_journal(demo) == journal


def test_uncertain_deletion_network_failure(demo, monkeypatch):
    journal = saved_journal(demo)
    journal.update(
        {
            "status": "DELETION_UNKNOWN",
            "old_document_id": "doc-old",
            "new_document_id": "doc-new",
        }
    )

    demo["journal_path"].write_text(json.dumps(journal), encoding="utf-8")

    monkeypatch.setenv("LIGHTRAG_API_KEY", "test-key")
    monkeypatch.setenv("LIGHTRAG_BASE_URL", "http://127.0.0.1:9621")

    real_client = httpx.AsyncClient

    monkeypatch.setattr(
        release_recovery.httpx,
        "AsyncClient",
        partial(
            real_client,
            transport=httpx.MockTransport(lambda request: httpx.Response(503)),
        ),
    )

    with pytest.raises(httpx.HTTPStatusError):
        asyncio.run(
            release_recovery.inspect_uncertain_deletion(
                proposal_id=demo["proposal_id"],
                journal_dir=demo["journal_dir"],
            )
        )

    assert saved_journal(demo) == journal
