"""Tests for KnowledgeOps release finalization."""

import asyncio
import json
from functools import partial
from uuid import uuid4

import httpx
import pytest

from deerflow.community.knowledgeops import release_finalization


@pytest.fixture
def demo(tmp_path, monkeypatch):
    knowledge_id = str(uuid4())
    proposal_id = str(uuid4())

    journal = {
        "knowledge_id": knowledge_id,
        "proposal_id": proposal_id,
        "old_version": 1,
        "new_version": 2,
        "old_track_id": "old-track",
        "old_document_id": "doc-old",
        "new_document_id": "doc-new",
        "status": "DELETION_STARTED",
        "deletion_response": {
            "status": "deletion_started",
        },
        "receipt": {
            "status": "SUBMITTED",
            "knowledge_id": knowledge_id,
            "version": 2,
            "track_id": "new-track",
        },
    }

    journal_path = tmp_path / f"{proposal_id}.json"
    journal_path.write_text(
        json.dumps(journal),
        encoding="utf-8",
    )

    history = {
        "current_version": 1,
        "versions": [
            {
                "version": 1,
                "content": "Old knowledge",
            }
        ],
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

    state = {
        "proposal_id": proposal_id,
        "journal_path": journal_path,
        "folder": tmp_path,
        "old_documents": [],
        "new_documents": [
            {
                "id": "doc-new",
                "status": "processed",
                "file_path": (f"knowledgeops-{knowledge_id}-v2.txt"),
            }
        ],
        "publish_calls": 0,
        "fail_publish": False,
        "http_error": False,
    }

    monkeypatch.setattr(
        release_finalization,
        "load_knowledge_history",
        lambda *args: history,
    )
    monkeypatch.setattr(
        release_finalization,
        "load_update_proposal",
        lambda *args: proposal,
    )

    def fake_publish(**kwargs):
        state["publish_calls"] += 1

        if state["fail_publish"]:
            raise OSError("Simulated local write failure")

        return {"current_version": 2}

    monkeypatch.setattr(
        release_finalization,
        "publish_approved_proposal",
        fake_publish,
    )

    monkeypatch.setenv("LIGHTRAG_API_KEY", "test-key")
    monkeypatch.setenv(
        "LIGHTRAG_BASE_URL",
        "http://127.0.0.1:9621",
    )

    def handler(request):
        assert request.method == "GET"
        assert request.headers["X-API-Key"] == "test-key"

        if state["http_error"]:
            return httpx.Response(503)

        if request.url.path.endswith("/old-track"):
            track_id = "old-track"
            documents = state["old_documents"]
        elif request.url.path.endswith("/new-track"):
            track_id = "new-track"
            documents = state["new_documents"]
        else:
            raise AssertionError("Unexpected API request")

        return httpx.Response(
            200,
            json={
                "track_id": track_id,
                "documents": documents,
            },
        )

    transport = httpx.MockTransport(handler)
    real_client = httpx.AsyncClient

    monkeypatch.setattr(
        release_finalization.httpx,
        "AsyncClient",
        partial(real_client, transport=transport),
    )

    return state


def run_finalization(demo):
    return asyncio.run(
        release_finalization.finalize_release_after_retirement(
            proposal_id=demo["proposal_id"],
            journal_dir=demo["folder"],
            knowledge_dir=demo["folder"] / "knowledge",
            proposal_dir=demo["folder"] / "proposals",
        )
    )


def saved_journal(demo):
    return json.loads(demo["journal_path"].read_text(encoding="utf-8"))


def test_successful_finalization(demo):
    result = run_finalization(demo)

    assert result["status"] == "PUBLISHED"
    assert result["current_version"] == 2
    assert demo["publish_calls"] == 1

    journal = saved_journal(demo)

    assert journal["status"] == "PUBLISHED"
    assert journal["published_version"] == 2

    # The same release must not be published twice.
    with pytest.raises(ValueError, match="not ready"):
        run_finalization(demo)

    assert demo["publish_calls"] == 1


def test_old_document_still_exists(demo):
    demo["old_documents"] = [{"id": "doc-old", "status": "processed"}]

    result = run_finalization(demo)

    assert result["status"] == "WAITING_FOR_DELETION"
    assert demo["publish_calls"] == 0
    assert saved_journal(demo)["status"] == "DELETION_STARTED"


def test_new_document_is_invalid(demo):
    demo["new_documents"][0]["status"] = "failed"

    with pytest.raises(RuntimeError, match="indexing state"):
        run_finalization(demo)

    assert demo["publish_calls"] == 0
    assert saved_journal(demo)["status"] == "DELETION_STARTED"


def test_local_publication_failure_is_recorded(demo):
    demo["fail_publish"] = True

    with pytest.raises(RuntimeError, match="reconciliation"):
        run_finalization(demo)

    assert demo["publish_calls"] == 1
    assert saved_journal(demo)["status"] == "LOCAL_COMMIT_UNKNOWN"


def test_remote_failure_does_not_publish(demo):
    demo["http_error"] = True

    with pytest.raises(httpx.HTTPStatusError):
        run_finalization(demo)

    assert demo["publish_calls"] == 0
    assert saved_journal(demo)["status"] == "DELETION_STARTED"
