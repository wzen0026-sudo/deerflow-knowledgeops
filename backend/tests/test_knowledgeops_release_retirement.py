"""Tests for controlled LightRAG document retirement."""

import asyncio
import json
from functools import partial
from uuid import uuid4

import httpx
import pytest

from deerflow.community.knowledgeops import release_retirement


@pytest.fixture
def demo(tmp_path, monkeypatch):
    knowledge_id = str(uuid4())
    proposal_id = str(uuid4())

    journal_dir = tmp_path / "journals"
    journal_dir.mkdir()

    journal_path = journal_dir / f"{proposal_id}.json"

    journal = {
        "knowledge_id": knowledge_id,
        "proposal_id": proposal_id,
        "old_version": 1,
        "new_version": 2,
        "status": "INDEXED",
        "receipt": {
            "status": "SUBMITTED",
            "knowledge_id": knowledge_id,
            "version": 2,
            "track_id": "insert_new_test",
        },
    }

    journal_path.write_text(json.dumps(journal), encoding="utf-8")

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

    monkeypatch.setattr(
        release_retirement,
        "load_knowledge_history",
        lambda *args: history,
    )
    monkeypatch.setattr(
        release_retirement,
        "load_update_proposal",
        lambda *args: proposal,
    )

    monkeypatch.setenv("LIGHTRAG_API_KEY", "test-key")
    monkeypatch.setenv("LIGHTRAG_BASE_URL", "http://127.0.0.1:9621")

    return {
        "knowledge_id": knowledge_id,
        "proposal_id": proposal_id,
        "journal_dir": journal_dir,
        "journal_path": journal_path,
        "history": history,
        "proposal": proposal,
        "delete_status": "deletion_started",
        "network_failure": False,
        "bad_new_source": False,
        "delete_calls": 0,
    }


@pytest.fixture
def mock_api(demo, monkeypatch):
    real_client = httpx.AsyncClient

    def handler(request):
        assert request.headers["X-API-Key"] == "test-key"

        if request.method == "GET":
            track_id = request.url.path.rsplit("/", 1)[-1]

            assert track_id in {
                "insert_old_test",
                "insert_new_test",
            }

            version = 1 if track_id == "insert_old_test" else 2

            source = f"knowledgeops-{demo['knowledge_id']}-v{version}.txt"

            if version == 2 and demo["bad_new_source"]:
                source = "wrong-source.txt"

            return httpx.Response(
                200,
                json={
                    "track_id": track_id,
                    "documents": [
                        {
                            "id": f"doc-version-{version}",
                            "status": "processed",
                            "file_path": source,
                        }
                    ],
                },
            )

        if request.method == "DELETE" and request.url.path == "/documents/delete_document":
            demo["delete_calls"] += 1

            body = json.loads(request.content)
            assert body["doc_ids"] == ["doc-version-1"]
            assert body["delete_file"] is False

            if demo["network_failure"]:
                raise httpx.ConnectError(
                    "Simulated network failure",
                    request=request,
                )

            return httpx.Response(
                200,
                json={"status": demo["delete_status"]},
            )

        raise AssertionError("Unexpected HTTP request")

    transport = httpx.MockTransport(handler)

    monkeypatch.setattr(
        release_retirement.httpx,
        "AsyncClient",
        partial(real_client, transport=transport),
    )

    return demo


def run_retirement(demo):
    return asyncio.run(
        release_retirement.request_old_version_retirement(
            proposal_id=demo["proposal_id"],
            old_track_id="insert_old_test",
            journal_dir=demo["journal_dir"],
            knowledge_dir=demo["journal_dir"],
            proposal_dir=demo["journal_dir"],
        )
    )


def saved_journal(demo):
    return json.loads(demo["journal_path"].read_text(encoding="utf-8"))


def test_retirement_request_and_duplicate_block(mock_api):
    demo = mock_api

    result = run_retirement(demo)

    assert result["status"] == "DELETION_STARTED"
    assert saved_journal(demo)["status"] == "DELETION_STARTED"
    assert demo["delete_calls"] == 1

    with pytest.raises(ValueError, match="INDEXED"):
        run_retirement(demo)

    assert demo["delete_calls"] == 1


def test_unapproved_proposal_is_blocked(mock_api):
    demo = mock_api
    demo["proposal"]["human_approved"] = False

    with pytest.raises(ValueError):
        run_retirement(demo)

    assert demo["delete_calls"] == 0
    assert saved_journal(demo)["status"] == "INDEXED"


def test_wrong_new_document_is_blocked(mock_api):
    demo = mock_api
    demo["bad_new_source"] = True

    with pytest.raises(RuntimeError, match="verification failed"):
        run_retirement(demo)

    assert demo["delete_calls"] == 0
    assert saved_journal(demo)["status"] == "INDEXED"


def test_network_failure_is_recorded(mock_api):
    demo = mock_api
    demo["network_failure"] = True

    with pytest.raises(RuntimeError, match="uncertain"):
        run_retirement(demo)

    assert demo["delete_calls"] == 1
    assert saved_journal(demo)["status"] == "DELETION_UNKNOWN"


def test_busy_response_requires_review(mock_api):
    demo = mock_api
    demo["delete_status"] = "busy"

    result = run_retirement(demo)

    assert result["status"] == "DELETION_NEEDS_REVIEW"
    assert saved_journal(demo)["status"] == "DELETION_NEEDS_REVIEW"
    assert demo["delete_calls"] == 1
