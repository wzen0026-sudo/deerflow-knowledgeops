"""Tests for KnowledgeOps indexing state advancement."""

import asyncio
import json
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

from deerflow.community.knowledgeops.release_workflow import (
    advance_release_indexing,
)


@pytest.mark.parametrize(
    ("remote_status", "expected_status", "journal_status"),
    [
        ("INDEXED", "INDEXED", "INDEXED"),
        ("PROCESSING", "INDEXING", "INDEXING"),
        ("FAILED", "INDEX_FAILED", "INDEX_FAILED"),
        ("NOT_FOUND", "WAITING_FOR_VISIBILITY", "SUBMITTED"),
        ("UNKNOWN", "NEEDS_REVIEW", "NEEDS_REVIEW"),
    ],
)
def test_indexing_advancement(
    tmp_path,
    remote_status,
    expected_status,
    journal_status,
):
    """Verify indexing states without real network requests."""

    proposal_id = str(uuid4())
    knowledge_id = str(uuid4())
    track_id = "insert_test_001"

    journal = {
        "knowledge_id": knowledge_id,
        "proposal_id": proposal_id,
        "old_version": 1,
        "new_version": 2,
        "status": "SUBMITTED",
        "receipt": {
            "status": "SUBMITTED",
            "knowledge_id": knowledge_id,
            "version": 2,
            "track_id": track_id,
        },
    }

    path = tmp_path / f"{proposal_id}.json"
    path.write_text(json.dumps(journal), encoding="utf-8")

    with patch(
        "deerflow.community.knowledgeops.lightrag_sync.check_version_indexing_status",
        new_callable=AsyncMock,
    ) as check_status:
        check_status.return_value = {
            "status": remote_status,
            "document_count": 1 if remote_status == "INDEXED" else 0,
        }

        result = asyncio.run(
            advance_release_indexing(
                proposal_id=proposal_id,
                journal_dir=tmp_path,
            )
        )

        check_status.assert_awaited_once_with(track_id)

    saved = json.loads(path.read_text(encoding="utf-8"))

    assert result["status"] == expected_status
    assert saved["status"] == journal_status
    assert saved["last_remote_status"] == remote_status


def test_unknown_submission_requires_reconciliation(tmp_path):
    """Do not automatically retry an uncertain submission."""

    proposal_id = str(uuid4())
    path = tmp_path / f"{proposal_id}.json"

    path.write_text(
        json.dumps(
            {
                "proposal_id": proposal_id,
                "status": "SUBMISSION_UNKNOWN",
                "receipt": None,
            }
        ),
        encoding="utf-8",
    )

    result = asyncio.run(
        advance_release_indexing(
            proposal_id=proposal_id,
            journal_dir=tmp_path,
        )
    )

    assert result["status"] == "NEEDS_RECONCILIATION"


def test_already_indexed_does_not_repeat_request(tmp_path):
    """An indexed journal should return its cached result."""

    proposal_id = str(uuid4())
    knowledge_id = str(uuid4())
    track_id = "insert_test_002"

    path = tmp_path / f"{proposal_id}.json"

    path.write_text(
        json.dumps(
            {
                "proposal_id": proposal_id,
                "knowledge_id": knowledge_id,
                "new_version": 2,
                "status": "INDEXED",
                "receipt": {
                    "status": "SUBMITTED",
                    "knowledge_id": knowledge_id,
                    "version": 2,
                    "track_id": track_id,
                },
            }
        ),
        encoding="utf-8",
    )

    with patch(
        "deerflow.community.knowledgeops.lightrag_sync.check_version_indexing_status",
        new_callable=AsyncMock,
    ) as check_status:
        result = asyncio.run(
            advance_release_indexing(
                proposal_id=proposal_id,
                journal_dir=tmp_path,
            )
        )

        check_status.assert_not_awaited()

    assert result["status"] == "INDEXED"
    assert result["cached"] is True
