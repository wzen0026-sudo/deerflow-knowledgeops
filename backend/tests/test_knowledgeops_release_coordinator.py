"""Regression tests for KnowledgeOps release verification."""

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from deerflow.community.knowledgeops.release_coordinator import (
    verify_published_release,
)


@pytest.mark.parametrize(
    ("old_status", "new_status", "expect_success"),
    [
        ("NOT_FOUND", "INDEXED", True),
        ("INDEXED", "INDEXED", False),
        ("NOT_FOUND", "FAILED", False),
    ],
)
def test_release_verification(
    tmp_path,
    old_status,
    new_status,
    expect_success,
):
    """Only accept a release when old is gone and new is indexed."""

    knowledge_id = "demo-knowledge"
    proposal_id = "demo-proposal"

    history = {
        "current_version": 2,
        "versions": [
            {
                "version": 1,
                "content": "Silver Otter 619",
            },
            {
                "version": 2,
                "content": "Copper Fox 853",
                "proposal_id": proposal_id,
            },
        ],
    }

    proposal = {
        "status": "APPROVED",
        "human_approved": True,
        "published": True,
        "knowledge_id": knowledge_id,
        "published_version": 2,
        "original_knowledge": "Silver Otter 619",
        "proposed_knowledge": "Copper Fox 853",
    }

    prefix = "deerflow.community.knowledgeops.release_coordinator"

    with (
        patch(
            f"{prefix}.load_knowledge_history",
            return_value=history,
        ),
        patch(
            f"{prefix}.load_update_proposal",
            return_value=proposal,
        ),
        patch(
            f"{prefix}.check_version_indexing_status",
            new_callable=AsyncMock,
        ) as check_status,
    ):
        check_status.side_effect = [
            {
                "status": old_status,
                "document_count": 0 if old_status == "NOT_FOUND" else 1,
            },
            {
                "status": new_status,
                "document_count": 1,
            },
        ]

        async def run_verification():
            return await verify_published_release(
                knowledge_id=knowledge_id,
                proposal_id=proposal_id,
                knowledge_dir=tmp_path / "knowledge",
                proposal_dir=tmp_path / "proposals",
                old_track_id="old-track",
                new_track_id="new-track",
            )

        if expect_success:
            result = asyncio.run(run_verification())

            assert result["status"] == "VERIFIED"
            assert result["current_version"] == 2
        else:
            with pytest.raises(RuntimeError):
                asyncio.run(run_verification())

        assert check_status.await_count == 2
