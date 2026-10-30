"""Tests for the KnowledgeOps command-line interface."""

import json
import sys
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from deerflow.community.knowledgeops import release_cli


@pytest.fixture
def case(tmp_path):
    proposal_id = str(uuid4())
    knowledge_id = str(uuid4())

    base = tmp_path / "data"
    journals = base / "journals"
    journals.mkdir(parents=True)

    journal = {
        "proposal_id": proposal_id,
        "knowledge_id": knowledge_id,
        "status": "DELETION_UNKNOWN",
        "old_version": 1,
        "new_version": 2,
    }

    (journals / f"{proposal_id}.json").write_text(
        json.dumps(journal),
        encoding="utf-8",
    )

    return {
        "base": base,
        "proposal_id": proposal_id,
        "knowledge_id": knowledge_id,
    }


def call_cli(monkeypatch, capsys, case, action, extra=None):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "knowledgeops",
            action,
            "--proposal-id",
            case["proposal_id"],
            "--data-dir",
            str(case["base"]),
            *(extra or []),
        ],
    )

    assert release_cli.main() == 0
    return json.loads(capsys.readouterr().out)


def test_status_command(monkeypatch, capsys, case):
    result = call_cli(monkeypatch, capsys, case, "status")

    assert result["status"] == "DELETION_UNKNOWN"
    assert result["old_version"] == 1
    assert result["new_version"] == 2


@pytest.mark.parametrize(
    ("action", "function_name"),
    [
        ("start", "start_release_indexing"),
        ("advance", "advance_release_pipeline"),
        ("inspect", "advance_release_pipeline"),
        ("recover", "recover_completed_local_release"),
    ],
)
def test_command_dispatch(monkeypatch, capsys, case, action, function_name):
    fake = AsyncMock(return_value={"status": f"MOCK_{action.upper()}"})
    monkeypatch.setattr(release_cli, function_name, fake)

    extra = ["--knowledge-id", case["knowledge_id"]] if action == "start" else []

    result = call_cli(monkeypatch, capsys, case, action, extra)

    assert result["status"] == f"MOCK_{action.upper()}"
    fake.assert_awaited_once()

    arguments = fake.await_args.kwargs

    if action == "start":
        assert arguments["knowledge_id"] == case["knowledge_id"]

    if action == "advance":
        assert arguments["allow_retirement"] is False

    if action == "inspect":
        assert arguments["inspect_deletion"] is True


def test_start_requires_knowledge_id(monkeypatch, case):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "knowledgeops",
            "start",
            "--proposal-id",
            case["proposal_id"],
            "--data-dir",
            str(case["base"]),
        ],
    )

    with pytest.raises(SystemExit) as error:
        release_cli.main()

    assert error.value.code == 1
