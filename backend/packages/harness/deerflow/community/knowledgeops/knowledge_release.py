import fcntl
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from tempfile import mkstemp

from deerflow.community.knowledgeops.knowledge_versions import (
    load_knowledge_history,
)
from deerflow.community.knowledgeops.proposal_store import (
    load_update_proposal,
)


def _save_json_atomic(path: Path, data: dict) -> None:
    """Replace a JSON file atomically within one filesystem."""

    fd, temp_path = mkstemp(
        prefix=".knowledge-",
        suffix=".tmp",
        dir=path.parent,
    )

    try:
        os.fchmod(fd, 0o600)

        with os.fdopen(fd, "w", encoding="utf-8") as file:
            json.dump(data, file, ensure_ascii=False, indent=2)
            file.flush()
            os.fsync(file.fileno())

        os.replace(temp_path, path)

    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def publish_approved_proposal(
    knowledge_id: str,
    proposal_id: str,
    knowledge_dir: str | Path,
    proposal_dir: str | Path,
) -> dict:
    """Publish a locally approved proposal as a new version."""

    knowledge_dir = Path(knowledge_dir)
    proposal_dir = Path(proposal_dir)

    lock_path = knowledge_dir / ".knowledge.lock"

    with lock_path.open("a+") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)

        history = load_knowledge_history(
            knowledge_id,
            knowledge_dir,
        )

        proposal = load_update_proposal(
            proposal_id,
            proposal_dir,
        )

        if proposal.get("status") != "APPROVED" or proposal.get("human_approved") is not True or not proposal.get("reviewer") or not proposal.get("reviewed_at_utc"):
            raise ValueError("Proposal has not completed local review.")

        existing = next(
            (version for version in history["versions"] if version.get("proposal_id") == proposal_id),
            None,
        )

        if existing is not None:
            raise ValueError("Proposal has already been published.")

        if proposal.get("published") is not False:
            raise ValueError("Proposal is already marked as published.")

        current = history["versions"][-1]

        if current["version"] != history["current_version"]:
            raise ValueError("Knowledge version history is inconsistent.")

        if proposal.get("original_knowledge") != current["content"]:
            raise ValueError("Knowledge has changed since the proposal was created.")

        new_content = proposal.get("proposed_knowledge")

        if not isinstance(new_content, str) or not new_content.strip():
            raise ValueError("Proposed knowledge is invalid.")

        new_content = new_content.strip()

        if new_content == current["content"]:
            raise ValueError("Proposed knowledge is unchanged.")

        new_version = history["current_version"] + 1
        published_at = datetime.now(UTC).isoformat()

        history["versions"].append(
            {
                "version": new_version,
                "content": new_content,
                "action": "PUBLISH",
                "created_at_utc": published_at,
                "proposal_id": proposal_id,
                "restored_from_version": None,
            }
        )

        history["current_version"] = new_version

        knowledge_path = knowledge_dir / f"{knowledge_id}.json"
        proposal_path = proposal_dir / f"{proposal_id}.json"

        # The history is the source of truth for published versions.
        _save_json_atomic(knowledge_path, history)

        proposal["published"] = True
        proposal["published_version"] = new_version
        proposal["published_at_utc"] = published_at
        proposal["knowledge_id"] = knowledge_id

        _save_json_atomic(proposal_path, proposal)

    return history


def rollback_knowledge_version(
    knowledge_id: str,
    target_version: int,
    reviewer: str,
    reason: str,
    storage_dir: str | Path,
) -> dict:
    """Create a new version restoring content from an older version."""

    if type(target_version) is not int or target_version < 1:
        raise ValueError("Invalid target version.")

    if not isinstance(reviewer, str) or not reviewer.strip():
        raise ValueError("Reviewer is required.")

    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("Rollback reason is required.")

    storage_dir = Path(storage_dir)
    lock_path = storage_dir / ".knowledge.lock"

    with lock_path.open("a+") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)

        history = load_knowledge_history(
            knowledge_id,
            storage_dir,
        )

        current = history["current_version"]

        if history["versions"][-1]["version"] != current:
            raise ValueError("Knowledge version history is inconsistent.")

        if target_version >= current:
            raise ValueError("Target must be an earlier version.")

        target = next(
            (version for version in history["versions"] if version["version"] == target_version),
            None,
        )

        if target is None:
            raise ValueError("Target version does not exist.")

        new_version = current + 1

        history["versions"].append(
            {
                "version": new_version,
                "content": target["content"],
                "action": "ROLLBACK",
                "created_at_utc": datetime.now(UTC).isoformat(),
                "proposal_id": None,
                "restored_from_version": target_version,
                "reviewer": reviewer.strip(),
                "reason": reason.strip(),
            }
        )

        history["current_version"] = new_version

        path = storage_dir / f"{knowledge_id}.json"
        _save_json_atomic(path, history)

    return history
