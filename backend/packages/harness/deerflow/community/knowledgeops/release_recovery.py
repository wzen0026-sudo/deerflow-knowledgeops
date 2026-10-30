"""Recover a release whose local publication may have been interrupted."""

import fcntl
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

from deerflow.community.knowledgeops.knowledge_versions import (
    load_knowledge_history,
)
from deerflow.community.knowledgeops.proposal_store import (
    load_update_proposal,
)
from deerflow.community.knowledgeops.release_coordinator import (
    verify_published_release,
)
from deerflow.community.knowledgeops.release_workflow import (
    _save_journal_atomic,
)


def _local_commit_complete(history: dict, proposal: dict, journal: dict) -> bool:
    """Check whether both local publication records were committed."""

    versions = history.get("versions")

    if not isinstance(versions, list) or len(versions) < 2:
        return False

    previous, current = versions[-2:]

    return (
        history.get("current_version") == journal["new_version"]
        and previous.get("version") == journal["old_version"]
        and current.get("version") == journal["new_version"]
        and current.get("proposal_id") == journal["proposal_id"]
        and previous.get("content") == proposal.get("original_knowledge")
        and current.get("content") == proposal.get("proposed_knowledge")
        and proposal.get("status") == "APPROVED"
        and proposal.get("human_approved") is True
        and proposal.get("published") is True
        and proposal.get("published_version") == journal["new_version"]
        and proposal.get("knowledge_id") == journal["knowledge_id"]
    )


async def recover_completed_local_release(
    *,
    proposal_id: str,
    journal_dir: str | Path,
    knowledge_dir: str | Path,
    proposal_dir: str | Path,
) -> dict:
    """Repair a journal only when publication is already fully committed.

    Never creates versions, submits documents, or deletes knowledge.
    """

    try:
        if str(UUID(proposal_id)) != proposal_id:
            raise ValueError("Noncanonical proposal ID")
    except (TypeError, ValueError, AttributeError) as exc:
        raise ValueError("Invalid proposal ID") from exc

    journal_folder = Path(journal_dir)
    journal_path = journal_folder / f"{proposal_id}.json"

    journal = json.loads(journal_path.read_text(encoding="utf-8"))

    if not isinstance(journal, dict) or journal.get("proposal_id") != proposal_id:
        raise ValueError("Invalid release journal")

    status = journal.get("status")

    if status == "PUBLISHED":
        return {"status": "ALREADY_PUBLISHED"}

    if status not in {"FINALIZING", "LOCAL_COMMIT_UNKNOWN"}:
        return {
            "status": "NEEDS_MANUAL_REVIEW",
            "reason": f"Unsupported recovery state: {status}",
        }

    knowledge_id = journal.get("knowledge_id")

    try:
        if str(UUID(knowledge_id)) != knowledge_id:
            raise ValueError("Noncanonical knowledge ID")
    except (TypeError, ValueError, AttributeError) as exc:
        raise ValueError("Invalid knowledge ID") from exc

    old_version = journal.get("old_version")
    new_version = journal.get("new_version")
    receipt = journal.get("receipt")

    if (
        type(old_version) is not int
        or type(new_version) is not int
        or new_version != old_version + 1
        or not isinstance(receipt, dict)
        or receipt.get("knowledge_id") != knowledge_id
        or receipt.get("version") != new_version
        or receipt.get("status") != "SUBMITTED"
    ):
        raise ValueError("Inconsistent release metadata")

    old_track_id = journal.get("old_track_id")
    new_track_id = receipt.get("track_id")

    if not isinstance(old_track_id, str) or not isinstance(new_track_id, str) or not old_track_id or not new_track_id or old_track_id == new_track_id:
        raise ValueError("Invalid release tracking IDs")

    def read_local():
        history = load_knowledge_history(knowledge_id, knowledge_dir)
        proposal = load_update_proposal(proposal_id, proposal_dir)
        return history, proposal

    history, proposal = read_local()

    if not _local_commit_complete(history, proposal, journal):
        return {
            "status": "PARTIAL_LOCAL_COMMIT",
            "reason": ("Local publication files are not fully consistent. No automatic replay will be attempted."),
        }

    # Verify that V1 is absent and V2 remains indexed.
    # Any remote failure prevents the journal repair.
    verification = await verify_published_release(
        knowledge_id=knowledge_id,
        proposal_id=proposal_id,
        knowledge_dir=knowledge_dir,
        proposal_dir=proposal_dir,
        old_track_id=old_track_id,
        new_track_id=new_track_id,
    )

    if verification.get("status") != "VERIFIED":
        raise RuntimeError("Remote publication verification failed")

    # Use the same local release lock as the orchestrator.
    lock_path = journal_folder / f".{knowledge_id}.release.lock"
    fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)

    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return {"status": "RELEASE_BUSY"}

        # Coordinate with the knowledge history's own lock.
        knowledge_lock = Path(knowledge_dir) / ".knowledge.lock"

        with knowledge_lock.open("a+") as lock_file:
            try:
                fcntl.flock(
                    lock_file.fileno(),
                    fcntl.LOCK_EX | fcntl.LOCK_NB,
                )
            except BlockingIOError:
                return {"status": "RELEASE_BUSY"}

            current_journal = json.loads(journal_path.read_text(encoding="utf-8"))

            if current_journal != journal:
                return {
                    "status": "NEEDS_MANUAL_REVIEW",
                    "reason": "Journal changed during recovery.",
                }

            history, proposal = read_local()

            if not _local_commit_complete(history, proposal, journal):
                return {
                    "status": "NEEDS_MANUAL_REVIEW",
                    "reason": "Local records changed during recovery.",
                }

            journal["status"] = "PUBLISHED"
            journal["published_version"] = new_version
            journal["recovered_at_utc"] = datetime.now(UTC).isoformat()

            _save_journal_atomic(journal_path, journal)

    finally:
        os.close(fd)

    return {
        "status": "RECOVERED",
        "knowledge_id": knowledge_id,
        "proposal_id": proposal_id,
        "current_version": new_version,
        "journal_status": "PUBLISHED",
    }
