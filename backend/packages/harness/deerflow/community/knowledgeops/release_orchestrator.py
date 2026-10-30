"""Unified state-driven KnowledgeOps release orchestration."""

import fcntl
import json
import os
from pathlib import Path
from uuid import UUID

from deerflow.community.knowledgeops.release_coordinator import (
    verify_published_release,
)
from deerflow.community.knowledgeops.release_finalization import (
    finalize_release_after_retirement,
)
from deerflow.community.knowledgeops.release_recovery import (
    inspect_uncertain_deletion,
)
from deerflow.community.knowledgeops.release_retirement import (
    request_old_version_retirement,
)
from deerflow.community.knowledgeops.release_workflow import (
    advance_release_indexing,
)


async def _advance_release_pipeline_unlocked(
    *,
    proposal_id: str,
    journal_dir: str | Path,
    knowledge_dir: str | Path,
    proposal_dir: str | Path,
    old_track_id: str | None = None,
    allow_retirement: bool = False,
    inspect_deletion: bool = False,
) -> dict:
    """Advance one stage of an existing release.

    Never resubmit uncertain requests. Old-version deletion requires
    explicit permission. Each call advances at most one stage.
    """

    try:
        if not isinstance(proposal_id, str):
            raise ValueError("Invalid proposal ID.")
        if str(UUID(proposal_id)) != proposal_id:
            raise ValueError("Invalid proposal ID.")
    except (ValueError, TypeError, AttributeError) as exc:
        raise ValueError("Invalid proposal ID.") from exc

    journal_path = Path(journal_dir) / f"{proposal_id}.json"

    if not journal_path.is_file():
        raise FileNotFoundError("Release journal missing. Start indexing first.")

    journal = json.loads(journal_path.read_text(encoding="utf-8"))

    if not isinstance(journal, dict) or journal.get("proposal_id") != proposal_id:
        raise ValueError("Invalid release journal.")

    status = journal.get("status")

    # Stage 1: Check indexing without resubmitting the document.
    if status in {"SUBMITTED", "INDEXING"}:
        result = await advance_release_indexing(
            proposal_id=proposal_id,
            journal_dir=journal_dir,
        )

        if result["status"] == "INDEXED":
            return {
                "status": "READY_FOR_RETIREMENT",
                "proposal_id": proposal_id,
            }

        return result

    # Stage 2: Require explicit authorization before deletion.
    if status == "INDEXED":
        if allow_retirement is not True:
            return {
                "status": "RETIREMENT_APPROVAL_REQUIRED",
                "proposal_id": proposal_id,
            }

        if not old_track_id:
            raise ValueError("Old tracking ID is required.")

        return await request_old_version_retirement(
            proposal_id=proposal_id,
            old_track_id=old_track_id,
            journal_dir=journal_dir,
            knowledge_dir=knowledge_dir,
            proposal_dir=proposal_dir,
        )

    # Stage 3: Confirm removal and publish the local version.
    if status == "DELETION_STARTED":
        return await finalize_release_after_retirement(
            proposal_id=proposal_id,
            journal_dir=journal_dir,
            knowledge_dir=knowledge_dir,
            proposal_dir=proposal_dir,
        )

    # Stage 4: Verify an already completed publication.
    if status == "PUBLISHED":
        receipt = journal.get("receipt")

        if not isinstance(receipt, dict):
            raise ValueError("Missing release receipt.")

        recorded_old_track = journal.get("old_track_id")
        recorded_new_track = receipt.get("track_id")
        knowledge_id = journal.get("knowledge_id")

        if not all(
            isinstance(value, str) and value
            for value in (
                recorded_old_track,
                recorded_new_track,
                knowledge_id,
            )
        ):
            raise ValueError("Incomplete published release record.")

        return await verify_published_release(
            knowledge_id=knowledge_id,
            proposal_id=proposal_id,
            knowledge_dir=knowledge_dir,
            proposal_dir=proposal_dir,
            old_track_id=recorded_old_track,
            new_track_id=recorded_new_track,
        )

    # Stage 5: Read-only inspection of an uncertain deletion.
    if (
        status
        in {
            "DELETION_REQUESTING",
            "DELETION_UNKNOWN",
            "DELETION_NEEDS_REVIEW",
        }
        and inspect_deletion is True
    ):
        inspection = await inspect_uncertain_deletion(
            proposal_id=proposal_id,
            journal_dir=journal_dir,
        )

        return {
            "status": "NEEDS_RECONCILIATION",
            "journal_status": status,
            "inspection": inspection,
            "reason": ("Read-only inspection completed. No deletion, retry, or publication was performed."),
        }

    # Unexpected or uncertain states must never trigger blind retries.
    return {
        "status": "NEEDS_RECONCILIATION",
        "journal_status": status,
        "reason": ("Manual investigation is required before proceeding."),
    }


async def advance_release_pipeline(
    *,
    proposal_id: str,
    journal_dir: str | Path,
    knowledge_dir: str | Path,
    proposal_dir: str | Path,
    old_track_id: str | None = None,
    allow_retirement: bool = False,
    inspect_deletion: bool = False,
) -> dict:
    """Advance one release stage with a per-knowledge process lock.

    The lock covers this orchestrator entry point on a local
    Linux filesystem. It is not a distributed lock.
    """

    # Validate the proposal ID before constructing a file path.
    try:
        if not isinstance(proposal_id, str) or str(UUID(proposal_id)) != proposal_id:
            raise ValueError("Invalid proposal ID.")
    except (ValueError, TypeError, AttributeError) as exc:
        raise ValueError("Invalid proposal ID.") from exc

    folder = Path(journal_dir)
    journal_path = folder / f"{proposal_id}.json"

    journal = json.loads(journal_path.read_text(encoding="utf-8"))

    if not isinstance(journal, dict) or journal.get("proposal_id") != proposal_id:
        raise ValueError("Invalid release journal.")

    knowledge_id = journal.get("knowledge_id")

    try:
        if not isinstance(knowledge_id, str) or str(UUID(knowledge_id)) != knowledge_id:
            raise ValueError("Invalid knowledge ID.")
    except (ValueError, TypeError, AttributeError) as exc:
        raise ValueError("Invalid knowledge ID.") from exc

    # Different proposals for the same knowledge share one lock.
    lock_path = folder / f".{knowledge_id}.release.lock"

    fd = os.open(
        lock_path,
        os.O_CREAT | os.O_RDWR,
        0o600,
    )

    locked = False

    try:
        try:
            # Do not wait indefinitely if another release is running.
            fcntl.flock(
                fd,
                fcntl.LOCK_EX | fcntl.LOCK_NB,
            )
            locked = True

        except BlockingIOError:
            return {
                "status": "RELEASE_BUSY",
                "knowledge_id": knowledge_id,
                "reason": ("Another release operation is already running for this knowledge."),
            }

        # Execute the existing orchestration under the lock.
        return await _advance_release_pipeline_unlocked(
            proposal_id=proposal_id,
            journal_dir=journal_dir,
            knowledge_dir=knowledge_dir,
            proposal_dir=proposal_dir,
            old_track_id=old_track_id,
            allow_retirement=allow_retirement,
            inspect_deletion=inspect_deletion,
        )

    finally:
        if locked:
            fcntl.flock(
                fd,
                fcntl.LOCK_UN,
            )

        os.close(fd)
