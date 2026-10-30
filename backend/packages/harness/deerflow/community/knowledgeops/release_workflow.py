"""Coordinate the indexing stage of a KnowledgeOps release."""

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from tempfile import mkstemp
from uuid import UUID

from deerflow.community.knowledgeops.knowledge_versions import (
    load_knowledge_history,
)
from deerflow.community.knowledgeops.lightrag_sync import (
    submit_version_for_indexing,
)
from deerflow.community.knowledgeops.proposal_store import (
    load_update_proposal,
)


def _save_journal_atomic(path: Path, data: dict) -> None:
    """Update a release journal without partially overwriting it."""
    fd, temp_path = mkstemp(
        prefix=".release-",
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


async def start_release_indexing(
    *,
    knowledge_id: str,
    proposal_id: str,
    knowledge_dir: str | Path,
    proposal_dir: str | Path,
    journal_dir: str | Path,
) -> dict:
    """Start an approved release's indexing stage exactly once locally.

    A submission with an uncertain result is never retried automatically.
    This function does not remove old documents or mark knowledge published.
    """

    # Validate identifiers before using them as filesystem paths.
    for identifier in (knowledge_id, proposal_id):
        if not isinstance(identifier, str):
            raise ValueError("Invalid release identifier.")
        try:
            if str(UUID(identifier)) != identifier:
                raise ValueError("Identifier must be a canonical UUID.")
        except ValueError as exc:
            raise ValueError("Invalid release identifier.") from exc

    history = load_knowledge_history(
        knowledge_id,
        knowledge_dir,
    )

    proposal = load_update_proposal(
        proposal_id,
        proposal_dir,
    )

    # Step 1: Require an approved but unpublished proposal.
    if proposal.get("status") != "APPROVED" or proposal.get("human_approved") is not True or proposal.get("published") is not False or not proposal.get("reviewer") or not proposal.get("reviewed_at_utc"):
        raise ValueError("Proposal is not approved for release.")

    current_version = history["current_version"]
    current = history["versions"][-1]

    if current["version"] != current_version:
        raise ValueError("Knowledge version history is inconsistent.")

    if proposal.get("original_knowledge") != current["content"]:
        raise ValueError("Knowledge changed since proposal approval.")

    new_content = proposal.get("proposed_knowledge")

    if not isinstance(new_content, str) or not new_content.strip():
        raise ValueError("Proposed knowledge is invalid.")

    new_content = new_content.strip()

    if new_content == current["content"]:
        raise ValueError("Proposed knowledge is unchanged.")

    new_version = current_version + 1

    # Step 2: Create an exclusive journal before the API request.
    journal_folder = Path(journal_dir)
    journal_folder.mkdir(parents=True, exist_ok=True, mode=0o700)

    journal_path = journal_folder / f"{proposal_id}.json"

    journal = {
        "schema_version": "1.0",
        "knowledge_id": knowledge_id,
        "proposal_id": proposal_id,
        "old_version": current_version,
        "new_version": new_version,
        "status": "SUBMISSION_PENDING",
        "started_at_utc": datetime.now(UTC).isoformat(),
        "receipt": None,
    }

    try:
        fd = os.open(
            journal_path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
    except FileExistsError as exc:
        raise RuntimeError("Release journal already exists. Inspect its state before attempting another submission.") from exc

    with os.fdopen(fd, "w", encoding="utf-8") as file:
        json.dump(journal, file, ensure_ascii=False, indent=2)
        file.flush()
        os.fsync(file.fileno())

    # Step 3: Submit the approved version to LightRAG.
    try:
        receipt = await submit_version_for_indexing(
            knowledge_id=knowledge_id,
            version=new_version,
            content=new_content,
        )
    except Exception as exc:
        journal["status"] = "SUBMISSION_UNKNOWN"
        _save_journal_atomic(journal_path, journal)

        raise RuntimeError("Release submission outcome is uncertain. Inspect LightRAG and the journal before retrying.") from exc

    # Step 4: Persist the successful submission receipt.
    journal["status"] = "SUBMITTED"
    journal["receipt"] = receipt

    _save_journal_atomic(journal_path, journal)

    return {
        "status": "SUBMITTED",
        "knowledge_id": knowledge_id,
        "proposal_id": proposal_id,
        "version": new_version,
        "track_id": receipt["track_id"],
        "journal_path": str(journal_path),
    }
