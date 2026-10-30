"""Verify LightRAG retirement and finalize a local knowledge release."""

import json
import os
import re
from datetime import UTC, datetime
from pathlib import Path

import httpx
from dotenv import find_dotenv, load_dotenv

from deerflow.community.knowledgeops.knowledge_release import (
    publish_approved_proposal,
)
from deerflow.community.knowledgeops.knowledge_versions import (
    load_knowledge_history,
)
from deerflow.community.knowledgeops.proposal_store import (
    load_update_proposal,
)
from deerflow.community.knowledgeops.release_workflow import (
    _save_journal_atomic,
)


async def finalize_release_after_retirement(
    *,
    proposal_id: str,
    journal_dir: str | Path,
    knowledge_dir: str | Path,
    proposal_dir: str | Path,
) -> dict:
    """Finalize only after old knowledge is absent and new knowledge is indexed.

    Never submits documents or issues deletion requests.
    """

    # Validate the identifier using the existing proposal loader.
    proposal = load_update_proposal(proposal_id, proposal_dir)

    journal_path = Path(journal_dir) / f"{proposal_id}.json"

    journal = json.loads(journal_path.read_text(encoding="utf-8"))

    if not isinstance(journal, dict):
        raise ValueError("Invalid release journal.")

    if journal.get("proposal_id") != proposal_id or journal.get("status") != "DELETION_STARTED" or journal.get("deletion_response", {}).get("status") != "deletion_started":
        raise ValueError("Release is not ready for finalization.")

    knowledge_id = journal.get("knowledge_id")
    old_version = journal.get("old_version")
    new_version = journal.get("new_version")

    old_track_id = journal.get("old_track_id")
    receipt = journal.get("receipt")

    if type(old_version) is not int or type(new_version) is not int or new_version != old_version + 1 or not isinstance(receipt, dict):
        raise ValueError("Invalid release version information.")

    new_track_id = receipt.get("track_id")

    for track_id in (old_track_id, new_track_id):
        if not isinstance(track_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", track_id):
            raise ValueError("Invalid LightRAG tracking ID.")

    if old_track_id == new_track_id or receipt.get("knowledge_id") != knowledge_id or receipt.get("version") != new_version or receipt.get("status") != "SUBMITTED":
        raise ValueError("Release receipt is inconsistent.")

    old_doc_id = journal.get("old_document_id")
    new_doc_id = journal.get("new_document_id")

    if not isinstance(old_doc_id, str) or not isinstance(new_doc_id, str) or not old_doc_id.startswith("doc-") or not new_doc_id.startswith("doc-") or old_doc_id == new_doc_id:
        raise ValueError("Invalid document identity mapping.")

    history = load_knowledge_history(knowledge_id, knowledge_dir)

    if (
        history["current_version"] != old_version
        or history["versions"][-1]["content"] != proposal.get("original_knowledge")
        or proposal.get("status") != "APPROVED"
        or proposal.get("human_approved") is not True
        or proposal.get("published") is not False
        or not proposal.get("reviewer")
        or not proposal.get("reviewed_at_utc")
    ):
        raise ValueError("Local publication state is inconsistent.")

    load_dotenv(find_dotenv(usecwd=True))

    api_key = os.getenv("LIGHTRAG_API_KEY")
    base_url = os.getenv(
        "LIGHTRAG_BASE_URL",
        "http://127.0.0.1:9621",
    ).rstrip("/")

    if not api_key:
        raise RuntimeError("LIGHTRAG_API_KEY is missing.")

    async with httpx.AsyncClient(timeout=30) as client:

        async def fetch_documents(track_id: str) -> list:
            response = await client.get(
                f"{base_url}/documents/track_status/{track_id}",
                headers={"X-API-Key": api_key},
            )
            response.raise_for_status()

            payload = response.json()

            if not isinstance(payload, dict) or payload.get("track_id") != track_id or not isinstance(payload.get("documents"), list):
                raise RuntimeError("Invalid LightRAG tracking response.")

            return payload["documents"]

        old_documents = await fetch_documents(old_track_id)
        new_documents = await fetch_documents(new_track_id)

    # Do not publish while the old document remains visible.
    if old_documents:
        if len(old_documents) == 1 and isinstance(old_documents[0], dict) and old_documents[0].get("id") == old_doc_id:
            return {
                "status": "WAITING_FOR_DELETION",
                "proposal_id": proposal_id,
            }

        raise RuntimeError("Unexpected old document identity.")

    # Verify the exact new document is still indexed.
    expected_source = f"knowledgeops-{knowledge_id}-v{new_version}.txt"

    if len(new_documents) != 1:
        raise RuntimeError("Expected exactly one new indexed document.")

    new_document = new_documents[0]

    if not isinstance(new_document, dict) or new_document.get("id") != new_doc_id or new_document.get("file_path") != expected_source or new_document.get("status") != "processed":
        raise RuntimeError("New document identity or indexing state is invalid.")

    # Persist intent before modifying local publication files.
    journal["status"] = "FINALIZING"
    journal["retirement_verified_at_utc"] = datetime.now(UTC).isoformat()

    _save_journal_atomic(journal_path, journal)

    try:
        updated_history = publish_approved_proposal(
            knowledge_id=knowledge_id,
            proposal_id=proposal_id,
            knowledge_dir=knowledge_dir,
            proposal_dir=proposal_dir,
        )
    except Exception as exc:
        # Publication may have partially committed.
        # Never automatically repeat it.
        journal["status"] = "LOCAL_COMMIT_UNKNOWN"
        _save_journal_atomic(journal_path, journal)

        raise RuntimeError("Local publication outcome requires reconciliation.") from exc

    if updated_history["current_version"] != new_version:
        journal["status"] = "LOCAL_COMMIT_UNKNOWN"
        _save_journal_atomic(journal_path, journal)

        raise RuntimeError("Published version is inconsistent.")

    journal["status"] = "PUBLISHED"
    journal["published_version"] = new_version
    journal["completed_at_utc"] = datetime.now(UTC).isoformat()

    _save_journal_atomic(journal_path, journal)

    return {
        "status": "PUBLISHED",
        "knowledge_id": knowledge_id,
        "proposal_id": proposal_id,
        "current_version": new_version,
        "old_document_status": "REMOVED",
        "new_document_status": "INDEXED",
    }
