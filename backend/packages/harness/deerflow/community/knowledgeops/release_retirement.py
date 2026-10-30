"""Safely request retirement of an old LightRAG knowledge version."""

import json
import os
import re
from datetime import UTC, datetime
from pathlib import Path

import httpx
from dotenv import find_dotenv, load_dotenv

from deerflow.community.knowledgeops.knowledge_versions import (
    load_knowledge_history,
)
from deerflow.community.knowledgeops.proposal_store import (
    load_update_proposal,
)
from deerflow.community.knowledgeops.release_workflow import (
    _save_journal_atomic,
)


async def request_old_version_retirement(
    *,
    proposal_id: str,
    old_track_id: str,
    journal_dir: str | Path,
    knowledge_dir: str | Path,
    proposal_dir: str | Path,
) -> dict:
    """Request deletion only after validating the replacement version.

    The deletion is asynchronous. A successful request does not
    mean that the old version has already disappeared.
    """

    journal_path = Path(journal_dir) / f"{proposal_id}.json"

    # Validate proposal ID through the existing storage loader.
    proposal = load_update_proposal(proposal_id, proposal_dir)

    journal = json.loads(journal_path.read_text(encoding="utf-8"))

    if not isinstance(journal, dict):
        raise ValueError("Invalid release journal.")

    if journal.get("proposal_id") != proposal_id:
        raise ValueError("Release journal belongs to another proposal.")

    # This prevents sequential duplicate deletion requests.
    if journal.get("status") != "INDEXED":
        raise ValueError("Release must be INDEXED and not previously retired.")

    knowledge_id = journal.get("knowledge_id")
    old_version = journal.get("old_version")
    new_version = journal.get("new_version")
    receipt = journal.get("receipt")

    if type(old_version) is not int or type(new_version) is not int or new_version != old_version + 1 or not isinstance(receipt, dict):
        raise ValueError("Invalid release version information.")

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
        raise ValueError("Local release approval is inconsistent.")

    new_track_id = receipt.get("track_id")

    if receipt.get("status") != "SUBMITTED" or receipt.get("knowledge_id") != knowledge_id or receipt.get("version") != new_version or not isinstance(new_track_id, str) or not isinstance(old_track_id, str) or old_track_id == new_track_id:
        raise ValueError("Invalid release tracking information.")

    for track_id in (old_track_id, new_track_id):
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", track_id):
            raise ValueError("Invalid tracking ID.")

    load_dotenv(find_dotenv(usecwd=True))

    api_key = os.getenv("LIGHTRAG_API_KEY")
    base_url = os.getenv("LIGHTRAG_BASE_URL", "http://127.0.0.1:9621").rstrip("/")

    if not api_key:
        raise RuntimeError("LIGHTRAG_API_KEY is missing.")

    headers = {"X-API-Key": api_key}

    async with httpx.AsyncClient(timeout=30) as client:

        async def get_document(track_id: str, version: int) -> dict:
            response = await client.get(
                f"{base_url}/documents/track_status/{track_id}",
                headers=headers,
            )
            response.raise_for_status()

            payload = response.json()

            if payload.get("track_id") != track_id:
                raise RuntimeError("LightRAG tracking ID mismatch.")

            documents = payload.get("documents")

            if not isinstance(documents, list) or len(documents) != 1:
                raise RuntimeError("Expected exactly one indexed document.")

            document = documents[0]

            expected_source = f"knowledgeops-{knowledge_id}-v{version}.txt"

            if document.get("status") != "processed" or document.get("file_path") != expected_source or not isinstance(document.get("id"), str) or not document["id"].startswith("doc-"):
                raise RuntimeError("Document identity or indexing verification failed.")

            return document

        # Verify the replacement BEFORE touching the old version.
        new_document = await get_document(new_track_id, new_version)
        old_document = await get_document(old_track_id, old_version)

        if old_document["id"] == new_document["id"]:
            raise RuntimeError("Old and new document IDs are identical.")

        old_document_id = old_document["id"]

        # Persist destructive intent before calling LightRAG.
        journal["status"] = "DELETION_REQUESTING"
        journal["old_track_id"] = old_track_id
        journal["old_document_id"] = old_document_id
        journal["new_document_id"] = new_document["id"]
        journal["deletion_requested_at_utc"] = datetime.now(UTC).isoformat()

        _save_journal_atomic(journal_path, journal)

        try:
            response = await client.request(
                "DELETE",
                f"{base_url}/documents/delete_document",
                headers=headers,
                json={
                    "doc_ids": [old_document_id],
                    "delete_file": False,
                    "delete_llm_cache": False,
                },
            )
            response.raise_for_status()
            payload = response.json()

            if not isinstance(payload, dict):
                raise ValueError("Invalid deletion response.")

        except (httpx.HTTPError, ValueError) as exc:
            journal["status"] = "DELETION_UNKNOWN"
            _save_journal_atomic(journal_path, journal)

            raise RuntimeError("Deletion outcome is uncertain. Reconcile remote state before any retry.") from exc

    remote_status = payload.get("status")

    if remote_status == "deletion_started":
        next_status = "DELETION_STARTED"
    elif remote_status in {"busy", "not_allowed"}:
        # No automatic retry of a destructive operation.
        next_status = "DELETION_NEEDS_REVIEW"
    else:
        next_status = "DELETION_UNKNOWN"

    journal["status"] = next_status
    journal["deletion_response"] = payload

    _save_journal_atomic(journal_path, journal)

    return {
        "status": next_status,
        "proposal_id": proposal_id,
        "old_document_id": old_document_id,
        "new_document_id": new_document["id"],
    }
