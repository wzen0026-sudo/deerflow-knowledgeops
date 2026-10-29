"""Coordinate and verify KnowledgeOps releases."""

from pathlib import Path

from deerflow.community.knowledgeops.knowledge_versions import (
    load_knowledge_history,
)
from deerflow.community.knowledgeops.lightrag_sync import (
    check_version_indexing_status,
)
from deerflow.community.knowledgeops.proposal_store import (
    load_update_proposal,
)


async def verify_published_release(
    *,
    knowledge_id: str,
    proposal_id: str,
    knowledge_dir: str | Path,
    proposal_dir: str | Path,
    old_track_id: str,
    new_track_id: str,
) -> dict:
    """Verify published local state against LightRAG indexing state.

    This function is read-only. It never publishes or deletes documents.
    """

    if old_track_id == new_track_id:
        raise ValueError("Old and new track IDs must differ.")

    history = load_knowledge_history(
        knowledge_id,
        knowledge_dir,
    )

    proposal = load_update_proposal(
        proposal_id,
        proposal_dir,
    )

    version = history["current_version"]
    versions = history["versions"]

    if version < 2 or len(versions) < 2:
        raise RuntimeError("No published update version found.")

    latest = versions[-1]
    previous = versions[-2]

    if latest["version"] != version:
        raise RuntimeError("Knowledge version history is inconsistent.")

    if latest.get("proposal_id") != proposal_id:
        raise RuntimeError("Published version does not match proposal.")

    if proposal.get("status") != "APPROVED" or proposal.get("human_approved") is not True or proposal.get("published") is not True or proposal.get("knowledge_id") != knowledge_id or proposal.get("published_version") != version:
        raise RuntimeError("Local publication record is inconsistent.")

    if proposal.get("original_knowledge") != previous["content"]:
        raise RuntimeError("Original knowledge does not match history.")

    if proposal.get("proposed_knowledge") != latest["content"]:
        raise RuntimeError("Published content does not match proposal.")

    # Verify the old version was removed from LightRAG.
    old_status = await check_version_indexing_status(old_track_id)

    # Verify the new version is still indexed.
    new_status = await check_version_indexing_status(new_track_id)

    if old_status["status"] != "NOT_FOUND":
        raise RuntimeError(f"Old version is still present: {old_status['status']}")

    if new_status["status"] != "INDEXED" or new_status.get("document_count") != 1:
        raise RuntimeError(f"New version is not fully indexed: {new_status['status']}")

    return {
        "status": "VERIFIED",
        "knowledge_id": knowledge_id,
        "proposal_id": proposal_id,
        "current_version": version,
        "old_index_status": old_status["status"],
        "new_index_status": new_status["status"],
        "local_publication": "CONFIRMED",
    }
