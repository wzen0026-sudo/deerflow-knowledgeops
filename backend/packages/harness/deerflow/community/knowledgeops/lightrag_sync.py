"""Submit versioned KnowledgeOps content to LightRAG."""

import os
from uuid import UUID

import httpx
from dotenv import find_dotenv, load_dotenv


async def submit_version_for_indexing(
    knowledge_id: str,
    version: int,
    content: str,
) -> dict:
    """Submit a knowledge version for indexing, not final publication."""

    # Validate inputs.
    try:
        normalized_id = str(UUID(knowledge_id))
    except (TypeError, ValueError, AttributeError) as exc:
        raise ValueError("Invalid knowledge ID.") from exc

    if normalized_id != knowledge_id:
        raise ValueError("Knowledge ID must be a canonical UUID.")

    if type(version) is not int or version < 1:
        raise ValueError("Version must be a positive integer.")

    if not isinstance(content, str) or not content.strip():
        raise ValueError("Knowledge content cannot be empty.")

    # Read credentials from the DeerFlow environment.
    load_dotenv(find_dotenv(usecwd=True))

    api_key = os.getenv("LIGHTRAG_API_KEY")
    base_url = os.getenv(
        "LIGHTRAG_BASE_URL",
        "http://127.0.0.1:9621",
    ).rstrip("/")

    if not api_key:
        raise RuntimeError("LIGHTRAG_API_KEY is not configured.")

    # Each version has a unique, traceable source name.
    file_source = f"knowledgeops-{knowledge_id}-v{version}.txt"

    async with httpx.AsyncClient(timeout=30) as client:
        try:
            response = await client.post(
                f"{base_url}/documents/text",
                headers={"X-API-Key": api_key},
                json={
                    "text": content.strip(),
                    "file_source": file_source,
                },
            )
        except httpx.RequestError as exc:
            raise RuntimeError("Unable to contact LightRAG. Check the server.") from exc

    if response.status_code == 409:
        raise RuntimeError("This knowledge version already exists in LightRAG. Reconcile its state before retrying.")

    if response.status_code != 200:
        raise RuntimeError(f"LightRAG rejected the request (HTTP {response.status_code}).")

    try:
        payload = response.json()
    except ValueError as exc:
        raise RuntimeError("LightRAG returned invalid JSON.") from exc

    if not isinstance(payload, dict):
        raise RuntimeError("Unexpected LightRAG response.")

    track_id = payload.get("track_id")

    if payload.get("status") != "success" or not isinstance(track_id, str) or not track_id:
        raise RuntimeError("LightRAG did not confirm the indexing request.")

    return {
        "status": "SUBMITTED",
        "knowledge_id": knowledge_id,
        "version": version,
        "file_source": file_source,
        "track_id": track_id,
    }


async def check_version_indexing_status(track_id: str) -> dict:
    """Check whether LightRAG has finished indexing a document."""

    import re

    if not isinstance(track_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", track_id):
        raise ValueError("Invalid LightRAG track ID.")

    load_dotenv(find_dotenv(usecwd=True))

    api_key = os.getenv("LIGHTRAG_API_KEY")
    base_url = os.getenv(
        "LIGHTRAG_BASE_URL",
        "http://127.0.0.1:9621",
    ).rstrip("/")

    if not api_key:
        raise RuntimeError("LIGHTRAG_API_KEY is not configured.")

    try:
        async with httpx.AsyncClient(timeout=20) as client:
            response = await client.get(
                f"{base_url}/documents/track_status/{track_id}",
                headers={"X-API-Key": api_key},
            )
    except httpx.RequestError as exc:
        raise RuntimeError("Unable to contact LightRAG.") from exc

    if response.status_code != 200:
        raise RuntimeError(f"LightRAG status check failed (HTTP {response.status_code}).")

    try:
        payload = response.json()
    except ValueError as exc:
        raise RuntimeError("LightRAG returned invalid JSON.") from exc

    if not isinstance(payload, dict):
        raise RuntimeError("Invalid LightRAG status response.")

    if payload.get("track_id") != track_id:
        raise RuntimeError("LightRAG returned a different track ID.")

    documents = payload.get("documents")

    if not isinstance(documents, list):
        raise RuntimeError("Invalid LightRAG documents response.")

    if not documents:
        return {
            "status": "NOT_FOUND",
            "track_id": track_id,
            "document_count": 0,
        }

    statuses = []

    for document in documents:
        if not isinstance(document, dict):
            raise RuntimeError("Invalid document record.")

        status = document.get("status")

        if not isinstance(status, str):
            raise RuntimeError("Invalid document status.")

        statuses.append(status.lower())

    if all(status == "processed" for status in statuses):
        overall_status = "INDEXED"
    elif "failed" in statuses:
        overall_status = "FAILED"
    elif any(
        status
        in {
            "pending",
            "processing",
            "preprocessing",
            "parsing",
            "analyzing",
        }
        for status in statuses
    ):
        overall_status = "PROCESSING"
    else:
        overall_status = "UNKNOWN"

    return {
        "status": overall_status,
        "track_id": track_id,
        "document_count": len(documents),
        "document_statuses": statuses,
    }
