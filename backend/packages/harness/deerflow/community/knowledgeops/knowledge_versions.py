import json
import os
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID, uuid4


def create_knowledge_history(
    initial_knowledge: str,
    storage_dir: str | Path,
) -> dict:
    """Create the first version of a knowledge record."""

    if not isinstance(initial_knowledge, str):
        raise ValueError("Knowledge must be text.")

    content = initial_knowledge.strip()

    if not content:
        raise ValueError("Initial knowledge cannot be empty.")

    knowledge_id = str(uuid4())
    created_at = datetime.now(UTC).isoformat()

    history = {
        "schema_version": "1.0",
        "knowledge_id": knowledge_id,
        "current_version": 1,
        "versions": [
            {
                "version": 1,
                "content": content,
                "action": "INITIAL",
                "created_at_utc": created_at,
                "proposal_id": None,
                "restored_from_version": None,
            }
        ],
    }

    folder = Path(storage_dir)
    folder.mkdir(parents=True, exist_ok=True, mode=0o700)

    path = folder / f"{knowledge_id}.json"

    # Create the history without overwriting an existing record.
    fd = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o600,
    )

    with os.fdopen(fd, "w", encoding="utf-8") as file:
        json.dump(history, file, ensure_ascii=False, indent=2)

    return history


def load_knowledge_history(
    knowledge_id: str,
    storage_dir: str | Path,
) -> dict:
    """Load a knowledge record and its version history."""

    try:
        normalized_id = str(UUID(knowledge_id))
    except (TypeError, ValueError, AttributeError) as exc:
        raise ValueError("Invalid knowledge ID.") from exc

    if normalized_id != knowledge_id:
        raise ValueError("Knowledge ID must be a canonical UUID.")

    path = Path(storage_dir) / f"{normalized_id}.json"

    with path.open("r", encoding="utf-8") as file:
        history = json.load(file)

    if not isinstance(history, dict):
        raise ValueError("Invalid knowledge history.")

    if history.get("knowledge_id") != normalized_id:
        raise ValueError("Knowledge ID mismatch.")

    if type(history.get("current_version")) is not int or not isinstance(history.get("versions"), list) or not history["versions"]:
        raise ValueError("Invalid knowledge version structure.")

    return history
