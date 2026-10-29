import json
import os
from pathlib import Path
from uuid import UUID


def _validate_proposal_id(proposal_id: str) -> str:
    """Validate a proposal ID before using it as a filename."""

    if not isinstance(proposal_id, str):
        raise ValueError("Proposal ID must be a string.")

    try:
        normalized = str(UUID(proposal_id))
    except ValueError as exc:
        raise ValueError("Invalid proposal ID.") from exc

    if normalized != proposal_id:
        raise ValueError("Proposal ID must be a canonical UUID.")

    return normalized


def save_update_proposal(
    proposal: dict,
    storage_dir: str | Path,
) -> str:
    """Save a pending proposal without overwriting existing data."""

    if not isinstance(proposal, dict):
        raise ValueError("Proposal must be a dictionary.")

    if proposal.get("status") != "PENDING_REVIEW" or proposal.get("human_approved") is not False or proposal.get("published") is not False:
        raise ValueError("Only unpublished pending proposals can be created.")

    proposal_id = _validate_proposal_id(proposal.get("proposal_id"))

    folder = Path(storage_dir)
    folder.mkdir(parents=True, exist_ok=True, mode=0o700)

    file_path = folder / f"{proposal_id}.json"

    content = json.dumps(
        proposal,
        ensure_ascii=False,
        indent=2,
    )

    # Exclusive creation prevents overwriting an existing proposal.
    fd = os.open(
        file_path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o600,
    )

    with os.fdopen(fd, "w", encoding="utf-8") as file:
        file.write(content)

    return str(file_path)


def load_update_proposal(
    proposal_id: str,
    storage_dir: str | Path,
) -> dict:
    """Load a previously saved proposal."""

    proposal_id = _validate_proposal_id(proposal_id)

    file_path = Path(storage_dir) / f"{proposal_id}.json"

    with file_path.open("r", encoding="utf-8") as file:
        proposal = json.load(file)

    if not isinstance(proposal, dict):
        raise ValueError("Stored proposal has invalid format.")

    if proposal.get("proposal_id") != proposal_id:
        raise ValueError("Stored proposal ID does not match.")

    return proposal
