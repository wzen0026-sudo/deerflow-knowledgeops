import fcntl
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from tempfile import mkstemp
from uuid import UUID


def review_update_proposal(
    proposal_id: str,
    decision: str,
    reviewer: str,
    review_reason: str,
    storage_dir: str | Path,
) -> dict:
    """Record a local human review without publishing knowledge."""

    try:
        normalized_id = str(UUID(proposal_id))
    except (TypeError, ValueError, AttributeError) as exc:
        raise ValueError("Invalid proposal ID.") from exc

    if normalized_id != proposal_id:
        raise ValueError("Proposal ID must be a canonical UUID.")

    if decision not in {"APPROVE", "REJECT"}:
        raise ValueError("Decision must be APPROVE or REJECT.")

    if not isinstance(reviewer, str) or not reviewer.strip():
        raise ValueError("Reviewer is required.")

    if not isinstance(review_reason, str) or not review_reason.strip():
        raise ValueError("Review reason is required.")

    folder = Path(storage_dir)
    path = folder / f"{normalized_id}.json"
    lock_path = folder / ".review.lock"

    # Prevent concurrent local reviews from overwriting each other.
    with lock_path.open("a+") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)

        proposal = json.loads(path.read_text(encoding="utf-8"))

        if not isinstance(proposal, dict) or proposal.get("proposal_id") != normalized_id:
            raise ValueError("Stored proposal does not match its ID.")

        if proposal.get("status") != "PENDING_REVIEW" or proposal.get("published") is not False:
            raise ValueError("Proposal is already reviewed or published.")

        approved = decision == "APPROVE"

        proposal.update(
            {
                "status": "APPROVED" if approved else "REJECTED",
                "human_approved": approved,
                "published": False,
                "reviewer": reviewer.strip(),
                "review_reason": review_reason.strip(),
                "reviewed_at_utc": datetime.now(UTC).isoformat(),
            }
        )

        # Write a temporary file and replace the original atomically.
        fd, temp_name = mkstemp(
            prefix=".proposal-",
            suffix=".tmp",
            dir=folder,
        )

        try:
            with os.fdopen(fd, "w", encoding="utf-8") as file:
                json.dump(
                    proposal,
                    file,
                    ensure_ascii=False,
                    indent=2,
                )
                file.flush()
                os.fsync(file.fileno())

            os.replace(temp_name, path)

        finally:
            if os.path.exists(temp_name):
                os.unlink(temp_name)

    return proposal
