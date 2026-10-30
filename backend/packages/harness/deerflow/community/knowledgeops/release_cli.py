"""Command-line interface for the KnowledgeOps release pipeline."""

import argparse
import asyncio
import json
from pathlib import Path
from uuid import UUID

from deerflow.community.knowledgeops.release_orchestrator import (
    advance_release_pipeline,
)
from deerflow.community.knowledgeops.release_recovery import (
    recover_completed_local_release,
)
from deerflow.community.knowledgeops.release_workflow import (
    start_release_indexing,
)


def canonical_uuid(value: str) -> str:
    try:
        if str(UUID(value)) != value:
            raise ValueError
    except (TypeError, ValueError, AttributeError) as exc:
        raise argparse.ArgumentTypeError("Expected a canonical UUID") from exc

    return value


def main() -> int:
    parser = argparse.ArgumentParser(description="KnowledgeOps release management CLI")

    parser.add_argument(
        "action",
        choices=["status", "start", "advance", "inspect", "recover"],
    )
    parser.add_argument("--proposal-id", type=canonical_uuid, required=True)
    parser.add_argument("--knowledge-id", type=canonical_uuid)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--old-track-id")
    parser.add_argument("--allow-retirement", action="store_true")

    args = parser.parse_args()

    base = args.data_dir.expanduser().resolve()
    journal_dir = base / "journals"
    knowledge_dir = base / "knowledge"
    proposal_dir = base / "proposals"

    async def execute():
        if args.action == "status":
            path = journal_dir / f"{args.proposal_id}.json"
            journal = json.loads(path.read_text(encoding="utf-8"))

            if not isinstance(journal, dict) or journal.get("proposal_id") != args.proposal_id:
                raise ValueError("Invalid release journal")

            return {
                "proposal_id": args.proposal_id,
                "knowledge_id": journal.get("knowledge_id"),
                "status": journal.get("status"),
                "old_version": journal.get("old_version"),
                "new_version": journal.get("new_version"),
            }

        if args.action == "start":
            if not args.knowledge_id:
                raise ValueError("--knowledge-id is required for start")

            return await start_release_indexing(
                knowledge_id=args.knowledge_id,
                proposal_id=args.proposal_id,
                knowledge_dir=knowledge_dir,
                proposal_dir=proposal_dir,
                journal_dir=journal_dir,
            )

        if args.action == "inspect":
            journal_path = journal_dir / f"{args.proposal_id}.json"
            journal = json.loads(journal_path.read_text(encoding="utf-8"))

            if journal.get("status") not in {
                "DELETION_REQUESTING",
                "DELETION_UNKNOWN",
                "DELETION_NEEDS_REVIEW",
            }:
                raise ValueError("No uncertain deletion to inspect")

            return await advance_release_pipeline(
                proposal_id=args.proposal_id,
                journal_dir=journal_dir,
                knowledge_dir=knowledge_dir,
                proposal_dir=proposal_dir,
                inspect_deletion=True,
            )

        if args.action == "recover":
            return await recover_completed_local_release(
                proposal_id=args.proposal_id,
                journal_dir=journal_dir,
                knowledge_dir=knowledge_dir,
                proposal_dir=proposal_dir,
            )

        return await advance_release_pipeline(
            proposal_id=args.proposal_id,
            journal_dir=journal_dir,
            knowledge_dir=knowledge_dir,
            proposal_dir=proposal_dir,
            old_track_id=args.old_track_id,
            allow_retirement=args.allow_retirement,
        )

    try:
        result = asyncio.run(execute())
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        parser.exit(1, f"KnowledgeOps error: {exc}\n")

    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
