import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def build_audit_report(
    investigation_result: dict[str, Any],
) -> dict[str, Any]:
    """Build an audit report with final and provisional evidence."""

    assessment = investigation_result.get("assessment") or {}
    sources = assessment.get("sources_used") or []
    failures = investigation_result.get("failures") or []

    audited_sources = []

    for source in sources:
        if not isinstance(source, dict):
            continue

        audited_sources.append(
            {
                "publisher": source.get("publisher"),
                "source_url": source.get("source_url"),
                "stance": source.get("stance"),
                "excerpt": source.get("excerpt"),
                "evidence_date": source.get("evidence_date"),
                "date_verification_status": source.get(
                    "date_verification_status",
                    "NOT_VERIFIED",
                ),
                "date_presence_status": source.get(
                    "date_presence_status",
                    "NOT_CHECKED",
                ),
                "publication_metadata_check": source.get(
                    "publication_metadata_check",
                    {"status": "NOT_CHECKED"},
                ),
                "date_provenance": source.get(
                    "date_provenance",
                    {"status": "NOT_CHECKED"},
                ),
                "source_type": source.get("source_type"),
                "is_primary_source": source.get("is_primary_source"),
                "directness": source.get("directness"),
                "reliability_score": source.get("reliability_score"),
                "reliability_details": source.get(
                    "reliability_details",
                    {},
                ),
            }
        )

    review_items = []

    for failure in failures:
        if not isinstance(failure, dict):
            continue

        evidence = failure.get("evidence")

        if not isinstance(evidence, dict) or not evidence:
            continue

        is_provisional = failure.get("status") == "PROVISIONAL_SCORED"

        review_items.append(
            {
                "source_url": failure.get("source_url"),
                "status": failure.get("status"),
                "stage": failure.get("stage"),
                "reason": failure.get("reason"),
                "evidence": evidence,
                "review_required": True,
                "provisional_result": (failure.get("provisional_result") if is_provisional else None),
                "date_verification_status": evidence.get(
                    "date_verification_status",
                    "NOT_VERIFIED",
                ),
                "date_presence_status": evidence.get(
                    "date_presence_status",
                    "NOT_CHECKED",
                ),
                "publication_metadata_check": evidence.get(
                    "publication_metadata_check",
                    {"status": "NOT_CHECKED"},
                ),
                "date_provenance": evidence.get(
                    "date_provenance",
                    {"status": "NOT_CHECKED"},
                ),
            }
        )

    return {
        "schema_version": "1.2",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "claim": investigation_result.get("claim"),
        "investigation_status": investigation_result.get("status"),
        "assessment_status": assessment.get("status"),
        "assessment_report": assessment.get("report"),
        "assessment_reason": assessment.get("reason"),
        "sources_requested": investigation_result.get("sources_requested"),
        "sources_scored": investigation_result.get(
            "sources_scored",
            0,
        ),
        "sources_provisional": investigation_result.get(
            "sources_provisional",
            0,
        ),
        "duplicates_skipped": assessment.get(
            "duplicates_skipped",
            0,
        ),
        "evidence": audited_sources,
        "review_items": review_items,
        "failures": failures,
        "limitations": [
            "Confidence scores are heuristic, not probabilities.",
            "Provisional scores are not final confidence scores.",
            "Source-provided dates are not independently verified.",
            "Different publishers may share original evidence.",
            "Evidence stance and relevance require further review.",
        ],
    }


def save_audit_report(
    audit: dict[str, Any],
    output_path: str,
) -> str:
    """Save a KnowledgeOps audit report to a JSON file."""

    path = Path(output_path).expanduser()

    # Create the output directory if needed.
    path.parent.mkdir(parents=True, exist_ok=True)

    # Use exclusive creation to avoid overwriting an audit.
    with path.open("x", encoding="utf-8") as file:
        json.dump(
            audit,
            file,
            ensure_ascii=False,
            indent=2,
        )
        file.write("\n")

    return str(path)
