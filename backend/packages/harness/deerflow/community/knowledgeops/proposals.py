from datetime import UTC, datetime
from uuid import uuid4


def create_update_proposal(
    original_knowledge: str,
    proposed_knowledge: str,
    reason: str,
    audit_report: dict,
) -> dict:
    """Create a knowledge update proposal requiring human review."""

    original_knowledge = original_knowledge.strip()
    proposed_knowledge = proposed_knowledge.strip()
    reason = reason.strip()

    if not original_knowledge or not proposed_knowledge or not reason:
        return {
            "status": "INVALID",
            "reason": "Original knowledge, proposed knowledge and reason are required.",
        }

    if original_knowledge == proposed_knowledge:
        return {
            "status": "INVALID",
            "reason": "Proposed knowledge must differ from original knowledge.",
        }

    if not isinstance(audit_report, dict):
        return {
            "status": "INVALID",
            "reason": "A valid audit report is required.",
        }

    if audit_report.get("claim") != original_knowledge:
        return {
            "status": "INVALID",
            "reason": "Audit report does not match the original knowledge.",
        }

    investigation_status = audit_report.get("investigation_status")

    if not investigation_status:
        return {
            "status": "INVALID",
            "reason": "Audit report has no investigation status.",
        }

    evidence = audit_report.get("evidence") or []
    review_items = audit_report.get("review_items") or []

    if not isinstance(evidence, list) or not isinstance(review_items, list):
        return {
            "status": "INVALID",
            "reason": "Audit evidence must be provided as lists.",
        }

    proposal = {
        "proposal_id": str(uuid4()),
        "status": "PENDING_REVIEW",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "original_knowledge": original_knowledge,
        "proposed_knowledge": proposed_knowledge,
        "reason": reason,
        "investigation_status": investigation_status,
        "assessment_status": audit_report.get("assessment_status"),
        "evidence": evidence,
        "review_items": review_items,
        "human_approved": False,
        "published": False,
        "reviewer": None,
        "reviewed_at_utc": None,
    }

    return proposal
