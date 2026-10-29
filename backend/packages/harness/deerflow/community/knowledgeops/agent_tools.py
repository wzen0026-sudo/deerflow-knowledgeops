import json

from langchain_core.tools import StructuredTool

from deerflow.community.knowledgeops.analyzer import investigate_claim
from deerflow.community.knowledgeops.audit import build_audit_report
from deerflow.community.knowledgeops.tools import FRESHNESS_POLICIES


async def knowledge_investigate_claim(
    claim: str,
    knowledge_type: str = "general",
) -> str:
    """Investigate a knowledge claim using multiple evidence sources."""

    claim = claim.strip()
    knowledge_type = knowledge_type.strip().lower()

    if not claim:
        return json.dumps(
            {
                "status": "INVALID",
                "reason": "Claim cannot be empty.",
            }
        )

    if len(claim) > 1000:
        return json.dumps(
            {
                "status": "INVALID",
                "reason": "Claim is too long.",
            }
        )

    if knowledge_type not in FRESHNESS_POLICIES:
        return json.dumps(
            {
                "status": "INVALID",
                "reason": "Unsupported knowledge type.",
            }
        )

    # Reuse the existing KnowledgeOps investigation pipeline.
    investigation = await investigate_claim(
        claim=claim,
        knowledge_type=knowledge_type,
        max_sources=3,
    )

    # Build an auditable result including provisional evidence.
    audit = build_audit_report(investigation)

    return json.dumps(
        audit,
        ensure_ascii=False,
        indent=2,
    )


knowledge_investigate_claim_tool = StructuredTool.from_function(
    coroutine=knowledge_investigate_claim,
    name="knowledge_investigate_claim",
    description=(
        "Investigate a factual knowledge claim by searching for "
        "independent sources, extracting evidence, evaluating "
        "source reliability, detecting conflicts, and returning "
        "an auditable report. Provisional scores are not final "
        "confidence scores. Use for claims requiring verification."
    ),
)
