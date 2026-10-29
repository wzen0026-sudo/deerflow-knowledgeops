from datetime import date, datetime

from langchain_core.tools import StructuredTool

FRESHNESS_POLICIES = {
    "software": 30,
    "api": 30,
    "price": 14,
    "policy": 14,
    "company_process": 90,
    "general": 180,
    "stable": 365,
}


def _parse_date(value: str) -> date | None:
    """Convert YYYY-MM-DD text into a date object."""
    try:
        return datetime.strptime(value.strip(), "%Y-%m-%d").date()
    except (ValueError, AttributeError):
        return None


async def knowledge_check_freshness(
    knowledge: str,
    last_verified: str,
    knowledge_type: str = "general",
    max_age_days: int | None = None,
) -> str:
    """Check whether a knowledge item may be stale."""

    knowledge = knowledge.strip()
    knowledge_type = knowledge_type.strip().lower()

    if not knowledge:
        return "Status: UNCERTAIN\nReason: knowledge must not be empty."

    verified_date = _parse_date(last_verified)

    if verified_date is None:
        return "Status: UNCERTAIN\nReason: last_verified must use YYYY-MM-DD format."

    if max_age_days is None:
        max_age_days = FRESHNESS_POLICIES.get(
            knowledge_type,
            FRESHNESS_POLICIES["general"],
        )

    if max_age_days <= 0:
        return "Status: UNCERTAIN\nReason: max_age_days must be greater than 0."

    today = date.today()
    age_days = (today - verified_date).days

    if age_days < 0:
        return "Status: UNCERTAIN\nReason: last_verified is in the future."

    if age_days > max_age_days:
        status = "STALE"
        reason = f"The knowledge was last verified {age_days} days ago, which exceeds the allowed age of {max_age_days} days."
    else:
        status = "CURRENT"
        reason = f"The knowledge was last verified {age_days} days ago, within the allowed age of {max_age_days} days."

    return f"Knowledge: {knowledge}\nKnowledge type: {knowledge_type}\nStatus: {status}\nAge: {age_days} days\nMaximum age: {max_age_days} days\nReason: {reason}"


knowledge_check_freshness_tool = StructuredTool.from_function(
    coroutine=knowledge_check_freshness,
    name="knowledge_check_freshness",
    description=("Check whether an existing knowledge item may be stale. Automatically applies a freshness policy based on knowledge type: software, api, price, policy, company_process, general, or stable."),
)
SOURCE_AUTHORITY_SCORES = {
    "official": 25,
    "academic": 23,
    "reputable_media": 20,
    "technical_community": 15,
    "personal_blog": 10,
    "unknown": 5,
}


def _calculate_freshness_score(age_days: int, max_age_days: int) -> int:
    """Calculate freshness score from 0 to 25."""

    ratio = age_days / max_age_days

    if ratio <= 0.25:
        return 25
    if ratio <= 0.50:
        return 22
    if ratio <= 1.00:
        return 18
    if ratio <= 1.50:
        return 10
    if ratio <= 2.00:
        return 5

    return 0


def _trust_verdict(score: int) -> str:
    """Convert numeric trust score into a clear human verdict."""

    if score < 40:
        return "不可信"
    if score < 60:
        return "有点假"
    if score < 80:
        return "应该是真的"

    return "大概率是真的"


async def knowledge_trust_score(
    knowledge: str,
    source_type: str,
    evidence_date: str,
    knowledge_type: str = "general",
    is_primary_source: bool = False,
    independent_confirmations: int = 0,
    high_quality_conflicts: int = 0,
) -> str:
    """Evaluate the trustworthiness of a knowledge item."""

    knowledge = knowledge.strip()
    source_type = source_type.strip().lower()
    knowledge_type = knowledge_type.strip().lower()

    if not knowledge:
        return "Status: INSUFFICIENT_EVIDENCE\nReason: knowledge must not be empty."

    evidence_date_parsed = _parse_date(evidence_date)

    if evidence_date_parsed is None:
        return "Status: INSUFFICIENT_EVIDENCE\nReason: evidence_date must use YYYY-MM-DD format."

    if independent_confirmations < 0 or high_quality_conflicts < 0:
        return "Status: INSUFFICIENT_EVIDENCE\nReason: confirmation and conflict counts cannot be negative."

    if source_type == "unknown" and independent_confirmations == 0:
        return "Status: INSUFFICIENT_EVIDENCE\nReason: source authority is unknown and there is no independent confirmation."

    max_age_days = FRESHNESS_POLICIES.get(
        knowledge_type,
        FRESHNESS_POLICIES["general"],
    )

    age_days = (date.today() - evidence_date_parsed).days

    if age_days < 0:
        return "Status: INSUFFICIENT_EVIDENCE\nReason: evidence_date is in the future."

    authority_score = SOURCE_AUTHORITY_SCORES.get(source_type, 5)

    freshness_score = _calculate_freshness_score(
        age_days,
        max_age_days,
    )

    primary_source_score = 15 if is_primary_source else 5

    if independent_confirmations == 0:
        cross_validation_score = 0
    elif independent_confirmations == 1:
        cross_validation_score = 8
    elif independent_confirmations == 2:
        cross_validation_score = 14
    else:
        cross_validation_score = 20

    if high_quality_conflicts == 0:
        conflict_score = 15
    elif high_quality_conflicts == 1:
        conflict_score = 7
    else:
        conflict_score = 0

    total_score = authority_score + freshness_score + primary_source_score + cross_validation_score + conflict_score

    verdict = _trust_verdict(total_score)

    return (
        f"Knowledge: {knowledge}\n"
        f"Trust Score: {total_score}/100\n"
        f"Verdict: {verdict}\n\n"
        f"Source authority: {authority_score}/25\n"
        f"Freshness: {freshness_score}/25\n"
        f"Primary source: {primary_source_score}/15\n"
        f"Cross validation: {cross_validation_score}/20\n"
        f"Conflict safety: {conflict_score}/15\n\n"
        f"Evidence age: {age_days} days\n"
        f"Knowledge type: {knowledge_type}\n"
        f"Source type: {source_type}"
    )


knowledge_trust_score_tool = StructuredTool.from_function(
    coroutine=knowledge_trust_score,
    name="knowledge_trust_score",
    description=(
        "Score the trustworthiness of a knowledge item from 0 to 100 "
        "using source authority, freshness, primary-source status, "
        "independent confirmation, and high-quality conflicting evidence. "
        "Returns a clear verdict: 不可信, 有点假, 应该是真的, or 大概率是真的."
    ),
)
EVIDENCE_AUTHORITY_SCORES = {
    "official": 40,
    "academic": 36,
    "reputable_media": 30,
    "technical_community": 22,
    "personal_blog": 15,
    "unknown": 5,
}

EVIDENCE_DIRECTNESS_SCORES = {
    "direct": 15,
    "indirect": 8,
    "weak": 3,
}


def _calculate_evidence_freshness_score(
    age_days: int,
    max_age_days: int,
) -> int:
    """Calculate evidence freshness score from 0 to 25."""

    ratio = age_days / max_age_days

    if ratio <= 0.25:
        return 25
    if ratio <= 0.50:
        return 22
    if ratio <= 1.00:
        return 18
    if ratio <= 1.50:
        return 10
    if ratio <= 2.00:
        return 5

    return 0


async def knowledge_evidence_reliability(
    evidence: str,
    source_url: str,
    source_type: str,
    evidence_date: str,
    knowledge_type: str,
    is_primary_source: bool,
    directness: str,
    stance: str,
) -> str:
    """
    Evaluate the reliability of one piece of evidence.

    Stance is kept separate from reliability:
    SUPPORT, CONTRADICT, or NEUTRAL.
    """

    evidence = evidence.strip()
    source_type = source_type.strip().lower()
    knowledge_type = knowledge_type.strip().lower()
    directness = directness.strip().lower()
    stance = stance.strip().upper()

    if not evidence:
        return "Status: INVALID\nReason: evidence must not be empty."

    if stance not in {"SUPPORT", "CONTRADICT", "NEUTRAL"}:
        return "Status: INVALID\nReason: stance must be SUPPORT, CONTRADICT, or NEUTRAL."

    if directness not in EVIDENCE_DIRECTNESS_SCORES:
        return "Status: INVALID\nReason: directness must be direct, indirect, or weak."

    parsed_date = _parse_date(evidence_date)

    if parsed_date is None:
        return "Status: INVALID\nReason: evidence_date must use YYYY-MM-DD format."

    age_days = (date.today() - parsed_date).days

    if age_days < 0:
        return "Status: INVALID\nReason: evidence_date is in the future."

    max_age_days = FRESHNESS_POLICIES.get(
        knowledge_type,
        FRESHNESS_POLICIES["general"],
    )

    authority_score = EVIDENCE_AUTHORITY_SCORES.get(
        source_type,
        EVIDENCE_AUTHORITY_SCORES["unknown"],
    )

    freshness_score = _calculate_evidence_freshness_score(
        age_days,
        max_age_days,
    )

    primary_score = 20 if is_primary_source else 8

    directness_score = EVIDENCE_DIRECTNESS_SCORES[directness]

    reliability_score = authority_score + freshness_score + primary_score + directness_score

    return (
        f"Evidence: {evidence}\n"
        f"Source: {source_url}\n"
        f"Evidence Reliability: {reliability_score}/100\n"
        f"Stance: {stance}\n\n"
        f"Source authority: {authority_score}/40\n"
        f"Freshness: {freshness_score}/25\n"
        f"Primary source: {primary_score}/20\n"
        f"Directness: {directness_score}/15\n\n"
        f"Evidence age: {age_days} days\n"
        f"Knowledge type: {knowledge_type}\n"
        f"Source type: {source_type}"
    )


knowledge_evidence_reliability_tool = StructuredTool.from_function(
    coroutine=knowledge_evidence_reliability,
    name="knowledge_evidence_reliability",
    description=("Evaluate how reliable one piece of evidence is while keeping its stance toward the claim separate. Stance must be SUPPORT, CONTRADICT, or NEUTRAL."),
)


def _claim_confidence_verdict(score: int) -> str:
    """Convert claim confidence into a clear human verdict."""

    if score < 40:
        return "不可信"
    if score < 60:
        return "有点假"
    if score < 80:
        return "应该是真的"

    return "大概率是真的"


async def knowledge_claim_confidence(
    claim: str,
    support_scores: list[int],
    contradict_scores: list[int],
) -> str:
    """
    Calculate final confidence for a claim from evidence reliability scores.

    support_scores:
        Reliability scores of evidence supporting the claim.

    contradict_scores:
        Reliability scores of evidence contradicting the claim.
    """

    claim = claim.strip()

    if not claim:
        return "Status: INSUFFICIENT_EVIDENCE\nReason: claim must not be empty."

    all_scores = support_scores + contradict_scores

    if not all_scores:
        return "Status: INSUFFICIENT_EVIDENCE\nReason: no supporting or contradicting evidence was provided."

    if any(score < 0 or score > 100 for score in all_scores):
        return "Status: INVALID\nReason: every evidence reliability score must be between 0 and 100."

    # If all available evidence is very weak,
    # do not force a confident conclusion.
    if max(all_scores) < 40:
        return "Status: INSUFFICIENT_EVIDENCE\nReason: all available evidence has low reliability."

    # Only the strongest few pieces of evidence should dominate.
    # This reduces the effect of flooding the system with many weak sources.
    strongest_support = sorted(
        support_scores,
        reverse=True,
    )[:3]

    strongest_contradictions = sorted(
        contradict_scores,
        reverse=True,
    )[:3]

    # Square reliability scores so that highly reliable evidence
    # carries substantially more weight than weak evidence.
    support_weight = sum(score**3 for score in strongest_support)

    contradict_weight = sum(score**3 for score in strongest_contradictions)

    total_weight = support_weight + contradict_weight

    if total_weight == 0:
        return "Status: INSUFFICIENT_EVIDENCE\nReason: usable evidence weight is zero."

    confidence_score = round(100 * support_weight / total_weight)

    verdict = _claim_confidence_verdict(confidence_score)

    return (
        f"Claim: {claim}\n"
        f"Claim Confidence: {confidence_score}/100\n"
        f"Verdict: {verdict}\n\n"
        f"Supporting evidence used: {strongest_support}\n"
        f"Contradicting evidence used: {strongest_contradictions}\n"
        f"Support weight: {support_weight}\n"
        f"Contradiction weight: {contradict_weight}\n"
    )


knowledge_claim_confidence_tool = StructuredTool.from_function(
    coroutine=knowledge_claim_confidence,
    name="knowledge_claim_confidence",
    description=("Calculate the final confidence of a claim using reliability scores from evidence that supports or contradicts the claim. Returns a 0-100 confidence score and a clear verdict."),
)


def calculate_evidence_reliability_details(
    *,
    source_type: str,
    evidence_date: str,
    knowledge_type: str,
    is_primary_source: bool,
    directness: str,
) -> dict[str, int]:
    """Calculate structured evidence reliability scores."""

    parsed_date = _parse_date(evidence_date)

    if parsed_date is None:
        raise ValueError("Invalid evidence date.")

    age_days = (date.today() - parsed_date).days

    if age_days < 0:
        raise ValueError("Evidence date cannot be in the future.")

    source_type = source_type.strip().lower()
    knowledge_type = knowledge_type.strip().lower()
    directness = directness.strip().lower()

    if directness not in EVIDENCE_DIRECTNESS_SCORES:
        raise ValueError("Invalid evidence directness.")

    max_age_days = FRESHNESS_POLICIES.get(
        knowledge_type,
        FRESHNESS_POLICIES["general"],
    )

    authority_score = EVIDENCE_AUTHORITY_SCORES.get(
        source_type,
        EVIDENCE_AUTHORITY_SCORES["unknown"],
    )

    freshness_score = _calculate_evidence_freshness_score(
        age_days,
        max_age_days,
    )

    primary_score = 20 if is_primary_source else 8
    directness_score = EVIDENCE_DIRECTNESS_SCORES[directness]

    total_score = authority_score + freshness_score + primary_score + directness_score

    return {
        "reliability_score": total_score,
        "authority_score": authority_score,
        "freshness_score": freshness_score,
        "primary_score": primary_score,
        "directness_score": directness_score,
        "age_days": age_days,
    }
