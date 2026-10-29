import re
from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, HttpUrl, field_validator


class EvidenceRecord(BaseModel):
    """A structured record of evidence for a knowledge claim."""

    claim: str
    source_url: HttpUrl
    excerpt: str

    source_type: Literal[
        "official",
        "academic",
        "reputable_media",
        "technical_community",
        "personal_blog",
        "unknown",
    ] = "unknown"

    evidence_date: date | None = None

    is_primary_source: bool = False

    directness: Literal[
        "direct",
        "indirect",
        "weak",
    ] = "weak"

    stance: Literal[
        "SUPPORT",
        "CONTRADICT",
        "NEUTRAL",
    ] = "NEUTRAL"

    @field_validator("claim", "excerpt")
    @classmethod
    def validate_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Text must not be empty.")
        return value.strip()

    @field_validator("evidence_date", mode="before")
    @classmethod
    def normalize_evidence_date(cls, value):
        """Normalize common missing-date values to None."""

        if isinstance(value, str):
            normalized = value.strip().lower()

            if normalized in {
                "",
                "unknown",
                "n/a",
                "none",
                "null",
                "not specified",
                "not available",
            }:
                return None

        return value


def verify_evidence_excerpt(
    page_text: str,
    excerpt: str,
) -> dict[str, bool | str]:
    """
    Verify whether an evidence excerpt appears in the page text.

    Whitespace differences are ignored.
    This checks textual existence, not factual correctness.
    """

    if not page_text.strip():
        return {
            "verified": False,
            "reason": "Page text is empty.",
        }

    if not excerpt.strip():
        return {
            "verified": False,
            "reason": "Evidence excerpt is empty.",
        }

    # Normalize spaces, tabs, and line breaks.
    def normalize(text: str) -> str:
        # Remove Markdown bold and italic markers.
        text = re.sub(
            r"(?<!\w)(\*\*|__|\*|_)(.+?)\1(?!\w)",
            r"\2",
            text,
        )
        return " ".join(text.split())

    normalized_page = normalize(page_text)
    normalized_excerpt = normalize(excerpt)

    # Reject very short excerpts to avoid accidental matches.
    if len(normalized_excerpt) < 20:
        return {
            "verified": False,
            "reason": "Evidence excerpt is too short.",
        }

    matched = normalized_excerpt in normalized_page

    return {
        "verified": matched,
        "reason": ("Evidence excerpt found in the page text." if matched else "Evidence excerpt not found in the page text."),
    }


def check_evidence_date_presence(
    page_text: str,
    evidence_date: str | date | None,
) -> dict[str, str]:
    """Check whether an evidence date appears in webpage text.

    A date mention does not prove it is the publication date.
    """

    if evidence_date is None or evidence_date == "":
        return {
            "status": "MISSING",
            "reason": "No evidence date was provided.",
        }

    try:
        parsed_date = evidence_date if isinstance(evidence_date, date) else date.fromisoformat(evidence_date)
    except (ValueError, TypeError):
        return {
            "status": "INVALID_DATE",
            "reason": "Evidence date must use YYYY-MM-DD.",
        }

    # Support common date formats in English webpages.
    candidates = [
        parsed_date.isoformat(),
        f"{parsed_date.strftime('%B')} {parsed_date.day}, {parsed_date.year}",
        f"{parsed_date.strftime('%b')} {parsed_date.day}, {parsed_date.year}",
        f"{parsed_date.day} {parsed_date.strftime('%B')} {parsed_date.year}",
        f"{parsed_date.day} {parsed_date.strftime('%b')} {parsed_date.year}",
    ]

    normalized_text = " ".join(page_text.split()).casefold()

    for candidate in candidates:
        pattern = (
            rf"(?<![a-z0-9])"
            rf"{re.escape(candidate.casefold())}"
            rf"(?![a-z0-9])"
        )

        if re.search(pattern, normalized_text):
            return {
                "status": "DATE_MENTIONED",
                "reason": "The date appears in the webpage text.",
            }

    return {
        "status": "DATE_NOT_FOUND",
        "reason": "The date was not found in the webpage text.",
    }


def check_publication_date_metadata(
    metadata: dict | None,
    evidence_date: str | date | None,
) -> dict:
    """Compare an evidence date against explicit publication metadata."""

    if evidence_date is None or evidence_date == "":
        return {
            "status": "MISSING_DATE",
            "reason": "No evidence date was supplied.",
        }

    try:
        expected_date = evidence_date if isinstance(evidence_date, date) else date.fromisoformat(evidence_date)
    except (ValueError, TypeError):
        return {
            "status": "INVALID_DATE",
            "reason": "Invalid evidence date.",
        }

    if not isinstance(metadata, dict):
        metadata = {}

    publication_fields = {
        "article:published_time",
        "datepublished",
        "date_published",
        "published_time",
        "pubdate",
    }

    dates_found = set()

    for key, value in metadata.items():
        if not isinstance(key, str):
            continue

        if key.casefold() not in publication_fields:
            continue

        if not isinstance(value, str):
            continue

        try:
            parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00")).date()
        except ValueError:
            continue

        dates_found.add(parsed)

    if not dates_found:
        return {
            "status": "NO_PUBLICATION_METADATA",
            "reason": "No usable publication date metadata found.",
        }

    if len(dates_found) > 1:
        return {
            "status": "METADATA_CONFLICT",
            "reason": "Publication metadata contains conflicting dates.",
        }

    actual_date = next(iter(dates_found))

    if expected_date == actual_date:
        return {
            "status": "METADATA_MATCH",
            "metadata_date": actual_date.isoformat(),
            "reason": "Evidence date matches publication metadata.",
        }

    return {
        "status": "METADATA_MISMATCH",
        "metadata_date": actual_date.isoformat(),
        "reason": "Evidence date differs from publication metadata.",
    }


def assess_date_provenance(
    evidence_date: str | date | None,
    date_presence_status: str,
    publication_metadata_check: dict | None,
) -> dict:
    """Assess date provenance without claiming independent verification."""

    base = {
        "independently_verified": False,
        "provisional_freshness_allowed": False,
    }

    if evidence_date is None:
        return {
            **base,
            "status": "NEEDS_REVIEW",
            "reason": "Evidence date is missing.",
        }

    try:
        parsed_date = evidence_date if isinstance(evidence_date, date) else date.fromisoformat(evidence_date)
    except (TypeError, ValueError):
        return {
            **base,
            "status": "NEEDS_REVIEW",
            "reason": "Evidence date is invalid.",
        }

    if parsed_date > date.today():
        return {
            **base,
            "status": "NEEDS_REVIEW",
            "reason": "Evidence date is in the future.",
        }

    metadata = publication_metadata_check or {}
    metadata_status = metadata.get("status")

    if metadata_status in {
        "METADATA_MISMATCH",
        "METADATA_CONFLICT",
    }:
        return {
            **base,
            "status": "DATE_CONFLICT",
            "reason": "Publication metadata is inconsistent.",
        }

    if metadata_status == "METADATA_MATCH":
        if metadata.get("metadata_date") != parsed_date.isoformat():
            return {
                **base,
                "status": "DATE_CONFLICT",
                "reason": "Reported metadata date is inconsistent.",
            }

        return {
            **base,
            "status": "SOURCE_ATTESTED",
            "provisional_freshness_allowed": True,
            "body_date_mentioned": (date_presence_status == "DATE_MENTIONED"),
            "reason": ("Publication date matches source-provided metadata. Independent verification is not established."),
        }

    return {
        **base,
        "status": "NEEDS_REVIEW",
        "reason": "No matching publication metadata is available.",
    }
