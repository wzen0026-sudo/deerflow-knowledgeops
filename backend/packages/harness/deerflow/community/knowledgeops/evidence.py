import re
from datetime import date
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
