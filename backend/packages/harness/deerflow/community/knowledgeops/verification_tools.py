import os
from urllib.parse import urlsplit

import httpx
from dotenv import find_dotenv, load_dotenv
from langchain_core.tools import StructuredTool

from deerflow.community.knowledgeops.evidence import verify_evidence_excerpt


async def knowledge_verify_evidence(
    source_url: str,
    excerpt: str,
) -> str:
    """Verify an excerpt against the actual source webpage."""

    parsed = urlsplit(source_url)

    if parsed.scheme != "https" or not parsed.hostname:
        return "Status: INVALID\nReason: A valid HTTPS URL is required."

    if not excerpt.strip():
        return "Status: INVALID\nReason: Excerpt cannot be empty."

    load_dotenv(find_dotenv(usecwd=True))
    api_key = os.getenv("JINA_API_KEY")

    if not api_key:
        return "Status: ERROR\nReason: JINA_API_KEY is not configured."

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.get(
                "https://r.jina.ai/" + source_url,
                headers={"Authorization": f"Bearer {api_key}"},
            )
    except httpx.HTTPError:
        return "Status: FETCH_FAILED\nReason: Could not retrieve the webpage."

    if response.status_code != 200:
        return f"Status: FETCH_FAILED\nHTTP status: {response.status_code}"

    result = verify_evidence_excerpt(response.text, excerpt)

    status = "VERIFIED" if result["verified"] else "NOT_FOUND"

    return f"Source: {source_url}\nStatus: {status}\nReason: {result['reason']}\nNote: This checks quotation existence, not factual correctness."


knowledge_verify_evidence_tool = StructuredTool.from_function(
    coroutine=knowledge_verify_evidence,
    name="knowledge_verify_evidence",
    description=("Fetch a source webpage using Jina Reader and verify whether a quoted evidence excerpt actually appears in the page. This does not determine whether the claim is true."),
)
