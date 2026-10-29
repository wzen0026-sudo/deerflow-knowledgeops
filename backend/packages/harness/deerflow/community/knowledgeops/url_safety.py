import ipaddress
import re
from urllib.parse import urlsplit

BLOCKED_SUFFIXES = (
    ".localhost",
    ".local",
    ".internal",
    ".test",
    ".invalid",
    ".example",
    ".onion",
    ".nip.io",
    ".sslip.io",
    ".localtest.me",
    ".lvh.me",
)


def validate_public_url(url: str) -> dict:
    """
    Perform preliminary safety checks on an external webpage URL.

    This checks URL structure and obvious internal-network targets.
    It does not verify DNS resolution or redirect destinations.
    """

    if not isinstance(url, str) or not url:
        return {
            "safe": False,
            "reason": "URL must be a non-empty string.",
        }

    if len(url) > 2048:
        return {
            "safe": False,
            "reason": "URL is too long.",
        }

    if any(char.isspace() or ord(char) < 32 for char in url):
        return {
            "safe": False,
            "reason": "URL contains invalid whitespace or control characters.",
        }

    try:
        parsed = urlsplit(url)
        hostname = parsed.hostname
        port = parsed.port
    except ValueError:
        return {
            "safe": False,
            "reason": "Malformed URL.",
        }

    if parsed.scheme != "https":
        return {
            "safe": False,
            "reason": "Only HTTPS URLs are allowed.",
        }

    if not hostname:
        return {
            "safe": False,
            "reason": "URL has no valid hostname.",
        }

    if parsed.username is not None or parsed.password is not None:
        return {
            "safe": False,
            "reason": "URLs containing credentials are not allowed.",
        }

    if port not in (None, 443):
        return {
            "safe": False,
            "reason": "Non-standard HTTPS ports are not allowed.",
        }

    hostname = hostname.lower()

    # Reject direct IP addresses, including public IP literals.
    try:
        ipaddress.ip_address(hostname)
    except ValueError:
        pass
    else:
        return {
            "safe": False,
            "reason": "Direct IP addresses are not allowed.",
        }

    if hostname == "localhost" or "localhost" in hostname.split("."):
        return {
            "safe": False,
            "reason": "Localhost is not allowed.",
        }

    if hostname.endswith(BLOCKED_SUFFIXES):
        return {
            "safe": False,
            "reason": "Internal or special-use hostname is not allowed.",
        }

    # Allow ordinary DNS hostnames only.
    labels = hostname.split(".")

    if len(labels) < 2 or any(not re.fullmatch(r"[a-z0-9-]+", label) or label.startswith("-") or label.endswith("-") for label in labels):
        return {
            "safe": False,
            "reason": "Invalid public hostname format.",
        }

    if labels[-1].isdigit():
        return {
            "safe": False,
            "reason": "Invalid hostname suffix.",
        }

    return {
        "safe": True,
        "reason": "URL passed preliminary safety checks.",
    }


# Explicitly permitted source hosts for the first prototype.
# Permission to fetch does NOT imply that a source is trustworthy.
ALLOWED_SOURCE_HOSTS = frozenset(
    {
        "python.org",
        "www.python.org",
        "docs.python.org",
        "devguide.python.org",
        "realpython.com",
        "www.realpython.com",
        "endoflife.date",
    }
)


def validate_allowed_source_url(url: str) -> dict:
    """
    Check both preliminary URL safety and the source allowlist.

    Passing this check does not guarantee that the destination
    or its content is safe or factually reliable.
    """

    # Step 1: Reject obviously unsafe URL formats.
    safety_result = validate_public_url(url)

    if not safety_result["safe"]:
        return safety_result

    # Step 2: Extract the hostname.
    hostname = urlsplit(url).hostname

    if hostname is None:
        return {
            "safe": False,
            "reason": "URL has no valid hostname.",
        }

    hostname = hostname.lower()

    # Step 3: Require an exact hostname match.
    if hostname not in ALLOWED_SOURCE_HOSTS:
        return {
            "safe": False,
            "reason": "Source hostname is not on the allowlist.",
        }

    return {
        "safe": True,
        "reason": "URL passed preliminary checks and source allowlist.",
    }


def get_source_publisher(url: str) -> str | None:
    """Identify the publisher of an allowlisted source URL."""

    safety_result = validate_allowed_source_url(url)

    if not safety_result["safe"]:
        return None

    hostname = urlsplit(url).hostname

    if hostname is None:
        return None

    hostname = hostname.lower()

    # Python official sites share the same publisher.
    if hostname == "python.org" or hostname.endswith(".python.org"):
        return "python.org"

    # Real Python hosts share the same publisher.
    if hostname in {"realpython.com", "www.realpython.com"}:
        return "realpython.com"

    return hostname
