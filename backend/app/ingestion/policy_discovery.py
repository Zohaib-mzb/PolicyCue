from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup

from backend.app.ingestion.policy_categories import PolicyCategory


KEYWORDS = {
    PolicyCategory.PRIVACY: [
        "privacy-policy",
        "privacy_policy",
        "privacy",
    ],
    PolicyCategory.TERMS: [
        "terms-of-service",
        "terms_of_service",
        "terms",
    ],
    PolicyCategory.COMMUNITY: [
        "community-guidelines",
        "community_guidelines",
        "professional-community-policies",
    ],
    PolicyCategory.COOKIE: [
        "cookie-policy",
        "cookie_policy",
        "cookies",
    ],
    PolicyCategory.DPA: [
        "data-processing-agreement",
        "data_processing_agreement",
        "data-processing",
        "dpa",
    ],
    PolicyCategory.AI: [
        "ai-policy",
        "ai_policy",
        "artificial-intelligence-policy",
    ],
    PolicyCategory.CONTENT: [
        "content-policy",
        "content_policy",
        "content-guidelines",
        "content_guidelines",
    ],
    PolicyCategory.SAFETY: [
        "safety-policy",
        "safety_policy",
        "safety-guidelines",
        "safety_guidelines",
    ],
    PolicyCategory.REFUND: [
        "refund-policy",
        "refund_policy",
        "return-policy",
        "return_policy",
    ],
    PolicyCategory.USER_AGREEMENT: [
        "user-agreement",
        "user_agreement",
    ],
}


COMMON_POLICY_PATHS = {
    PolicyCategory.PRIVACY: [
        "/privacy",
        "/privacy-policy",
        "/legal/privacy",
        "/legal/privacy-policy",
    ],
    PolicyCategory.TERMS: [
        "/terms",
        "/terms-of-service",
        "/legal/terms",
        "/legal/terms-of-service",
    ],
    PolicyCategory.COMMUNITY: [
        "/community-guidelines",
        "/legal/community-guidelines",
    ],
    PolicyCategory.COOKIE: [
        "/cookies",
        "/cookie-policy",
        "/legal/cookies",
        "/legal/cookie-policy",
    ],
    PolicyCategory.DPA: [
        "/dpa",
        "/data-processing-agreement",
        "/legal/dpa",
    ],
    PolicyCategory.AI: [
        "/ai-policy",
        "/legal/ai-policy",
    ],
    PolicyCategory.CONTENT: [
        "/content-policy",
        "/legal/content-policy",
    ],
    PolicyCategory.SAFETY: [
        "/safety-policy",
        "/legal/safety-policy",
    ],
    PolicyCategory.REFUND: [
        "/refund-policy",
        "/legal/refund-policy",
        "/refunds",
    ],
    PolicyCategory.USER_AGREEMENT: [
        "/user-agreement",
        "/legal/user-agreement",
    ],
}


def normalize_url(url: str) -> str:
    parsed = urlparse(url)

    return parsed._replace(
        query="",
        fragment="",
    ).geturl().rstrip("/")


def discover_policy_links(
    html: str,
    base_url: str,
) -> dict[PolicyCategory, list[str]]:
    soup = BeautifulSoup(html, "html.parser")

    results = {
        category: []
        for category in PolicyCategory
    }

    base_domain = urlparse(base_url).netloc.lower()

    for link in soup.find_all("a", href=True):
        href = normalize_url(
            urljoin(base_url, link["href"])
        )

        parsed = urlparse(href)

        if parsed.scheme not in {"http", "https"}:
            continue

        if parsed.netloc.lower() != base_domain:
            continue

        searchable = (
            f"{link.get_text(' ', strip=True)} "
            f"{parsed.path}"
        ).lower()

        for category, keywords in KEYWORDS.items():
            if any(
                keyword in searchable
                for keyword in keywords
            ):
                if href not in results[category]:
                    results[category].append(href)

    return results


async def discover_common_policy_paths(
    base_url: str,
) -> dict[PolicyCategory, list[str]]:
    from backend.app.ingestion.policy_fetcher import fetch_policy
    from backend.app.ingestion.policy_validator import is_valid_policy

    results = {
        category: []
        for category in PolicyCategory
    }

    parsed = urlparse(base_url)
    origin = f"{parsed.scheme}://{parsed.netloc}"

    for category, paths in COMMON_POLICY_PATHS.items():
        for path in paths:
            url = normalize_url(f"{origin}{path}")

            try:
                text = await fetch_policy(url)

                if is_valid_policy(text, category):
                    results[category].append(url)

            except Exception:
                continue

    return results