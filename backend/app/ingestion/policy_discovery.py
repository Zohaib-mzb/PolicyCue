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
        "artificial_intelligence_policy",
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
        href = normalize_url(urljoin(base_url, link["href"]))
        parsed = urlparse(href)

        if parsed.scheme not in {"http", "https"}:
            continue

        if parsed.netloc.lower() != base_domain:
            continue

        path = parsed.path.lower().rstrip("/")
        text = link.get_text(" ", strip=True).lower()

        searchable = f"{text} {path}"

        for category, keywords in KEYWORDS.items():
            if any(keyword in searchable for keyword in keywords):
                if href not in results[category]:
                    results[category].append(href)

    return results