from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup

from backend.app.ingestion.policy_categories import PolicyCategory


KEYWORDS = {
    PolicyCategory.PRIVACY: [
        "privacy",
    ],
    PolicyCategory.TERMS: [
        "terms",
        "terms-of-service",
        "terms_of_service",
    ],
    PolicyCategory.COMMUNITY: [
        "community-guidelines",
        "community_guidelines",
        "community",
    ],
    PolicyCategory.COOKIE: [
        "cookie",
    ],
    PolicyCategory.DPA: [
        "data-processing",
        "data_processing",
        "dpa",
    ],
    PolicyCategory.AI: [
        "ai-policy",
        "ai_policy",
        "artificial-intelligence",
    ],
    PolicyCategory.CONTENT: [
        "content-policy",
        "content_policy",
        "content",
    ],
    PolicyCategory.SAFETY: [
        "safety-policy",
        "safety_policy",
        "safety",
    ],
    PolicyCategory.REFUND: [
        "refund",
        "return-policy",
    ],
    PolicyCategory.USER_AGREEMENT: [
        "user-agreement",
        "user_agreement",
    ],
}


def discover_policy_links(
    html: str,
    base_url: str,
) -> dict[PolicyCategory, list[str]]:
    soup = BeautifulSoup(html, "html.parser")

    results = {
        category: []
        for category in PolicyCategory
    }

    base_domain = urlparse(base_url).netloc

    for link in soup.find_all("a", href=True):
        href = urljoin(base_url, link["href"])
        parsed = urlparse(href)

        if parsed.scheme not in {"http", "https"}:
            continue

        if parsed.netloc != base_domain:
            continue

        searchable = (
            f"{link.get_text(' ', strip=True)} "
            f"{parsed.path}"
        ).lower()

        for category, keywords in KEYWORDS.items():
            if any(keyword in searchable for keyword in keywords):
                if href not in results[category]:
                    results[category].append(href)

    return results