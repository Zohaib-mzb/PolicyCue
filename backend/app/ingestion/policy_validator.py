from backend.app.ingestion.policy_categories import PolicyCategory


CATEGORY_KEYWORDS = {
    PolicyCategory.PRIVACY: [
        "privacy policy",
        "personal data",
        "personal information",
        "data protection",
    ],
    PolicyCategory.TERMS: [
        "terms of service",
        "terms of use",
        "terms and conditions",
    ],
    PolicyCategory.COMMUNITY: [
        "community guidelines",
        "community policies",
    ],
    PolicyCategory.COOKIE: [
        "cookie policy",
        "cookies",
    ],
    PolicyCategory.DPA: [
        "data processing agreement",
        "data processing",
    ],
    PolicyCategory.AI: [
        "ai policy",
        "artificial intelligence",
        "generative ai",
    ],
    PolicyCategory.CONTENT: [
        "content policy",
        "content guidelines",
    ],
    PolicyCategory.SAFETY: [
        "safety policy",
        "safety guidelines",
    ],
    PolicyCategory.REFUND: [
        "refund policy",
        "return policy",
    ],
    PolicyCategory.USER_AGREEMENT: [
        "user agreement",
    ],
}


def is_valid_policy(
    text: str,
    category: PolicyCategory,
) -> bool:
    normalized = text.lower()

    keywords = CATEGORY_KEYWORDS.get(category, [])

    matches = sum(
        1 for keyword in keywords
        if keyword in normalized
    )

    return matches >= 1