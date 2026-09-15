import re

from backend.app.ingestion.policy_categories import CATEGORY_PHRASES, PolicyCategory


MIN_POLICY_WORDS = 80
CATEGORY_TITLES = {
    **CATEGORY_PHRASES,
    # "Privacy" alone also labels help indexes and marketing/privacy hubs.
    PolicyCategory.PRIVACY: ("privacy policy", "privacy notice", "privacy statement", "data protection policy"),
}
CATEGORY_EVIDENCE = {
    PolicyCategory.PRIVACY: ("personal data", "personal information", "collect", "share", "retention", "processing"),
    PolicyCategory.TERMS: ("agree", "agreement", "obligations", "liability", "termination", "governing law"),
    PolicyCategory.COOKIE: ("cookies", "browser", "tracking", "consent", "preferences", "device"),
    PolicyCategory.REFUND: ("refund", "return", "purchase", "days", "eligible", "receipt"),
    PolicyCategory.SHIPPING: ("shipping", "delivery", "order", "business days", "carrier", "tracking"),
    PolicyCategory.PAYMENT: ("payment", "fees", "billing", "subscription", "charge", "invoice"),
    PolicyCategory.ACCEPTABLE_USE: ("prohibited", "must not", "may not", "abuse", "unauthorized", "suspend"),
    PolicyCategory.ACCESSIBILITY: ("wcag", "disabilities", "assistive", "accessible", "accessibility", "contact"),
    PolicyCategory.DISCLAIMER: ("warranty", "warranties", "liability", "accuracy", "as is", "advice"),
    PolicyCategory.COPYRIGHT: ("infringement", "copyright", "notice", "rights holder", "counter notification", "designated agent"),
    PolicyCategory.DPA: ("processor", "controller", "personal data", "subprocessor", "processing"),
    PolicyCategory.COMMUNITY: ("harassment", "respect", "prohibited", "content", "violence", "misinformation"),
    PolicyCategory.AI: ("artificial intelligence", "generated", "models", "transparency", "training", "responsible"),
    PolicyCategory.CONTENT: ("content", "prohibited", "remove", "moderation", "publish", "rights"),
    PolicyCategory.SAFETY: ("harm", "report", "abuse", "safety", "protect", "threats"),
    PolicyCategory.USER_AGREEMENT: ("agree", "agreement", "account", "services", "terminate", "liability"),
}


def normalized_words(text: str) -> str:
    return re.sub(r"[^\w]+", " ", text.casefold()).strip()


def has_phrase(text: str, phrase: str) -> bool:
    return f" {phrase} " in f" {text} "


def classify_policy(text: str, headings: str = "") -> list[PolicyCategory]:
    normalized = normalized_words(text)
    if len(normalized.split()) < MIN_POLICY_WORDS:
        return []
    # A title/section heading plus multiple substantive indicators is required.
    # Navigation and footer text are removed before this function is called.
    labels = normalized_words(headings or text[:300])
    return [
        category for category, phrases in CATEGORY_TITLES.items()
        if any(has_phrase(labels, phrase) for phrase in phrases)
        and sum(has_phrase(normalized, word) for word in CATEGORY_EVIDENCE[category]) >= 2
    ]


def is_valid_policy(text: str, category: PolicyCategory) -> bool:
    return category in classify_policy(text)
