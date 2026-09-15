from enum import Enum


class PolicyCategory(str, Enum):
    PRIVACY = "privacy_policy"
    TERMS = "terms_of_service"
    COMMUNITY = "community_guidelines"
    COOKIE = "cookie_policy"
    DPA = "data_processing_agreement"
    AI = "ai_policy"
    CONTENT = "content_policy"
    SAFETY = "safety_policy"
    REFUND = "refund_policy"
    USER_AGREEMENT = "user_agreement"
    SHIPPING = "shipping_policy"
    PAYMENT = "pricing_payment_terms"
    ACCEPTABLE_USE = "acceptable_use_policy"
    ACCESSIBILITY = "accessibility_statement"
    DISCLAIMER = "disclaimer"
    COPYRIGHT = "dmca_copyright"


CATEGORY_PHRASES = {
    PolicyCategory.PRIVACY: ("privacy", "privacy policy", "privacy notice"),
    PolicyCategory.TERMS: ("terms", "terms of service", "terms of use", "terms and conditions", "service agreement"),
    PolicyCategory.COOKIE: ("cookies", "cookie policy", "cookie notice"),
    PolicyCategory.REFUND: ("refund policy", "return policy", "refunds", "returns", "refund and return policy"),
    PolicyCategory.SHIPPING: ("shipping", "delivery policy", "shipping policy", "delivery terms"),
    PolicyCategory.PAYMENT: ("pricing", "payment terms", "payment policy", "billing terms", "subscription terms"),
    PolicyCategory.ACCEPTABLE_USE: ("acceptable use", "aup"),
    PolicyCategory.ACCESSIBILITY: ("accessibility", "accessibility statement"),
    PolicyCategory.DISCLAIMER: ("disclaimer", "disclaimers"),
    PolicyCategory.COPYRIGHT: ("dmca", "copyright", "copyright policy", "copyright notice"),
    PolicyCategory.DPA: ("data processing agreement", "data processing", "dpa"),
    PolicyCategory.COMMUNITY: ("community guidelines", "community policies", "professional community policies"),
    PolicyCategory.AI: ("ai policy", "artificial intelligence policy", "generative ai policy"),
    PolicyCategory.CONTENT: ("content policy", "content guidelines"),
    PolicyCategory.SAFETY: ("safety policy", "safety guidelines"),
    PolicyCategory.USER_AGREEMENT: ("user agreement",),
}
