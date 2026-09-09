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