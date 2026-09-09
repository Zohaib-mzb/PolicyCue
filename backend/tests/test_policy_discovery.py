from backend.app.ingestion.policy_discovery import (
    discover_policy_links,
)
from backend.app.ingestion.policy_categories import PolicyCategory


def test_discovers_policy_links():
    html = """
    <html>
        <body>
            <a href="/privacy-policy">Privacy Policy</a>
            <a href="/terms-of-service">Terms of Service</a>
            <a href="/community-guidelines">
                Community Guidelines
            </a>
            <a href="https://other-site.com/privacy">
                Other Privacy
            </a>
        </body>
    </html>
    """

    result = discover_policy_links(
        html,
        "https://example.com",
    )

    assert (
        "https://example.com/privacy-policy"
        in result[PolicyCategory.PRIVACY]
    )

    assert (
        "https://example.com/terms-of-service"
        in result[PolicyCategory.TERMS]
    )

    assert (
        "https://example.com/community-guidelines"
        in result[PolicyCategory.COMMUNITY]
    )

    assert all(
        "other-site.com" not in url
        for urls in result.values()
        for url in urls
    )