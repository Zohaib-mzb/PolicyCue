from backend.app.ingestion.html_parser import extract_text
from backend.app.ingestion.policy_discovery import discover_policy_links
from backend.app.ingestion.url_fetcher import fetch_url


async def process_url(url: str) -> dict:
    final_url, html = await fetch_url(url)

    text = extract_text(html)

    policies = discover_policy_links(
        html,
        final_url,
    )

    return {
        "url": final_url,
        "text": text,
        "policies": policies,
    }


def process_text(text: str) -> dict:
    return {
        "text": text.strip(),
    }