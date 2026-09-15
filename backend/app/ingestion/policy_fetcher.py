from backend.app.ingestion.html_parser import extract_text
from backend.app.ingestion.url_fetcher import fetch_url
from backend.app.ingestion.url_security import is_public_hostname


def _is_private_host(hostname: str) -> bool:
    return not is_public_hostname(hostname)


async def fetch_policy(url: str) -> str:
    _, html = await fetch_url(url)
    text = extract_text(html)
    if not text:
        raise ValueError("Policy page contains no readable text.")
    return text
