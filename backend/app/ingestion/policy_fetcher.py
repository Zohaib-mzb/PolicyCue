from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup


USER_AGENT = "PolicyLens/1.0"
MAX_RESPONSE_SIZE = 5 * 1024 * 1024


async def fetch_policy(url: str) -> str:
    parsed = urlparse(url)

    if parsed.scheme not in {"http", "https"}:
        raise ValueError("Invalid URL scheme.")

    async with httpx.AsyncClient(
        timeout=15.0,
        follow_redirects=False,
        headers={"User-Agent": USER_AGENT},
    ) as client:
        response = await client.get(url)

    if response.status_code != 200:
        raise ValueError(
            f"Unable to fetch policy page: HTTP {response.status_code}"
        )

    content_length = response.headers.get("content-length")

    if content_length and int(content_length) > MAX_RESPONSE_SIZE:
        raise ValueError("Policy page is too large.")

    soup = BeautifulSoup(response.text, "html.parser")

    for element in soup(
        ["script", "style", "noscript", "svg"]
    ):
        element.decompose()

    text = soup.get_text(" ", strip=True)

    if not text:
        raise ValueError("Policy page contains no readable text.")

    return text