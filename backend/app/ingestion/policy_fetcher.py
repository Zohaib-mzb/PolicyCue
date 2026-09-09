import ipaddress
import socket
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup


USER_AGENT = "PolicyLens/1.0"
MAX_RESPONSE_SIZE = 5 * 1024 * 1024
TIMEOUT = 15.0


def _is_private_host(hostname: str) -> bool:
    try:
        ip = ipaddress.ip_address(hostname)
        return (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_reserved
            or ip.is_multicast
        )
    except ValueError:
        pass

    try:
        addresses = socket.getaddrinfo(
            hostname,
            None,
            type=socket.SOCK_STREAM,
        )

        for address in addresses:
            ip = ipaddress.ip_address(address[4][0])

            if (
                ip.is_private
                or ip.is_loopback
                or ip.is_link_local
                or ip.is_reserved
                or ip.is_multicast
            ):
                return True

    except socket.gaierror:
        return True

    return False


async def fetch_policy(url: str) -> str:
    parsed = urlparse(url)

    if parsed.scheme not in {"http", "https"}:
        raise ValueError("Invalid URL scheme.")

    if not parsed.hostname:
        raise ValueError("URL must contain a hostname.")

    if _is_private_host(parsed.hostname):
        raise ValueError("Private or local URLs are not allowed.")

    async with httpx.AsyncClient(
        timeout=TIMEOUT,
        follow_redirects=False,
        headers={"User-Agent": USER_AGENT},
    ) as client:
        response = await client.get(url)

    if response.status_code != 200:
        raise ValueError(
            f"Unable to fetch policy page: HTTP {response.status_code}"
        )

    content_length = response.headers.get("content-length")

    if content_length:
        try:
            if int(content_length) > MAX_RESPONSE_SIZE:
                raise ValueError("Policy page is too large.")
        except ValueError as exc:
            if str(exc) == "Policy page is too large.":
                raise
            raise ValueError("Invalid content length.") from exc

    if len(response.content) > MAX_RESPONSE_SIZE:
        raise ValueError("Policy page is too large.")

    soup = BeautifulSoup(response.text, "html.parser")

    for element in soup(["script", "style", "noscript", "svg"]):
        element.decompose()

    text = soup.get_text(" ", strip=True)

    if not text:
        raise ValueError("Policy page contains no readable text.")

    return text