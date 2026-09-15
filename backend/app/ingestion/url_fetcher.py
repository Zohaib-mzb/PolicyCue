import asyncio
from urllib.parse import urljoin, urlparse

import httpx

from backend.app.ingestion.url_security import (
    public_addresses, same_site, valid_url_shape,
)


MAX_RESPONSE_SIZE = 5 * 1024 * 1024
REQUEST_TIMEOUT = 10.0
FETCH_DEADLINE = 20.0
MAX_REDIRECTS = 3
HTML_CONTENT_TYPES = {"text/html", "text/plain", "application/xhtml+xml"}
XML_CONTENT_TYPES = {"application/xml", "text/xml", "application/x-sitemap+xml"}


class URLFetchError(ValueError):
    pass


async def fetch_url(
    url: str,
    *,
    site_url: str | None = None,
    allow_xml: bool = False,
    max_size: int = MAX_RESPONSE_SIZE,
) -> tuple[str, str]:
    site_url = site_url or url
    try:
        async with asyncio.timeout(FETCH_DEADLINE):
            async with httpx.AsyncClient(
                timeout=REQUEST_TIMEOUT, follow_redirects=False, trust_env=False,
            ) as client:
                current_url = url
                for redirect_count in range(MAX_REDIRECTS + 1):
                    if not valid_url_shape(current_url) or not same_site(current_url, site_url):
                        raise URLFetchError("URL or redirect destination is not allowed.")
                    parsed = urlparse(current_url)
                    addresses = await asyncio.to_thread(public_addresses, parsed.hostname)
                    if not addresses:
                        raise URLFetchError("Private, local, or unresolved host is not allowed.")
                    # Pin the connection to the validated DNS result. Preserve Host
                    # and TLS certificate verification, without a second DNS lookup.
                    target = httpx.URL(current_url).copy_with(host=addresses[0])
                    async with client.stream(
                        "GET", target,
                        headers={"User-Agent": "PolicyLens/1.0", "Host": parsed.netloc},
                        extensions={"sni_hostname": parsed.hostname},
                    ) as response:
                        if response.is_redirect:
                            location = response.headers.get("location")
                            if not location or redirect_count == MAX_REDIRECTS:
                                raise URLFetchError("Invalid or excessive redirects.")
                            current_url = urljoin(current_url, location)
                            continue
                        if response.status_code != 200:
                            raise URLFetchError(f"Website returned HTTP {response.status_code}.")
                        length = response.headers.get("content-length")
                        if length:
                            try:
                                size = int(length)
                            except ValueError:
                                raise URLFetchError("Invalid content length.") from None
                            if size < 0 or size > max_size:
                                raise URLFetchError("Response is too large or invalid.")
                        content_type = response.headers.get("content-type", "").split(";")[0].strip().lower()
                        allowed = HTML_CONTENT_TYPES | (XML_CONTENT_TYPES if allow_xml else set())
                        if content_type not in allowed:
                            raise URLFetchError("Unsupported content type.")
                        body = bytearray()
                        async for part in response.aiter_bytes(chunk_size=16384):
                            body.extend(part)
                            if len(body) > max_size:
                                raise URLFetchError("Response is too large.")
                        return current_url, body.decode(response.encoding or "utf-8", errors="replace")
    except URLFetchError:
        raise
    except (httpx.HTTPError, TimeoutError, ValueError, UnicodeError, LookupError) as exc:
        raise URLFetchError("Website request failed or timed out.") from exc
    raise URLFetchError("Unable to fetch website.")
