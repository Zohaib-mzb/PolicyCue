import pytest

from backend.app.ingestion.url_fetcher import (
    URLFetchError,
    fetch_url,
)


@pytest.mark.anyio
async def test_invalid_url_is_rejected():
    with pytest.raises(URLFetchError):
        await fetch_url("http://127.0.0.1")


@pytest.mark.anyio
async def test_file_scheme_is_rejected():
    with pytest.raises(URLFetchError):
        await fetch_url("file:///etc/passwd")

import httpx
from unittest.mock import patch

from backend.app.ingestion import url_fetcher


@pytest.mark.anyio
async def test_fetch_pins_public_address_and_preserves_tls_hostname():
    seen = []
    def handle(request):
        seen.append(request)
        if len(seen) == 1:
            return httpx.Response(302, headers={"location": "/privacy"})
        return httpx.Response(200, headers={"content-type": "text/html"}, text="policy")
    client = httpx.AsyncClient(transport=httpx.MockTransport(handle))
    with patch.object(url_fetcher, "public_addresses", return_value=["93.184.216.34"]), patch.object(url_fetcher.httpx, "AsyncClient", return_value=client):
        assert await fetch_url("https://example.com/") == ("https://example.com/privacy", "policy")
    assert len(seen) == 2
    assert all(r.url.host == "93.184.216.34" for r in seen)
    assert all(r.headers["host"] == "example.com" for r in seen)
    assert all(r.extensions["sni_hostname"] == "example.com" for r in seen)


@pytest.mark.anyio
@pytest.mark.parametrize("location", ["http://127.0.0.1/", "https://evil.com/", "https://example.com:8080/"])
async def test_redirect_rejected_before_destination_request(location):
    seen = []
    def handle(request):
        seen.append(request)
        return httpx.Response(302, headers={"location": location})
    client = httpx.AsyncClient(transport=httpx.MockTransport(handle))
    with patch.object(url_fetcher, "public_addresses", return_value=["93.184.216.34"]), patch.object(url_fetcher.httpx, "AsyncClient", return_value=client):
        with pytest.raises(URLFetchError):
            await fetch_url("https://example.com/")
    assert len(seen) == 1


@pytest.mark.anyio
@pytest.mark.parametrize("headers,body,allow_xml", [
    ({"content-type": "text/html"}, "x" * 21, False),
    ({"content-type": "text/html", "content-length": "invalid"}, "x", False),
    ({"content-type": "application/pdf"}, "pdf", False),
    ({"content-type": "application/xml"}, "<urlset/>", False),
])
async def test_invalid_or_oversized_response(headers, body, allow_xml):
    client = httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(200, headers=headers, content=body)))
    with patch.object(url_fetcher, "public_addresses", return_value=["93.184.216.34"]), patch.object(url_fetcher.httpx, "AsyncClient", return_value=client):
        with pytest.raises(URLFetchError):
            await fetch_url("https://example.com/", max_size=20, allow_xml=allow_xml)


@pytest.mark.anyio
async def test_xml_allowed_for_sitemap_fetch_only():
    client = httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(200, headers={"content-type": "application/xml"}, text="<urlset/>")))
    with patch.object(url_fetcher, "public_addresses", return_value=["93.184.216.34"]), patch.object(url_fetcher.httpx, "AsyncClient", return_value=client):
        assert await fetch_url("https://example.com/sitemap.xml", allow_xml=True) == ("https://example.com/sitemap.xml", "<urlset/>")


@pytest.mark.anyio
async def test_stream_size_limit_without_content_length():
    class OversizedStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b"x" * 16384
            yield b"x" * 16384
            raise AssertionError("Fetcher should stop before reading the whole response")
    client = httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(200, headers={"content-type": "text/html"}, stream=OversizedStream())))
    with patch.object(url_fetcher, "public_addresses", return_value=["93.184.216.34"]), patch.object(url_fetcher.httpx, "AsyncClient", return_value=client):
        with pytest.raises(URLFetchError, match="too large"):
            await fetch_url("https://example.com/", max_size=20000)


@pytest.mark.anyio
async def test_redirect_count_is_bounded():
    seen = []
    def handle(request):
        seen.append(request)
        return httpx.Response(302, headers={"location": "/loop"})
    client = httpx.AsyncClient(transport=httpx.MockTransport(handle))
    with patch.object(url_fetcher, "public_addresses", return_value=["93.184.216.34"]), patch.object(url_fetcher.httpx, "AsyncClient", return_value=client):
        with pytest.raises(URLFetchError, match="redirects"):
            await fetch_url("https://example.com/")
    assert len(seen) == url_fetcher.MAX_REDIRECTS + 1
