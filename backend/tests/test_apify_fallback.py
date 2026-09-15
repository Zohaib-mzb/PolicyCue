from unittest.mock import AsyncMock, patch

import pytest

from backend.app.ingestion.apify_fetcher import APIFY_MAX_CANDIDATES, fetch_policy_candidates_with_apify
from backend.app.ingestion.policy_discovery import discover_website_policies


class Settings:
    apify_api_token = "token-value"


@pytest.mark.anyio
async def test_no_token_returns_native_only(monkeypatch):
    class NoToken:
        apify_api_token = None

    monkeypatch.setattr("backend.app.ingestion.apify_fetcher.get_settings", lambda: NoToken())
    result = await fetch_policy_candidates_with_apify(["https://example.com/privacy"], "https://example.com")
    assert result == {"pages": [], "used": False, "status": "not_configured", "sent": 0}


@pytest.mark.anyio
async def test_native_success_does_not_call_apify():
    html = """
    <html><title>Privacy Policy</title><main><h1>Privacy Policy</h1>
    We collect personal information and share personal data with service providers.
    We describe retention, processing, privacy choices, and customer rights in this privacy policy.
    This notice explains personal information collection and disclosure for users of the service.
    We explain retention periods, account controls, processing purposes, security safeguards,
    data sharing, analytics, communications, legal compliance, and how users can exercise
    privacy rights related to personal information collected through the service.
    This privacy policy describes how personal information is collected, used, retained,
    disclosed, transferred, processed, protected, corrected, deleted, and shared when
    people use the website, mobile application, support tools, payment features, and
    communications services provided by the company to customers and visitors.
    </main></html>
    """

    async def fake_fetch(url, **kwargs):
        return url, html

    with patch("backend.app.ingestion.policy_discovery.validate_url", return_value=True), patch(
        "backend.app.ingestion.policy_discovery.fetch_url", side_effect=fake_fetch
    ), patch("backend.app.ingestion.policy_discovery.fetch_policy_candidates_with_apify", new=AsyncMock()) as apify:
        result = await discover_website_policies("https://example.com/privacy")

    assert result["pages"]
    assert result["fallback_used"] == "none"
    assert result["coverage_status"] == "partial"
    apify.assert_not_called()


@pytest.mark.anyio
async def test_native_403_high_confidence_candidate_calls_apify_and_enters_pipeline():
    page_text = "Privacy Policy " + "We collect personal information and share personal data with service providers. " * 8

    async def failed_fetch(url, **kwargs):
        return None

    apify_result = {
        "used": True,
        "status": "succeeded",
        "sent": 1,
        "returned": 1,
        "pages": [{"source_url": "https://example.com/privacy", "final_url": "https://example.com/privacy", "title": "Privacy Policy", "text": page_text}],
    }

    with patch("backend.app.ingestion.policy_discovery.validate_url", return_value=True), patch(
        "backend.app.ingestion.policy_discovery.fetch_url", side_effect=failed_fetch
    ), patch("backend.app.ingestion.policy_discovery.fetch_policy_candidates_with_apify", new=AsyncMock(return_value=apify_result)) as apify:
        result = await discover_website_policies("https://example.com/privacy")

    apify.assert_called_once()
    assert result["fallback_used"] == "apify"
    assert result["coverage_status"] == "partial"
    assert len(result["pages"]) == 1
    assert result["pages"][0]["acquisition_method"] == "apify"


@pytest.mark.anyio
async def test_low_confidence_failed_candidate_not_sent_to_apify():
    async def failed_fetch(url, **kwargs):
        return None

    with patch("backend.app.ingestion.policy_discovery.validate_url", return_value=True), patch(
        "backend.app.ingestion.policy_discovery.fetch_url", side_effect=failed_fetch
    ), patch("backend.app.ingestion.policy_discovery.fetch_policy_candidates_with_apify", new=AsyncMock(return_value={"used": False})) as apify:
        result = await discover_website_policies("https://example.com/random-page")

    sent = apify.call_args.args[0]
    assert "https://example.com/random-page" not in sent
    assert result["coverage_status"] == "access_limited"


@pytest.mark.anyio
async def test_apify_limits_and_unsafe_returned_canonical_rejected(monkeypatch):
    monkeypatch.setattr("backend.app.ingestion.apify_fetcher.get_settings", lambda: Settings())
    sent_payloads = []

    class Response:
        def __init__(self, status_code, data):
            self.status_code = status_code
            self._data = data
        def json(self):
            return self._data
        def raise_for_status(self):
            if self.status_code >= 400:
                raise AssertionError("unexpected status")

    class Client:
        async def __aenter__(self):
            return self
        async def __aexit__(self, *args):
            return None
        async def post(self, url, params=None, json=None):
            sent_payloads.append(json)
            return Response(201, {"data": {"id": "run"}})
        async def get(self, url, params=None):
            if "actor-runs" in url:
                return Response(200, {"data": {"status": "SUCCEEDED", "defaultDatasetId": "dataset"}})
            return Response(200, [{
                "url": "https://example.com/privacy",
                "metadata": {"canonicalUrl": "http://127.0.0.1/private", "title": "Privacy Policy"},
                "text": "Privacy Policy " + "We collect personal information and share personal data. " * 8,
            }])

    monkeypatch.setattr("backend.app.ingestion.apify_fetcher.validate_url", lambda url: True)
    monkeypatch.setattr("backend.app.ingestion.apify_fetcher.httpx.AsyncClient", lambda **kwargs: Client())
    monkeypatch.setattr("backend.app.ingestion.apify_fetcher.asyncio.sleep", AsyncMock())
    urls = [f"https://example.com/privacy/{i}" for i in range(APIFY_MAX_CANDIDATES + 5)]
    result = await fetch_policy_candidates_with_apify(urls, "https://example.com")

    assert result["sent"] == APIFY_MAX_CANDIDATES
    assert len(sent_payloads) == APIFY_MAX_CANDIDATES
    assert all(len(payload["startUrls"]) == 1 for payload in sent_payloads)
    assert result["pages"][0]["final_url"] == "https://example.com/privacy"


@pytest.mark.anyio
async def test_apify_sorts_candidates_before_applying_limit(monkeypatch):
    monkeypatch.setattr("backend.app.ingestion.apify_fetcher.get_settings", lambda: Settings())
    sent_payloads = []

    class Response:
        status_code = 201
        def __init__(self, data):
            self._data = data
        def json(self):
            return self._data
        def raise_for_status(self):
            return None

    class Client:
        async def __aenter__(self):
            return self
        async def __aexit__(self, *args):
            return None
        async def post(self, url, params=None, json=None):
            sent_payloads.append(json)
            return Response({"data": {"id": "run"}})
        async def get(self, url, params=None):
            if "actor-runs" in url:
                return Response({"data": {"status": "SUCCEEDED", "defaultDatasetId": "dataset"}})
            return Response([])

    monkeypatch.setattr("backend.app.ingestion.apify_fetcher.validate_url", lambda url: True)
    monkeypatch.setattr("backend.app.ingestion.apify_fetcher.httpx.AsyncClient", lambda **kwargs: Client())
    monkeypatch.setattr("backend.app.ingestion.apify_fetcher.asyncio.sleep", AsyncMock())
    urls = [f"https://example.com/refund-policy/{i}" for i in range(APIFY_MAX_CANDIDATES)] + [
        "https://example.com/legal",
        "https://example.com/legal-portal/legal-terms/terms-of-service",
    ]

    await fetch_policy_candidates_with_apify(urls, "https://example.com")

    sent = [payload["startUrls"][0]["url"] for payload in sent_payloads]
    assert sent[0] == "https://example.com/legal-portal/legal-terms/terms-of-service"
    assert "https://example.com/legal" in sent
    assert len(sent) == APIFY_MAX_CANDIDATES


@pytest.mark.anyio
async def test_fallback_failure_does_not_crash_and_reports_access_limited():
    async def failed_fetch(url, **kwargs):
        return None

    with patch("backend.app.ingestion.policy_discovery.validate_url", return_value=True), patch(
        "backend.app.ingestion.policy_discovery.fetch_url", side_effect=failed_fetch
    ), patch("backend.app.ingestion.policy_discovery.fetch_policy_candidates_with_apify", new=AsyncMock(return_value={"used": True, "status": "failed", "sent": 2, "pages": []})):
        result = await discover_website_policies("https://example.com/privacy")

    assert result["coverage_status"] == "access_limited"
    assert result["fallback_used"] == "apify"
    assert result["pages"] == []


@pytest.mark.anyio
async def test_duplicate_fallback_content_deduplicated():
    text = "Terms of Service " + "Users agree to these terms and this agreement may terminate services. " * 8
    async def failed_fetch(url, **kwargs):
        return None
    apify_result = {"used": True, "status": "succeeded", "sent": 2, "returned": 2, "pages": [
        {"source_url": "https://example.com/terms", "final_url": "https://example.com/terms", "title": "Terms of Service", "text": text},
        {"source_url": "https://example.com/terms-of-service", "final_url": "https://example.com/terms-of-service", "title": "Terms of Service", "text": text},
    ]}
    with patch("backend.app.ingestion.policy_discovery.validate_url", return_value=True), patch(
        "backend.app.ingestion.policy_discovery.fetch_url", side_effect=failed_fetch
    ), patch("backend.app.ingestion.policy_discovery.fetch_policy_candidates_with_apify", new=AsyncMock(return_value=apify_result)):
        result = await discover_website_policies("https://example.com/terms")
    assert len(result["pages"]) == 1
    assert result["skipped"]["duplicate_content"] >= 1
