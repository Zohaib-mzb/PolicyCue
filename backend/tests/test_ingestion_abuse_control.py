from io import BytesIO
from unittest.mock import AsyncMock, patch
from uuid import uuid4

from fastapi.testclient import TestClient

from backend.app.core.rate_limit import reset_rate_limits
from backend.app.core.session import SESSION_COOKIE_NAME, create_session_token
from backend.app.main import app


def setup_function():
    reset_rate_limits()


def _empty_policy_document():
    return {
        "url": "https://example.com/",
        "pages": [],
        "warnings": [],
        "skipped": {},
        "candidates": 0,
    }


def test_repeated_url_ingestion_without_preserving_cookies_hits_client_limit():
    with patch("backend.app.main.settings.ingestion_client_rate_limit", 1), patch(
        "backend.app.main.settings.ingestion_owner_rate_limit", 10
    ), patch(
        "backend.app.main.discover_website_policies",
        new=AsyncMock(return_value=_empty_policy_document()),
    ) as discover, patch("backend.app.main.store_chunks") as store:
        first = TestClient(app).post("/api/v1/ingest/url", json={"url": "https://example.com"})
        second = TestClient(app).post("/api/v1/ingest/url", json={"url": "https://example.com"})

    assert first.status_code == 200
    assert second.status_code == 429
    assert SESSION_COOKIE_NAME not in second.cookies
    discover.assert_awaited_once()
    store.assert_not_called()


def test_repeated_pdf_ingestion_without_preserving_cookies_hits_client_limit():
    with patch("backend.app.main.settings.ingestion_client_rate_limit", 1), patch(
        "backend.app.main.settings.ingestion_owner_rate_limit", 10
    ), patch("backend.app.main.extract_pdf_text", return_value="Policy text") as extract, patch(
        "backend.app.main.chunk_text", return_value=["chunk"]
    ), patch("backend.app.main.store_chunks") as store:
        first = TestClient(app).post(
            "/api/v1/ingest/pdf",
            files={"file": ("policy.pdf", b"%PDF", "application/pdf")},
        )
        second = TestClient(app).post(
            "/api/v1/ingest/pdf",
            files={"file": ("policy.pdf", b"%PDF", "application/pdf")},
        )

    assert first.status_code == 200
    assert second.status_code == 429
    assert SESSION_COOKIE_NAME not in second.cookies
    assert extract.call_count == 1
    assert store.call_count == 1


def test_pre_session_url_rejection_does_not_discover_or_store():
    with patch("backend.app.main.settings.ingestion_client_rate_limit", 1), patch(
        "backend.app.main.settings.ingestion_owner_rate_limit", 10
    ), patch(
        "backend.app.main.discover_website_policies",
        new=AsyncMock(return_value=_empty_policy_document()),
    ) as discover, patch("backend.app.main.store_chunks") as store:
        first = TestClient(app).post("/api/v1/ingest/url", json={"url": "https://example.com"})
        second = TestClient(app).post("/api/v1/ingest/url", json={"url": "https://example.com"})

    assert first.status_code == 200
    assert second.status_code == 429
    assert SESSION_COOKIE_NAME not in second.cookies
    discover.assert_awaited_once()
    store.assert_not_called()


def test_pre_session_pdf_rejection_does_not_parse_or_store():
    with patch("backend.app.main.settings.ingestion_client_rate_limit", 1), patch(
        "backend.app.main.settings.ingestion_owner_rate_limit", 10
    ), patch("backend.app.main.extract_pdf_text", return_value="Policy text") as extract, patch(
        "backend.app.main.chunk_text", return_value=["chunk"]
    ), patch("backend.app.main.store_chunks") as store:
        first = TestClient(app).post(
            "/api/v1/ingest/pdf",
            files={"file": ("policy.pdf", b"%PDF", "application/pdf")},
        )
        second = TestClient(app).post(
            "/api/v1/ingest/pdf",
            files={"file": ("policy.pdf", b"%PDF", "application/pdf")},
        )

    assert first.status_code == 200
    assert second.status_code == 429
    assert SESSION_COOKIE_NAME not in second.cookies
    assert extract.call_count == 1
    assert store.call_count == 1


def test_existing_owner_is_subject_to_owner_ingestion_limit():
    owner_id = str(uuid4())
    client = TestClient(app)
    client.cookies.set(SESSION_COOKIE_NAME, create_session_token(owner_id))

    with patch("backend.app.main.settings.ingestion_client_rate_limit", 10), patch(
        "backend.app.main.settings.ingestion_owner_rate_limit", 1
    ), patch(
        "backend.app.main.discover_website_policies",
        new=AsyncMock(return_value=_empty_policy_document()),
    ) as discover:
        first = client.post("/api/v1/ingest/url", json={"url": "https://example.com"})
        second = client.post("/api/v1/ingest/url", json={"url": "https://example.com"})

    assert first.status_code == 200
    assert second.status_code == 429
    discover.assert_awaited_once()


def test_existing_owner_is_also_subject_to_client_ingestion_limit():
    owner_a = str(uuid4())
    owner_b = str(uuid4())

    with patch("backend.app.main.settings.ingestion_client_rate_limit", 1), patch(
        "backend.app.main.settings.ingestion_owner_rate_limit", 10
    ), patch(
        "backend.app.main.discover_website_policies",
        new=AsyncMock(return_value=_empty_policy_document()),
    ) as discover:
        client_a = TestClient(app)
        client_a.cookies.set(SESSION_COOKIE_NAME, create_session_token(owner_a))
        first = client_a.post("/api/v1/ingest/url", json={"url": "https://example.com"})

        client_b = TestClient(app)
        client_b.cookies.set(SESSION_COOKIE_NAME, create_session_token(owner_b))
        second = client_b.post("/api/v1/ingest/url", json={"url": "https://example.com"})

    assert first.status_code == 200
    assert second.status_code == 429
    discover.assert_awaited_once()


def test_clearing_or_replacing_cookies_does_not_bypass_client_limit():
    with patch("backend.app.main.settings.ingestion_client_rate_limit", 1), patch(
        "backend.app.main.settings.ingestion_owner_rate_limit", 10
    ), patch(
        "backend.app.main.discover_website_policies",
        new=AsyncMock(return_value=_empty_policy_document()),
    ) as discover:
        first_client = TestClient(app)
        first = first_client.post("/api/v1/ingest/url", json={"url": "https://example.com"})

        replacement_client = TestClient(app)
        replacement_client.cookies.set(SESSION_COOKIE_NAME, create_session_token(str(uuid4())))
        second = replacement_client.post("/api/v1/ingest/url", json={"url": "https://example.com"})

    assert first.status_code == 200
    assert second.status_code == 429
    discover.assert_awaited_once()


def test_malformed_session_cookie_does_not_bypass_client_limit():
    with patch("backend.app.main.settings.ingestion_client_rate_limit", 1), patch(
        "backend.app.main.settings.ingestion_owner_rate_limit", 10
    ), patch(
        "backend.app.main.discover_website_policies",
        new=AsyncMock(return_value=_empty_policy_document()),
    ) as discover:
        first = TestClient(app).post("/api/v1/ingest/url", json={"url": "https://example.com"})
        client = TestClient(app)
        client.cookies.set(SESSION_COOKIE_NAME, "malformed")
        second = client.post("/api/v1/ingest/url", json={"url": "https://example.com"})

    assert first.status_code == 200
    assert second.status_code == 429
    discover.assert_awaited_once()


def test_url_and_pdf_ingestion_behavior_remains_unchanged_under_limits():
    with patch("backend.app.main.settings.ingestion_client_rate_limit", 10), patch(
        "backend.app.main.settings.ingestion_owner_rate_limit", 10
    ), patch(
        "backend.app.main.discover_website_policies",
        new=AsyncMock(return_value=_empty_policy_document()),
    ), patch("backend.app.main.extract_pdf_text", return_value="Policy text"), patch(
        "backend.app.main.chunk_text", return_value=["chunk"]
    ), patch("backend.app.main.store_chunks") as store:
        client = TestClient(app)
        url_response = client.post("/api/v1/ingest/url", json={"url": "https://example.com"})
        pdf_response = client.post(
            "/api/v1/ingest/pdf",
            files={"file": ("policy.pdf", b"%PDF", "application/pdf")},
        )

    assert url_response.status_code == 200
    assert url_response.json()["status"] == "no_policies_found"
    assert pdf_response.status_code == 200
    assert pdf_response.json()["status"] == "success"
    assert store.call_count == 1
