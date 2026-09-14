from unittest.mock import patch

import pytest
from fastapi import HTTPException
from starlette.requests import Request

from backend.app.core.rate_limit import (
    _buckets,
    check_rate_limit,
    reset_rate_limits,
)
from backend.app.core.session import SESSION_COOKIE_NAME, create_session_token


def _request(cookie: str | None = None, host: str = "127.0.0.1"):
    headers = []
    if cookie:
        headers.append((b"cookie", cookie.encode()))
    return Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/",
            "headers": headers,
            "client": (host, 12345),
        }
    )


def _session_cookie(owner_id: str) -> str:
    return f"{SESSION_COOKIE_NAME}={create_session_token(owner_id)}"


def setup_function():
    reset_rate_limits()


def test_request_under_limit_is_allowed():
    request = _request(_session_cookie("11111111-1111-4111-8111-111111111111"))

    check_rate_limit(request, scope="ask", limit=2, window_seconds=60)

    assert len(_buckets) == 1


def test_request_exceeding_ask_limit_returns_429_with_retry_after():
    request = _request(_session_cookie("11111111-1111-4111-8111-111111111111"))

    with patch("backend.app.core.rate_limit.time.monotonic", side_effect=[0, 1]):
        check_rate_limit(request, scope="ask", limit=1, window_seconds=60)
        with pytest.raises(HTTPException) as caught:
            check_rate_limit(request, scope="ask", limit=1, window_seconds=60)

    assert caught.value.status_code == 429
    assert caught.value.detail == "Too many requests. Please retry later."
    assert caught.value.headers == {"Retry-After": "59"}


def test_request_exceeding_ingestion_limit_returns_429():
    request = _request(_session_cookie("11111111-1111-4111-8111-111111111111"))

    check_rate_limit(request, scope="ingest", limit=1, window_seconds=60)
    with pytest.raises(HTTPException) as caught:
        check_rate_limit(request, scope="ingest", limit=1, window_seconds=60)

    assert caught.value.status_code == 429


def test_separate_owners_do_not_share_limit():
    owner_a = _request(_session_cookie("11111111-1111-4111-8111-111111111111"))
    owner_b = _request(_session_cookie("22222222-2222-4222-8222-222222222222"))

    check_rate_limit(owner_a, scope="ask", limit=1, window_seconds=60)
    check_rate_limit(owner_b, scope="ask", limit=1, window_seconds=60)

    assert len(_buckets) == 2


def test_missing_or_invalid_session_cannot_bypass_limiter():
    missing = _request(host="203.0.113.7")
    invalid = _request(f"{SESSION_COOKIE_NAME}=invalid", host="203.0.113.7")

    check_rate_limit(missing, scope="ask", limit=1, window_seconds=60)
    with pytest.raises(HTTPException) as caught:
        check_rate_limit(invalid, scope="ask", limit=1, window_seconds=60)

    assert caught.value.status_code == 429


def test_stale_entries_are_cleaned_and_memory_is_bounded(monkeypatch):
    monkeypatch.setattr(
        "backend.app.core.rate_limit.get_settings",
        lambda: type("Settings", (), {"rate_limit_max_keys": 2})(),
    )

    with patch("backend.app.core.rate_limit.time.monotonic", return_value=0):
        check_rate_limit(_request(host="203.0.113.1"), scope="ask", limit=2, window_seconds=10)
        check_rate_limit(_request(host="203.0.113.2"), scope="ask", limit=2, window_seconds=10)
    with patch("backend.app.core.rate_limit.time.monotonic", return_value=20):
        check_rate_limit(_request(host="203.0.113.3"), scope="ask", limit=2, window_seconds=10)
        check_rate_limit(_request(host="203.0.113.4"), scope="ask", limit=2, window_seconds=10)
        check_rate_limit(_request(host="203.0.113.5"), scope="ask", limit=2, window_seconds=10)

    assert len(_buckets) <= 2
