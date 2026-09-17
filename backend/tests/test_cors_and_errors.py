from types import SimpleNamespace
from uuid import uuid4
from unittest.mock import patch
from contextlib import contextmanager

from fastapi import Response
from fastapi.testclient import TestClient

from backend.app.core.rate_limit import reset_rate_limits
from backend.app.core.session import (
    SESSION_COOKIE_NAME,
    _set_session_cookie,
    create_session_token,
)
from backend.app import main
from backend.app.main import app


@contextmanager
def production_origin_settings():
    """Rebuild the test application's static CORS middleware for this origin."""
    production_settings = SimpleNamespace(
        app_env="production",
        cors_allowed_origins=["https://policycue.vercel.app"],
        ask_rate_limit=20,
    )
    cors = next(middleware for middleware in app.user_middleware if middleware.cls.__name__ == "CORSMiddleware")
    original_settings = main.settings
    original_options = cors.kwargs.copy()
    main.settings = production_settings
    cors.kwargs["allow_origins"] = production_settings.cors_allowed_origins
    app.middleware_stack = None
    try:
        yield production_settings
    finally:
        main.settings = original_settings
        cors.kwargs.clear()
        cors.kwargs.update(original_options)
        app.middleware_stack = None


@app.get("/__test_unexpected_error")
async def __test_unexpected_error():
    raise RuntimeError("raw secret must not reach response")


def setup_function():
    reset_rate_limits()


def test_allowed_cors_origin_gets_credentials_headers():
    client = TestClient(app)

    response = client.options(
        "/health",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "GET",
        },
    )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://localhost:5173"
    assert response.headers["access-control-allow-credentials"] == "true"


def test_disallowed_cors_origin_is_not_reflected():
    client = TestClient(app)

    response = client.options(
        "/health",
        headers={
            "Origin": "https://evil.example",
            "Access-Control-Request-Method": "GET",
        },
    )

    assert response.headers.get("access-control-allow-origin") != "https://evil.example"


def test_security_headers_are_added_to_api_responses():
    client = TestClient(app)

    response = client.get("/health")

    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["referrer-policy"] == "no-referrer"


def test_session_cookie_is_httponly_secure_in_production_and_uses_configured_samesite():
    response = Response()
    owner_id = str(uuid4())
    production_settings = SimpleNamespace(
        app_env="production",
        session_cookie_samesite="none",
        secret_key="a" * 32,
    )

    with patch("backend.app.core.session.get_settings", return_value=production_settings):
        _set_session_cookie(response, owner_id)

    cookie = response.headers["set-cookie"]
    assert "HttpOnly" in cookie
    assert "Secure" in cookie
    assert "SameSite=none" in cookie
    assert "Path=/" in cookie
    assert "Domain=" not in cookie


def test_local_session_cookie_remains_lax_without_secure_flag():
    response = Response()
    development_settings = SimpleNamespace(
        app_env="development",
        session_cookie_samesite="lax",
        secret_key="a" * 32,
    )

    with patch("backend.app.core.session.get_settings", return_value=development_settings):
        _set_session_cookie(response, str(uuid4()))

    cookie = response.headers["set-cookie"]
    assert "HttpOnly" in cookie
    assert "SameSite=lax" in cookie
    assert "Secure" not in cookie


def test_production_allows_the_explicit_vercel_origin_for_cookie_authenticated_ask():
    owner_id = str(uuid4())
    with production_origin_settings(), patch(
        "backend.app.main.answer_question",
        return_value={"answer": "Grounded answer.", "sources": [{"text": "evidence"}], "document_found": True},
    ):
        client = TestClient(app)
        client.cookies.set(SESSION_COOKIE_NAME, create_session_token(owner_id))
        response = client.post(
            "/api/v1/ask",
            headers={"Origin": "https://policycue.vercel.app"},
            json={"question": "What is covered?", "document_id": str(uuid4())},
        )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "https://policycue.vercel.app"
    assert response.headers["access-control-allow-credentials"] == "true"


def test_production_rejects_missing_or_disallowed_origins_before_owner_lookup():
    with production_origin_settings(), patch(
        "backend.app.main.require_owner_id"
    ) as owner_lookup:
        client = TestClient(app)
        missing = client.post("/api/v1/ask", json={"question": "What is covered?", "document_id": str(uuid4())})
        disallowed = client.delete(
            f"/api/v1/documents/{uuid4()}",
            headers={"Origin": "https://evil.example"},
        )

    assert missing.status_code == 403
    assert disallowed.status_code == 403
    owner_lookup.assert_not_called()


def test_production_preflight_allows_only_the_explicit_vercel_origin():
    with production_origin_settings():
        client = TestClient(app)
        response = client.options(
            "/api/v1/ask",
            headers={
                "Origin": "https://policycue.vercel.app",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "content-type",
            },
        )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "https://policycue.vercel.app"
    assert response.headers["access-control-allow-origin"] != "*"


def test_known_http_exception_response_is_preserved_for_missing_session():
    client = TestClient(app)

    response = client.post(
        "/api/v1/ask",
        json={"question": "What is covered?", "document_id": str(uuid4())},
    )

    assert response.status_code == 404
    assert response.json() == {"detail": "Document not found."}


def test_unexpected_exception_returns_generic_500_without_raw_error_text():
    client = TestClient(app, raise_server_exceptions=False)

    response = client.get("/__test_unexpected_error")

    assert response.status_code == 500
    assert response.json() == {"detail": "Internal server error."}
    assert "raw secret" not in response.text


def test_rate_limit_429_response_remains_controlled():
    owner_id = str(uuid4())
    client = TestClient(app)
    client.cookies.set(SESSION_COOKIE_NAME, create_session_token(owner_id))

    with patch("backend.app.main.settings.ask_rate_limit", 1), patch(
        "backend.app.main.answer_question",
        return_value={"answer": "ok", "sources": [{"text": "x"}]},
    ):
        first = client.post(
            "/api/v1/ask",
            json={"question": "One?", "document_id": str(uuid4())},
        )
        second = client.post(
            "/api/v1/ask",
            json={"question": "Two?", "document_id": str(uuid4())},
        )

    assert first.status_code == 200
    assert second.status_code == 429
    assert second.json()["detail"] == "Too many requests. Please retry later."


def test_question_answering_503_response_remains_controlled():
    owner_id = str(uuid4())
    client = TestClient(app)
    client.cookies.set(SESSION_COOKIE_NAME, create_session_token(owner_id))

    with patch("backend.app.main.answer_question", side_effect=RuntimeError("provider secret")):
        response = client.post(
            "/api/v1/ask",
            json={"question": "What is covered?", "document_id": str(uuid4())},
        )

    assert response.status_code == 503
    assert response.json() == {
        "detail": "Question answering is temporarily unavailable. Please retry later."
    }
    assert "provider secret" not in response.text
