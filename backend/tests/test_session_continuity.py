from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from backend.app import main
from backend.app.core.rate_limit import reset_rate_limits
from backend.app.core.session import (
    SESSION_COOKIE_NAME,
    SESSION_HEADER_NAME,
    SESSION_MAX_AGE_SECONDS,
    create_session_token,
    read_owner_id_from_token,
)
from backend.app.ingestion.policy_categories import PolicyCategory


OWNER_A = "11111111-1111-4111-8111-111111111111"
OWNER_B = "22222222-2222-4222-8222-222222222222"
DOCUMENT_ID = "33333333-3333-4333-8333-333333333333"
PRIVATE_QUESTION = "Private question that must not be logged"


def setup_function():
    reset_rate_limits()


def _ingestion_response(source: str, client: TestClient):
    page = {
        "url": "https://example.com/privacy",
        "source_urls": ["https://example.com/privacy"],
        "text": "Policy text with enough detail for an ingestion test.",
        "categories": [PolicyCategory.PRIVACY],
        "content_hash": "test-hash",
    }
    document = {
        "url": "https://example.com",
        "pages": [page],
        "warnings": [],
        "skipped": {},
        "candidates": 1,
    }
    with patch.object(
        main,
        "discover_website_policies",
        new=AsyncMock(return_value=document),
    ), patch.object(
        main,
        "extract_pdf_text",
        return_value=page["text"],
    ), patch.object(
        main,
        "chunk_text",
        return_value=[page["text"]],
    ), patch.object(main, "store_chunks") as store:
        if source == "url":
            response = client.post(
                "/api/v1/ingest/url",
                json={"url": "https://example.com"},
            )
        elif source == "pdf":
            response = client.post(
                "/api/v1/ingest/pdf",
                files={"file": ("policy.pdf", b"%PDF", "application/pdf")},
            )
        else:
            response = client.post(
                "/api/v1/ingest/text",
                json={"text": page["text"] * 2},
            )
    return response, store


@pytest.mark.parametrize("source", ["url", "pdf", "text"])
def test_ingestion_returns_token_for_the_owner_used_during_storage(source):
    response, store = _ingestion_response(source, TestClient(main.app))

    assert response.status_code == 200
    body = response.json()
    token = body["session_token"]
    owner_id = read_owner_id_from_token(token)
    assert owner_id
    assert store.call_args.kwargs["owner_id"] == owner_id
    assert body["document_id"] == store.call_args.args[0]
    assert "owner_id" not in body
    assert main.settings.secret_key not in response.text
    assert SESSION_COOKIE_NAME in response.cookies


def test_header_session_reaches_owner_scoped_ask_without_cookie(caplog):
    token = create_session_token(OWNER_A)
    client = TestClient(main.app)
    with patch.object(
        main,
        "answer_question",
        return_value={
            "answer": "Grounded answer.",
            "sources": [{"text": "evidence"}],
            "source_attributions": [],
            "document_found": True,
        },
    ) as answer:
        response = client.post(
            "/api/v1/ask",
            headers={SESSION_HEADER_NAME: token},
            json={"question": PRIVATE_QUESTION, "document_id": DOCUMENT_ID},
        )

    assert response.status_code == 200
    answer.assert_called_once_with(PRIVATE_QUESTION, 5, DOCUMENT_ID, OWNER_A)
    assert token not in caplog.text
    assert PRIVATE_QUESTION not in caplog.text


@pytest.mark.parametrize("kind", ["invalid", "expired"])
def test_invalid_or_expired_header_session_returns_404_without_lookup(kind):
    token = "invalid-token"
    if kind == "expired":
        with patch("backend.app.core.session.time.time", return_value=0):
            token = create_session_token(OWNER_A)
        time_patch = patch(
            "backend.app.core.session.time.time",
            return_value=SESSION_MAX_AGE_SECONDS + 1,
        )
    else:
        time_patch = patch("backend.app.core.session.time.time", wraps=__import__("time").time)

    client = TestClient(main.app)
    with time_patch, patch.object(main, "answer_question") as answer:
        response = client.post(
            "/api/v1/ask",
            headers={SESSION_HEADER_NAME: token},
            json={"question": PRIVATE_QUESTION, "document_id": DOCUMENT_ID},
        )

    assert response.status_code == 404
    assert response.json() == {"detail": "Document not found."}
    answer.assert_not_called()


def test_cookie_fallback_still_reaches_ask():
    client = TestClient(main.app)
    client.cookies.set(SESSION_COOKIE_NAME, create_session_token(OWNER_A))
    with patch.object(
        main,
        "answer_question",
        return_value={"answer": "ok", "sources": [{"text": "evidence"}]},
    ) as answer:
        response = client.post(
            "/api/v1/ask",
            json={"question": "What?", "document_id": DOCUMENT_ID},
        )
    assert response.status_code == 200
    answer.assert_called_once_with("What?", 5, DOCUMENT_ID, OWNER_A)


def test_header_precedes_cookie_and_invalid_header_never_falls_back():
    client = TestClient(main.app)
    client.cookies.set(SESSION_COOKIE_NAME, create_session_token(OWNER_A))
    with patch.object(main, "answer_question") as answer:
        invalid = client.post(
            "/api/v1/ask",
            headers={SESSION_HEADER_NAME: "invalid-token"},
            json={"question": "What?", "document_id": DOCUMENT_ID},
        )
    assert invalid.status_code == 404
    answer.assert_not_called()

    with patch.object(
        main,
        "answer_question",
        return_value={"answer": "ok", "sources": [{"text": "evidence"}]},
    ) as answer:
        valid = client.post(
            "/api/v1/ask",
            headers={SESSION_HEADER_NAME: create_session_token(OWNER_B)},
            json={"question": "What?", "document_id": DOCUMENT_ID},
        )
    assert valid.status_code == 200
    answer.assert_called_once_with("What?", 5, DOCUMENT_ID, OWNER_B)


def test_explicit_session_is_reused_by_later_ingestion():
    token = create_session_token(OWNER_A)
    response, store = _ingestion_response(
        "text",
        TestClient(main.app, headers={SESSION_HEADER_NAME: token}),
    )
    assert response.status_code == 200
    assert response.json()["session_token"] == token
    assert store.call_args.kwargs["owner_id"] == OWNER_A


def test_invalid_explicit_session_cannot_start_a_replacement_owner():
    client = TestClient(main.app, headers={SESSION_HEADER_NAME: "invalid-token"})
    with patch.object(main, "store_chunks") as store:
        response = client.post(
            "/api/v1/ingest/text",
            json={"text": "A sufficiently long policy statement for ingestion." * 2},
        )
    assert response.status_code == 404
    assert response.json() == {"detail": "Document not found."}
    assert "session_token" not in response.json()
    store.assert_not_called()


def test_delete_with_header_is_owner_scoped():
    token = create_session_token(OWNER_A)
    with patch.object(main, "delete_document_vectors") as delete:
        response = TestClient(main.app).delete(
            f"/api/v1/documents/{DOCUMENT_ID}",
            headers={SESSION_HEADER_NAME: token},
        )
    assert response.status_code == 200
    delete.assert_called_once_with(DOCUMENT_ID, OWNER_A)


def test_different_valid_owner_cannot_retrieve_or_delete_as_original_owner():
    token = create_session_token(OWNER_B)
    client = TestClient(main.app, headers={SESSION_HEADER_NAME: token})
    with patch.object(
        main,
        "answer_question",
        return_value={"answer": "No answer", "sources": [], "document_found": False},
    ) as answer:
        asked = client.post(
            "/api/v1/ask",
            json={"question": "What?", "document_id": DOCUMENT_ID},
        )
    assert asked.status_code == 404
    answer.assert_called_once_with("What?", 5, DOCUMENT_ID, OWNER_B)

    with patch.object(main, "delete_document_vectors") as delete:
        deleted = client.delete(f"/api/v1/documents/{DOCUMENT_ID}")
    assert deleted.status_code == 200
    delete.assert_called_once_with(DOCUMENT_ID, OWNER_B)


def test_preflight_allows_the_dedicated_session_header():
    response = TestClient(main.app).options(
        "/api/v1/ask",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": SESSION_HEADER_NAME,
        },
    )
    assert response.status_code == 200
    assert SESSION_HEADER_NAME.lower() in response.headers["access-control-allow-headers"].lower()
