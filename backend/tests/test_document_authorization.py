from unittest.mock import patch

from fastapi.testclient import TestClient

from backend.app.core.session import (
    SESSION_MAX_AGE_SECONDS,
    SESSION_COOKIE_NAME,
    create_session_token,
    read_owner_id_from_token,
)
from backend.app.core.rate_limit import reset_rate_limits
from backend.app.main import app


def setup_function():
    reset_rate_limits()


def test_text_ingestion_route_is_available_and_validates_input():
    client = TestClient(app)

    response = client.post(
        "/api/v1/ingest/text",
        json={"text": "Policy text"},
    )

    assert response.status_code == 400
    assert response.json() == {"detail": "Text is too short to analyze."}


def test_owner_can_query_own_document():
    owner_id = "11111111-1111-4111-8111-111111111111"
    client = TestClient(app)
    client.cookies.set(SESSION_COOKIE_NAME, create_session_token(owner_id))

    with patch(
        "backend.app.main.answer_question",
        return_value={
            "answer": "Grounded answer.",
            "sources": [{"document_id": "doc-a", "owner_id": owner_id}],
        },
    ) as answer:
        response = client.post(
            "/api/v1/ask",
            json={"question": "What does it say?", "document_id": "doc-a"},
        )

    assert response.status_code == 200
    answer.assert_called_once_with(
        "What does it say?",
        5,
        "doc-a",
        owner_id,
    )


def test_owner_cannot_query_another_owners_document():
    owner_id = "11111111-1111-4111-8111-111111111111"
    client = TestClient(app)
    client.cookies.set(SESSION_COOKIE_NAME, create_session_token(owner_id))

    with patch(
        "backend.app.main.answer_question",
        return_value={"answer": "I could not find that information in the provided document.", "sources": []},
    ):
        response = client.post(
            "/api/v1/ask",
            json={"question": "What does it say?", "document_id": "doc-b"},
        )

    assert response.status_code == 404
    assert response.json() == {"detail": "Document not found."}




def test_authorized_document_with_insufficient_evidence_returns_no_answer_not_404():
    owner_id = "11111111-1111-4111-8111-111111111111"
    client = TestClient(app)
    client.cookies.set(SESSION_COOKIE_NAME, create_session_token(owner_id))

    with patch(
        "backend.app.main.answer_question",
        return_value={
            "answer": "I could not find that information in the provided document.",
            "sources": [],
            "source_attributions": [],
            "document_found": True,
        },
    ):
        response = client.post(
            "/api/v1/ask",
            json={"question": "Who won the latest FIFA World Cup?", "document_id": "doc-a"},
        )

    assert response.status_code == 200
    assert response.json() == {
        "answer": "I could not find that information in the provided document.",
        "sources": [],
        "source_attributions": [],
        "document_found": True,
    }


def test_invalid_session_cookie_fails_safely_without_lookup():
    client = TestClient(app)
    client.cookies.set(SESSION_COOKIE_NAME, "not-a-valid-token")

    with patch("backend.app.main.answer_question") as answer:
        response = client.post(
            "/api/v1/ask",
            json={"question": "What does it say?", "document_id": "doc-a"},
        )

    assert response.status_code == 404
    assert response.json() == {"detail": "Document not found."}
    answer.assert_not_called()


def test_expired_session_token_fails_safely():
    owner_id = "11111111-1111-4111-8111-111111111111"

    with patch("backend.app.core.session.time.time", return_value=0):
        token = create_session_token(owner_id)
    with patch(
        "backend.app.core.session.time.time",
        return_value=SESSION_MAX_AGE_SECONDS + 2,
    ):
        assert read_owner_id_from_token(token) is None


def test_unauthorized_responses_do_not_reveal_document_existence():
    client = TestClient(app)

    existing = client.post(
        "/api/v1/ask",
        json={"question": "What does it say?", "document_id": "existing-doc"},
    )
    missing = client.post(
        "/api/v1/ask",
        json={"question": "What does it say?", "document_id": "missing-doc"},
    )

    assert existing.status_code == 404
    assert missing.status_code == 404
    assert existing.json() == missing.json() == {"detail": "Document not found."}


def test_ask_rate_limit_returns_429_before_expensive_work():
    owner_id = "11111111-1111-4111-8111-111111111111"
    client = TestClient(app)
    client.cookies.set(SESSION_COOKIE_NAME, create_session_token(owner_id))

    with patch("backend.app.main.settings.ask_rate_limit", 1), patch("backend.app.main.answer_question", return_value={"answer": "ok", "sources": [{"text": "x"}]}) as answer:
        first = client.post(
            "/api/v1/ask",
            json={"question": "What does it say?", "document_id": "doc-a"},
        )
        second = client.post(
            "/api/v1/ask",
            json={"question": "What does it say?", "document_id": "doc-a"},
        )

    assert first.status_code == 200
    assert second.status_code == 429
    assert second.headers["Retry-After"]
    assert second.json() == {"detail": "Too many requests. Please retry later."}
    answer.assert_called_once()


def test_missing_session_is_still_rate_limited():
    client = TestClient(app)

    with patch("backend.app.main.settings.ask_rate_limit", 1), patch("backend.app.main.answer_question") as answer:
        first = client.post(
            "/api/v1/ask",
            json={"question": "What does it say?", "document_id": "doc-a"},
        )
        second = client.post(
            "/api/v1/ask",
            json={"question": "What does it say?", "document_id": "doc-a"},
        )

    assert first.status_code == 404
    assert second.status_code == 429
    answer.assert_not_called()


def test_separate_owner_ask_limits_do_not_collide():
    owner_a = "11111111-1111-4111-8111-111111111111"
    owner_b = "22222222-2222-4222-8222-222222222222"
    client_a = TestClient(app)
    client_b = TestClient(app)
    client_a.cookies.set(SESSION_COOKIE_NAME, create_session_token(owner_a))
    client_b.cookies.set(SESSION_COOKIE_NAME, create_session_token(owner_b))

    with patch("backend.app.main.settings.ask_rate_limit", 1), patch("backend.app.main.answer_question", return_value={"answer": "ok", "sources": [{"text": "x"}]}) as answer:
        response_a = client_a.post(
            "/api/v1/ask",
            json={"question": "What does it say?", "document_id": "doc-a"},
        )
        response_b = client_b.post(
            "/api/v1/ask",
            json={"question": "What does it say?", "document_id": "doc-b"},
        )

    assert response_a.status_code == 200
    assert response_b.status_code == 200
    assert answer.call_count == 2


def test_generation_failure_returns_controlled_service_unavailable():
    owner_id = "11111111-1111-4111-8111-111111111111"
    client = TestClient(app)
    client.cookies.set(SESSION_COOKIE_NAME, create_session_token(owner_id))

    with patch("backend.app.main.answer_question", side_effect=RuntimeError("gemini unavailable")):
        response = client.post(
            "/api/v1/ask",
            json={"question": "What does it say?", "document_id": "doc-a"},
        )

    assert response.status_code == 503
    assert response.json() == {
        "detail": "Question answering is temporarily unavailable. Please retry later."
    }
