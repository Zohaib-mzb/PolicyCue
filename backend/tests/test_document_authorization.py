from unittest.mock import patch

from fastapi.testclient import TestClient

from backend.app.core.session import (
    SESSION_MAX_AGE_SECONDS,
    SESSION_COOKIE_NAME,
    create_session_token,
    read_owner_id_from_token,
)
from backend.app.main import app


def test_ingestion_issues_owner_cookie_and_stores_owner_metadata():
    client = TestClient(app)

    with patch("backend.app.main.process_text", return_value={"text": "Policy text"}), patch("backend.app.main.chunk_text", return_value=["chunk"]), patch("backend.app.main.store_chunks") as store:
        response = client.post(
            "/api/v1/ingest/text",
            json={"text": "Policy text"},
        )

    assert response.status_code == 200
    token = response.cookies.get(SESSION_COOKIE_NAME)
    owner_id = read_owner_id_from_token(token)
    assert owner_id
    assert store.call_args.kwargs["owner_id"] == owner_id


def test_text_ingestion_storage_failure_cleans_up_owner_document():
    owner_id = "11111111-1111-4111-8111-111111111111"
    client = TestClient(app)
    client.cookies.set(SESSION_COOKIE_NAME, create_session_token(owner_id))

    with patch("backend.app.main.uuid4", return_value="failed-text-document"), patch("backend.app.main.process_text", return_value={"text": "Policy text"}), patch("backend.app.main.chunk_text", return_value=["chunk"]), patch("backend.app.main.store_chunks", side_effect=RuntimeError("upsert failed")), patch("backend.app.main.delete_document_vectors") as cleanup:
        response = client.post(
            "/api/v1/ingest/text",
            json={"text": "Policy text"},
        )

    assert response.status_code == 503
    assert response.json() == {"detail": "Document storage failed. Please retry ingestion."}
    cleanup.assert_called_once_with("failed-text-document", owner_id)


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
