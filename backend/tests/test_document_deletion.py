from unittest.mock import patch

from fastapi.testclient import TestClient

from backend.app.core.rate_limit import reset_rate_limits
from backend.app.core.session import SESSION_COOKIE_NAME, create_session_token
from backend.app.main import app


OWNER_A = "11111111-1111-4111-8111-111111111111"
OWNER_B = "22222222-2222-4222-8222-222222222222"
DOCUMENT_ID = "33333333-3333-4333-8333-333333333333"


def setup_function():
    reset_rate_limits()


def _client(owner_id: str | None = OWNER_A) -> TestClient:
    client = TestClient(app)
    if owner_id:
        client.cookies.set(SESSION_COOKIE_NAME, create_session_token(owner_id))
    return client


def test_owner_can_delete_owned_document():
    client = _client(OWNER_A)

    with patch("backend.app.main.delete_document_vectors") as delete:
        response = client.delete(f"/api/v1/documents/{DOCUMENT_ID}")

    assert response.status_code == 200
    assert response.json() == {"status": "deleted", "document_id": DOCUMENT_ID}
    delete.assert_called_once_with(DOCUMENT_ID, OWNER_A)


def test_owner_cannot_delete_another_owner_document():
    client = _client(OWNER_A)

    with patch("backend.app.main.delete_document_vectors") as delete:
        response = client.delete(f"/api/v1/documents/{DOCUMENT_ID}")

    assert response.status_code == 200
    delete.assert_called_once_with(DOCUMENT_ID, OWNER_A)


def test_wrong_owner_cannot_affect_target_vectors():
    client = _client(OWNER_B)

    with patch("backend.app.main.delete_document_vectors") as delete:
        response = client.delete(f"/api/v1/documents/{DOCUMENT_ID}")

    assert response.status_code == 200
    delete.assert_called_once_with(DOCUMENT_ID, OWNER_B)


def test_nonexistent_document_deletion_is_idempotent_and_does_not_leak_existence():
    client = _client(OWNER_A)

    with patch("backend.app.main.delete_document_vectors") as delete:
        first = client.delete(f"/api/v1/documents/{DOCUMENT_ID}")
        second = client.delete(f"/api/v1/documents/{DOCUMENT_ID}")

    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json() == second.json() == {
        "status": "deleted",
        "document_id": DOCUMENT_ID,
    }
    assert delete.call_count == 2


def test_client_cannot_supply_or_override_owner_id():
    client = _client(OWNER_A)

    with patch("backend.app.main.delete_document_vectors") as delete:
        response = client.request(
            "DELETE",
            f"/api/v1/documents/{DOCUMENT_ID}?owner_id={OWNER_B}",
            json={"owner_id": OWNER_B},
        )

    assert response.status_code == 200
    delete.assert_called_once_with(DOCUMENT_ID, OWNER_A)


def test_invalid_session_fails_safely_without_delete():
    client = TestClient(app)
    client.cookies.set(SESSION_COOKIE_NAME, "invalid-token")

    with patch("backend.app.main.delete_document_vectors") as delete:
        response = client.delete(f"/api/v1/documents/{DOCUMENT_ID}")

    assert response.status_code == 404
    assert response.json() == {"detail": "Document not found."}
    delete.assert_not_called()


def test_malformed_document_id_fails_safely_without_delete():
    client = _client(OWNER_A)

    with patch("backend.app.main.delete_document_vectors") as delete:
        response = client.delete("/api/v1/documents/not-a-document-id")

    assert response.status_code == 404
    assert response.json() == {"detail": "Document not found."}
    delete.assert_not_called()


def test_permanent_deletion_failure_returns_controlled_503():
    client = _client(OWNER_A)

    with patch("backend.app.main.delete_document_vectors", side_effect=RuntimeError("pinecone down")):
        response = client.delete(f"/api/v1/documents/{DOCUMENT_ID}")

    assert response.status_code == 503
    assert response.json() == {
        "detail": "Document deletion is temporarily unavailable. Please retry later."
    }


def test_ask_cannot_retrieve_after_successful_delete_using_mocked_state():
    owner_id = OWNER_A
    document_id = DOCUMENT_ID
    state = {"deleted": False}
    client = _client(owner_id)

    def delete(document: str, owner: str) -> None:
        assert document == document_id
        assert owner == owner_id
        state["deleted"] = True

    def answer(question: str, top_k: int, document: str, owner: str) -> dict:
        if state["deleted"]:
            return {"answer": "I could not find that information in the provided document.", "sources": []}
        return {"answer": "Grounded answer.", "sources": [{"document_id": document, "owner_id": owner}]}

    with patch("backend.app.main.delete_document_vectors", side_effect=delete), patch("backend.app.main.answer_question", side_effect=answer):
        deleted = client.delete(f"/api/v1/documents/{document_id}")
        asked = client.post(
            "/api/v1/ask",
            json={"question": "What does it say?", "document_id": document_id},
        )

    assert deleted.status_code == 200
    assert asked.status_code == 404
    assert asked.json() == {"detail": "Document not found."}
