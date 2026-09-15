from unittest.mock import patch

from fastapi.testclient import TestClient

from backend.app.analysis.answer_generator import NO_ANSWER_MESSAGE
from backend.app.core.rate_limit import reset_rate_limits
from backend.app.core.session import SESSION_COOKIE_NAME, create_session_token
from backend.app.main import app


def setup_function():
    reset_rate_limits()


def test_empty_whitespace_and_short_text_rejected():
    for text, detail in [("", "Text is required."), ("   \n\t", "Text is required."), ("too short", "Text is too short to analyze.")]:
        reset_rate_limits()
        client = TestClient(app)
        response = client.post("/api/v1/ingest/text", json={"text": text})
        assert response.status_code == 400
        assert response.json()["detail"] == detail


def test_oversized_text_rejected_without_storage():
    client = TestClient(app)
    with patch("backend.app.main.store_chunks") as store:
        response = client.post("/api/v1/ingest/text", json={"text": "a" * 100_001})
    assert response.status_code == 413
    assert response.json()["detail"] == "Text is too large to paste directly. Upload it as a PDF instead."
    store.assert_not_called()


def test_valid_small_text_stores_owner_and_title_metadata():
    owner_id = "11111111-1111-4111-8111-111111111111"
    client = TestClient(app)
    client.cookies.set(SESSION_COOKIE_NAME, create_session_token(owner_id))
    text = "Privacy Policy\n" + "We collect personal information and share data with service providers. " * 2

    with patch("backend.app.main.chunk_text", return_value=["chunk"]), patch("backend.app.main.store_chunks") as store:
        response = client.post("/api/v1/ingest/text", json={"text": text, "title": " My Policy "})

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "success"
    assert body["source_type"] == "text"
    assert body["title"] == "My Policy"
    store.assert_called_once()
    assert store.call_args.kwargs["owner_id"] == owner_id
    assert store.call_args.kwargs["chunk_metadata"] == [{"source_type": "text", "title": "My Policy"}]


def test_missing_title_uses_pasted_text_and_no_fake_source_url():
    client = TestClient(app)
    text = "Terms of Service\n" + "Users agree to follow these terms and may have access terminated. " * 2
    with patch("backend.app.main.chunk_text", return_value=["chunk"]), patch("backend.app.main.store_chunks") as store:
        response = client.post("/api/v1/ingest/text", json={"text": text})
    assert response.status_code == 200
    assert response.json()["title"] == "Pasted text"
    assert store.call_args.kwargs["chunk_metadata"] == [{"source_type": "text", "title": "Pasted text"}]


def test_100000_chars_accepted_and_large_text_creates_multiple_chunks():
    client = TestClient(app)
    text = "Privacy Policy\n" + "a" * (100_000 - len("Privacy Policy\n"))
    with patch("backend.app.main.chunk_text", return_value=["chunk1", "chunk2"]), patch("backend.app.main.store_chunks") as store:
        response = client.post("/api/v1/ingest/text", json={"text": text})
    assert response.status_code == 200
    assert response.json()["chunks"] == 2
    assert store.call_args.args[1] == ["chunk1", "chunk2"]


def test_text_storage_failure_triggers_cleanup():
    owner_id = "22222222-2222-4222-8222-222222222222"
    client = TestClient(app)
    client.cookies.set(SESSION_COOKIE_NAME, create_session_token(owner_id))
    text = "Privacy Policy\n" + "We collect personal information and share data. " * 3
    with patch("backend.app.main.uuid4", return_value="33333333-3333-4333-8333-333333333333"), patch("backend.app.main.chunk_text", return_value=["chunk"]), patch("backend.app.main.store_chunks", side_effect=RuntimeError("boom")), patch("backend.app.main.delete_document_vectors") as cleanup:
        response = client.post("/api/v1/ingest/text", json={"text": text})
    assert response.status_code == 503
    assert response.json()["detail"] == "Text storage failed. Please retry ingestion."
    cleanup.assert_called_once_with("33333333-3333-4333-8333-333333333333", owner_id)


def test_text_wrong_owner_404_and_authorized_abstention_200():
    owner_id = "44444444-4444-4444-8444-444444444444"
    client = TestClient(app)
    client.cookies.set(SESSION_COOKIE_NAME, create_session_token(owner_id))
    with patch("backend.app.main.answer_question", return_value={"answer": NO_ANSWER_MESSAGE, "sources": [], "source_attributions": [], "document_found": False}):
        wrong = client.post("/api/v1/ask", json={"document_id": "doc", "question": "Who founded Amazon?"})
    assert wrong.status_code == 404
    with patch("backend.app.main.answer_question", return_value={"answer": NO_ANSWER_MESSAGE, "sources": [], "source_attributions": [], "document_found": True}):
        supported_doc = client.post("/api/v1/ask", json={"document_id": "doc", "question": "Who founded Amazon?"})
    assert supported_doc.status_code == 200
    assert supported_doc.json()["sources"] == []


def test_delete_text_document_is_owner_scoped():
    owner_id = "55555555-5555-4555-8555-555555555555"
    document_id = "66666666-6666-4666-8666-666666666666"
    client = TestClient(app)
    client.cookies.set(SESSION_COOKIE_NAME, create_session_token(owner_id))
    with patch("backend.app.main.delete_document_vectors") as delete:
        response = client.delete(f"/api/v1/documents/{document_id}")
    assert response.status_code == 200
    delete.assert_called_once_with(document_id, owner_id)
