from http.cookies import SimpleCookie
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from backend.app import main
from backend.app.core import session
from backend.app.core.rate_limit import reset_rate_limits
from backend.app.ingestion.policy_categories import PolicyCategory


ORIGIN = "https://policycue.vercel.app"
TEST_SECRET = "session-continuity-test-secret-00000000"


@pytest.fixture
def production_client(monkeypatch):
    settings = main.settings.model_copy(update={
        "app_env": "production",
        "session_cookie_samesite": "none",
        "secret_key": TEST_SECRET,
        "cors_allowed_origins": [ORIGIN],
    })
    monkeypatch.setattr(main, "settings", settings)
    monkeypatch.setattr(session, "get_settings", lambda: settings)
    cors = next(m for m in main.app.user_middleware if m.cls.__name__ == "CORSMiddleware")
    monkeypatch.setitem(cors.kwargs, "allow_origins", [ORIGIN])
    main.app.middleware_stack = None
    reset_rate_limits()
    try:
        with TestClient(main.app, base_url="https://backend.example", headers={"Origin": ORIGIN}) as client:
            yield client
    finally:
        main.app.middleware_stack = None
        reset_rate_limits()


@pytest.mark.parametrize("source", ["url", "pdf", "text"])
def test_successful_ingestion_issues_cookie_and_same_owner_reaches_retrieval(
    source, production_client, caplog,
):
    page = {
        "url": "https://example.com/privacy",
        "source_urls": ["https://example.com/privacy"],
        "text": "Private document text that must not be logged.",
        "categories": [PolicyCategory.PRIVACY],
        "content_hash": "test-hash",
    }
    document = {
        "url": "https://example.com", "pages": [page],
        "warnings": [], "skipped": {}, "candidates": 1,
    }
    with patch.object(main, "discover_website_policies", new=AsyncMock(return_value=document)), patch.object(
        main, "extract_pdf_text", return_value=page["text"],
    ), patch.object(main, "chunk_text", return_value=[page["text"]]), patch.object(main, "store_chunks") as store:
        if source == "pdf":
            response = production_client.post(
                "/api/v1/ingest/pdf", files={"file": ("policy.pdf", b"%PDF", "application/pdf")},
            )
        else:
            body = {"url": "https://example.com"} if source == "url" else {"text": page["text"] * 3}
            response = production_client.post(f"/api/v1/ingest/{source}", json=body)

    assert response.status_code == 200
    assert response.json()["status"] == "success"
    assert response.headers["access-control-allow-origin"] == ORIGIN
    assert response.headers["access-control-allow-credentials"] == "true"
    cookies = SimpleCookie()
    cookies.load(response.headers["set-cookie"])
    cookie = cookies[session.SESSION_COOKIE_NAME]
    assert cookie["httponly"] and cookie["secure"]
    assert cookie["samesite"].lower() == "none"
    assert cookie["path"] == "/"
    assert not cookie["domain"]
    assert int(cookie["max-age"]) == session.SESSION_MAX_AGE_SECONDS
    owner = session.read_owner_id_from_token(cookie.value)
    assert owner and store.call_args.kwargs["owner_id"] == owner
    document_id = response.json()["document_id"]
    assert store.call_args.args[0] == document_id

    # Keep real /ask, session verification, and answer_question orchestration.
    # Stop at the retrieval boundary; no external provider is contacted.
    question = "Private question that must not be logged"
    with patch("backend.app.retrieval.vector_store.search_chunks", return_value=[{"text": page["text"]}]) as search, patch(
        "backend.app.analysis.answer_generator.generate_answer", return_value="Grounded answer.",
    ):
        answer = production_client.post("/api/v1/ask", json={"document_id": document_id, "question": question})

    assert answer.status_code == 200
    assert answer.json()["document_found"] is True
    search.assert_called_once_with(query=question, top_k=5, document_id=document_id, owner_id=owner)
    assert "set-cookie" not in answer.headers
    assert answer.headers["access-control-allow-origin"] == ORIGIN
    assert answer.headers["access-control-allow-credentials"] == "true"
    for sensitive in (cookie.value, owner, TEST_SECRET, question, page["text"]):
        assert sensitive not in caplog.text
    assert "ask_session_rejected" not in caplog.text


@pytest.mark.parametrize("failure", [
    "missing_cookie", "malformed_cookie", "invalid_signature", "expired_cookie", "invalid_payload",
])
def test_session_rejections_are_private_and_keep_identical_404(failure, production_client, caplog):
    owner = str(uuid4())
    token = session.create_session_token(owner)
    if failure == "malformed_cookie":
        token = "malformed-private-cookie"
    elif failure == "invalid_signature":
        payload, signature = token.split(".")
        token = f"{payload}.{'A' if signature[0] != 'A' else 'B'}{signature[1:]}"
    elif failure == "expired_cookie":
        with patch.object(session.time, "time", return_value=0):
            token = session.create_session_token(owner)
    elif failure == "invalid_payload":
        payload = session._b64encode(b"invalid-private-payload")
        token = f"{payload}.{session._sign(payload)}"
    if failure != "missing_cookie":
        production_client.cookies.set(session.SESSION_COOKIE_NAME, token)

    question = "Private question that must not be logged"
    with patch.object(main, "answer_question") as answer:
        response = production_client.post("/api/v1/ask", json={"document_id": str(uuid4()), "question": question})

    assert response.status_code == 404
    assert response.json() == {"detail": "Document not found."}
    assert "set-cookie" not in response.headers
    assert response.headers["access-control-allow-origin"] == ORIGIN
    assert response.headers["access-control-allow-credentials"] == "true"
    answer.assert_not_called()
    events = [record.getMessage() for record in caplog.records if record.name == session.__name__]
    assert events == [f"ask_session_rejected reason={failure}"]
    for sensitive in (token, owner, TEST_SECRET, question):
        assert sensitive not in caplog.text


def test_changed_signing_key_rejects_cookie_without_logging_keys(production_client, monkeypatch, caplog):
    original_settings = session.get_settings()
    owner = str(uuid4())
    token = session.create_session_token(owner)
    replacement_key = "replacement-test-secret-00000000000000"
    monkeypatch.setattr(session, "get_settings", lambda: original_settings.model_copy(update={"secret_key": replacement_key}))
    production_client.cookies.set(session.SESSION_COOKIE_NAME, token)
    with patch.object(main, "answer_question") as answer:
        response = production_client.post("/api/v1/ask", json={"document_id": str(uuid4()), "question": "What is covered?"})
    assert response.status_code == 404
    assert response.json() == {"detail": "Document not found."}
    answer.assert_not_called()
    assert "ask_session_rejected reason=invalid_signature" in caplog.text
    for sensitive in (token, owner, TEST_SECRET, replacement_key):
        assert sensitive not in caplog.text


@pytest.mark.parametrize("path", ["ingest/url", "ingest/pdf", "ingest/text", "ask"])
def test_production_post_preflight_preserves_credentials(path, production_client):
    response = production_client.options(f"/api/v1/{path}", headers={
        "Access-Control-Request-Method": "POST",
        "Access-Control-Request-Headers": "content-type",
    })
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == ORIGIN
    assert response.headers["access-control-allow-credentials"] == "true"
