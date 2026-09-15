import asyncio
from io import BytesIO
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from fastapi import HTTPException, UploadFile
from starlette.requests import Request
from starlette.responses import Response

from backend.app.core.rate_limit import reset_rate_limits
from backend.app.core.rate_limit import check_rate_limit
from backend.app.core.session import SESSION_COOKIE_NAME, create_session_token
from backend.app.ingestion.policy_categories import PolicyCategory
from backend.app.main import (
    QuestionRequest,
    TextRequest,
    URLRequest,
    _ingest_pdf_document,
    _ingest_text_document,
    _delete_owned_document,
    _prepare_url_document,
    _store_url_batch,
    app,
    ask_question,
    delete_document,
    ingest_pdf,
    ingest_text,
    ingest_url,
)


OWNER_ID = "11111111-1111-4111-8111-111111111111"


def setup_function():
    reset_rate_limits()


def _request(cookie: str | None = None):
    headers = []
    if cookie:
        headers.append((b"cookie", cookie.encode()))
    return Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/",
            "headers": headers,
            "client": ("127.0.0.1", 12345),
        }
    )


def _owned_request():
    token = create_session_token(OWNER_ID)
    return _request(f"{SESSION_COOKIE_NAME}={token}")


@pytest.mark.anyio
async def test_ask_dispatches_blocking_workflow_off_event_loop():
    async def fake_to_thread(func, *args, **kwargs):
        assert func.__name__ == "answer_question"
        assert args == ("What?", 5, "doc-a", OWNER_ID)
        assert kwargs == {}
        return {"answer": "ok", "sources": [{"text": "context"}]}

    with patch("backend.app.main.asyncio.to_thread", side_effect=fake_to_thread) as to_thread:
        result = await ask_question(
            QuestionRequest(question="What?", document_id="doc-a"),
            _owned_request(),
        )

    assert result["answer"] == "ok"
    to_thread.assert_awaited_once()


@pytest.mark.anyio
async def test_pdf_ingestion_dispatches_processing_storage_off_event_loop():
    file = UploadFile(
        filename="policy.pdf",
        file=BytesIO(b"%PDF"),
        headers={"content-type": "application/pdf"},
    )

    async def fake_to_thread(func, *args, **kwargs):
        assert func is _ingest_pdf_document
        assert args == (b"%PDF", "policy.pdf", OWNER_ID)
        assert kwargs == {}
        return {"status": "success", "document_id": "doc-a", "filename": "policy.pdf", "chunks": 1}

    with patch("backend.app.main.asyncio.to_thread", side_effect=fake_to_thread) as to_thread:
        result = await ingest_pdf(_owned_request(), Response(), file)

    assert result["status"] == "success"
    to_thread.assert_awaited_once()


@pytest.mark.anyio
async def test_text_ingestion_dispatches_processing_storage_off_event_loop():
    async def fake_to_thread(func, *args, **kwargs):
        assert func is _ingest_text_document
        assert args == ("Policy text", OWNER_ID)
        assert kwargs == {}
        return {"status": "success", "document_id": "doc-a", "chunks": 1}

    with patch("backend.app.main.asyncio.to_thread", side_effect=fake_to_thread) as to_thread:
        result = await ingest_text(
            TextRequest(text="Policy text"),
            _owned_request(),
            Response(),
        )

    assert result["status"] == "success"
    to_thread.assert_awaited_once()


@pytest.mark.anyio
async def test_url_fetch_remains_async_and_storage_is_offloaded():
    document = {
        "url": "https://example.com/",
        "pages": [
            {
                "url": "https://example.com/privacy",
                "source_urls": ["https://example.com/privacy"],
                "text": "Privacy policy",
                "categories": [PolicyCategory.PRIVACY],
                "content_hash": "hash",
            }
        ],
        "warnings": [],
        "skipped": {},
        "candidates": 1,
    }
    dispatched = []

    async def fake_to_thread(func, *args, **kwargs):
        dispatched.append(func)
        if func is _prepare_url_document:
            return ["chunk"], [{"source_url": "https://example.com/privacy"}], {"privacy_policy": ["https://example.com/privacy"]}, []
        assert func is _store_url_batch
        return None

    with patch("backend.app.main.discover_website_policies", new=AsyncMock(return_value=document)) as discover, patch("backend.app.main.asyncio.to_thread", side_effect=fake_to_thread):
        result = await ingest_url(
            URLRequest(url="https://example.com/"),
            _owned_request(),
            Response(),
        )

    discover.assert_awaited_once_with("https://example.com/")
    assert dispatched == [_prepare_url_document, _store_url_batch]
    assert result["status"] == "success"


@pytest.mark.anyio
async def test_ownership_check_occurs_before_expensive_thread_work():
    with patch("backend.app.main.asyncio.to_thread") as to_thread:
        with pytest.raises(HTTPException) as caught:
            await ask_question(
                QuestionRequest(question="What?", document_id="doc-a"),
                _request("policylens_session=invalid"),
            )

    assert caught.value.status_code == 404
    to_thread.assert_not_called()


@pytest.mark.anyio
async def test_rate_limit_occurs_before_expensive_thread_work():
    request = _owned_request()
    check_rate_limit(request, scope="ask", limit=1)
    with patch("backend.app.main.settings.ask_rate_limit", 1), patch("backend.app.main.asyncio.to_thread") as to_thread:
        with pytest.raises(HTTPException) as caught:
            await ask_question(
                QuestionRequest(question="What?", document_id="doc-a"),
                request,
            )

    assert caught.value.status_code == 429
    to_thread.assert_not_called()


@pytest.mark.anyio
async def test_document_deletion_dispatches_delete_off_event_loop():
    async def fake_to_thread(func, *args, **kwargs):
        assert func is _delete_owned_document
        assert args == ("33333333-3333-4333-8333-333333333333", OWNER_ID)
        assert kwargs == {}
        return None

    with patch("backend.app.main.asyncio.to_thread", side_effect=fake_to_thread) as to_thread:
        result = await delete_document(
            "33333333-3333-4333-8333-333333333333",
            _owned_request(),
        )

    assert result == {
        "status": "deleted",
        "document_id": "33333333-3333-4333-8333-333333333333",
    }
    to_thread.assert_awaited_once()


@pytest.mark.anyio
async def test_health_responds_while_mocked_blocking_work_is_pending():
    started = False

    async def fake_to_thread(func, *args, **kwargs):
        nonlocal started
        if func.__name__ != "answer_question":
            return func(*args, **kwargs)
        started = True
        await wait_event.wait()
        return {"answer": "ok", "sources": [{"text": "context"}]}

    wait_event = asyncio.Event()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        token = create_session_token(OWNER_ID)
        client.cookies.set(SESSION_COOKIE_NAME, token)
        with patch("backend.app.main.asyncio.to_thread", side_effect=fake_to_thread):
            ask_task = asyncio.create_task(
                client.post(
                    "/api/v1/ask",
                    json={"question": "What?", "document_id": "doc-a"},
                )
            )
            while not started:
                await asyncio.sleep(0)
            health = await client.get("/health")
            wait_event.set()
            ask_response = await ask_task

    assert health.status_code == 200
    assert health.json() == {"status": "ok"}
    assert ask_response.status_code == 200
