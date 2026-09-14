from io import BytesIO
from unittest.mock import patch

import pytest
from fastapi import HTTPException, UploadFile
from starlette.requests import Request
from starlette.responses import Response

from backend.app.main import ingest_pdf
from backend.app.core.session import SESSION_COOKIE_NAME, create_session_token
from backend.app.core.rate_limit import reset_rate_limits


def setup_function():
    reset_rate_limits()


def _request_response(cookie: str | None = None):
    headers = []
    if cookie:
        headers.append((b"cookie", cookie.encode()))
    scope = {"type": "http", "method": "POST", "path": "/", "headers": headers}
    return Request(scope), Response()


def _owned_request_response(owner_id: str):
    return _request_response(
        f"{SESSION_COOKIE_NAME}={create_session_token(owner_id)}"
    )


@pytest.mark.anyio
async def test_successful_pdf_ingestion_does_not_cleanup_vectors():
    http_request, response = _request_response()
    file = UploadFile(
        filename="policy.pdf",
        file=BytesIO(b"%PDF"),
        headers={"content-type": "application/pdf"},
    )

    with patch("backend.app.main.extract_pdf_text", return_value="Policy text"), patch("backend.app.main.chunk_text", return_value=["chunk"]), patch("backend.app.main.store_chunks") as store, patch("backend.app.main.delete_document_vectors") as cleanup:
        result = await ingest_pdf(http_request, response, file)

    assert result["status"] == "success"
    assert result["filename"] == "policy.pdf"
    store.assert_called_once()
    assert store.call_args.kwargs["owner_id"]
    cleanup.assert_not_called()


@pytest.mark.anyio
async def test_pdf_storage_failure_triggers_cleanup_for_failed_document():
    http_request, response = _owned_request_response(
        "44444444-4444-4444-8444-444444444444"
    )
    file = UploadFile(
        filename="policy.pdf",
        file=BytesIO(b"%PDF"),
        headers={"content-type": "application/pdf"},
    )

    with patch("backend.app.main.uuid4", return_value="failed-pdf-document"), patch("backend.app.main.extract_pdf_text", return_value="Policy text"), patch("backend.app.main.chunk_text", return_value=["chunk"]), patch("backend.app.main.store_chunks", side_effect=RuntimeError("upsert failed")), patch("backend.app.main.delete_document_vectors") as cleanup:
        with pytest.raises(HTTPException) as caught:
            await ingest_pdf(http_request, response, file)

    assert caught.value.status_code == 503
    assert caught.value.detail == "Document storage failed. Please retry ingestion."
    cleanup.assert_called_once_with(
        "failed-pdf-document",
        "44444444-4444-4444-8444-444444444444",
    )


@pytest.mark.anyio
async def test_pdf_cleanup_failure_keeps_original_error(caplog):
    http_request, response = _owned_request_response(
        "55555555-5555-4555-8555-555555555555"
    )
    file = UploadFile(
        filename="policy.pdf",
        file=BytesIO(b"%PDF"),
        headers={"content-type": "application/pdf"},
    )

    with patch("backend.app.main.uuid4", return_value="failed-pdf-cleanup"), patch("backend.app.main.extract_pdf_text", return_value="Policy text"), patch("backend.app.main.chunk_text", return_value=["chunk"]), patch("backend.app.main.store_chunks", side_effect=RuntimeError("original storage failure")), patch("backend.app.main.delete_document_vectors", side_effect=RuntimeError("cleanup failed")):
        with pytest.raises(HTTPException) as caught:
            await ingest_pdf(http_request, response, file)

    assert caught.value.status_code == 503
    assert caught.value.detail == "Document storage failed. Please retry ingestion."
    assert "failed-pdf-cleanup" in caplog.text
