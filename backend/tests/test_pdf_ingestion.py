from io import BytesIO
from unittest.mock import patch

import pytest
from fastapi import HTTPException, UploadFile
from starlette.requests import Request
from starlette.responses import Response

from backend.app.main import ingest_pdf
from backend.app.retrieval.vector_store import VectorStorageError
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


@pytest.mark.anyio
async def test_later_pdf_batch_failure_logs_context_and_attempts_owned_cleanup(caplog):
    owner_id = "99999999-9999-4999-8999-999999999999"
    http_request, response = _owned_request_response(owner_id)
    file = UploadFile(
        filename="large-policy.pdf",
        file=BytesIO(b"%PDF"),
        headers={"content-type": "application/pdf"},
    )
    provider_error = RuntimeError("provider rejected batch; api_key=must-not-appear")
    storage_error = VectorStorageError(
        stage="upsert",
        total_chunks=65,
        batch_number=3,
        batch_size=1,
    )
    storage_error.__cause__ = provider_error

    with patch("backend.app.main.uuid4", return_value="failed-large-pdf"), patch(
        "backend.app.main.extract_pdf_text", return_value="Policy text"
    ), patch(
        "backend.app.main.chunk_text", return_value=[f"chunk-{index}" for index in range(65)]
    ), patch(
        "backend.app.main.store_chunks", side_effect=storage_error
    ), patch("backend.app.main.delete_document_vectors") as cleanup:
        with pytest.raises(HTTPException) as caught:
            await ingest_pdf(http_request, response, file)

    assert caught.value.status_code == 503
    assert caught.value.detail == "Document storage failed. Please retry ingestion."
    assert "provider rejected batch" not in caught.value.detail
    cleanup.assert_called_once_with("failed-large-pdf", owner_id)
    assert "stage=upsert" in caplog.text
    assert "total_chunks=65" in caplog.text
    assert "batch_number=3" in caplog.text
    assert "batch_size=1" in caplog.text
    assert "must-not-appear" not in caplog.text
    assert "[REDACTED]" in caplog.text

from backend.app.main import (
    MAX_FILENAME_METADATA_LENGTH,
    PDF_READ_CHUNK_SIZE,
    _read_pdf_upload,
)
from backend.app.ingestion.pdf_processor import MAX_PDF_SIZE


class ChunkedAsyncFile:
    def __init__(self, chunks):
        self.chunks = list(chunks)
        self.read_sizes = []

    async def read(self, size=-1):
        self.read_sizes.append(size)
        if not self.chunks:
            return b""
        return self.chunks.pop(0)


@pytest.mark.anyio
async def test_pdf_upload_is_read_incrementally_without_unbounded_read():
    file = ChunkedAsyncFile([b"%PDF-", b"body", b""])

    content = await _read_pdf_upload(file)

    assert content == b"%PDF-body"
    assert file.read_sizes == [PDF_READ_CHUNK_SIZE, PDF_READ_CHUNK_SIZE, PDF_READ_CHUNK_SIZE]


@pytest.mark.anyio
async def test_oversized_upload_stops_once_byte_limit_is_exceeded():
    file = ChunkedAsyncFile([b"a" * MAX_PDF_SIZE, b"b"])

    with pytest.raises(HTTPException) as caught:
        await _read_pdf_upload(file)

    assert caught.value.status_code == 400
    assert caught.value.detail == "PDF is too large."
    assert file.read_sizes == [PDF_READ_CHUNK_SIZE, PDF_READ_CHUNK_SIZE]


@pytest.mark.anyio
async def test_oversized_pdf_upload_performs_no_parsing_or_storage():
    http_request, response = _owned_request_response(
        "66666666-6666-4666-8666-666666666666"
    )
    file = ChunkedAsyncFile([b"a" * MAX_PDF_SIZE, b"b"])
    file.filename = "policy.pdf"
    file.content_type = "application/pdf"

    with patch("backend.app.main.extract_pdf_text") as extract, patch("backend.app.main.store_chunks") as store:
        with pytest.raises(HTTPException) as caught:
            await ingest_pdf(http_request, response, file)

    assert caught.value.status_code == 400
    assert caught.value.detail == "PDF is too large."
    extract.assert_not_called()
    store.assert_not_called()


@pytest.mark.anyio
async def test_content_type_alone_is_insufficient_for_pdf_ingestion():
    http_request, response = _owned_request_response(
        "77777777-7777-4777-8777-777777777777"
    )
    file = UploadFile(
        filename="fake.pdf",
        file=BytesIO(b"not a pdf"),
        headers={"content-type": "application/pdf"},
    )

    with patch("backend.app.main.store_chunks") as store:
        with pytest.raises(HTTPException) as caught:
            await ingest_pdf(http_request, response, file)

    assert caught.value.status_code == 400
    assert caught.value.detail == "Uploaded file is not a valid PDF."
    store.assert_not_called()


@pytest.mark.anyio
async def test_long_filename_metadata_is_bounded_and_not_used_as_path():
    http_request, response = _owned_request_response(
        "88888888-8888-4888-8888-888888888888"
    )
    long_filename = "../" + "a" * 400 + ".pdf"
    file = UploadFile(
        filename=long_filename,
        file=BytesIO(b"%PDF"),
        headers={"content-type": "application/pdf"},
    )

    with patch("backend.app.main.extract_pdf_text", return_value="Policy text"), patch("backend.app.main.chunk_text", return_value=["chunk"]), patch("backend.app.main.store_chunks") as store:
        result = await ingest_pdf(http_request, response, file)

    assert result["filename"] == long_filename[:MAX_FILENAME_METADATA_LENGTH]
    assert store.call_args.kwargs["filename"] == long_filename[:MAX_FILENAME_METADATA_LENGTH]
    assert len(store.call_args.kwargs["filename"]) == MAX_FILENAME_METADATA_LENGTH
