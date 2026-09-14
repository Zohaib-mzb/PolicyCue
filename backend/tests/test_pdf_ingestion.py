from io import BytesIO
from unittest.mock import patch

import pytest
from fastapi import HTTPException, UploadFile

from backend.app.main import ingest_pdf


@pytest.mark.anyio
async def test_successful_pdf_ingestion_does_not_cleanup_vectors():
    file = UploadFile(
        filename="policy.pdf",
        file=BytesIO(b"%PDF"),
        headers={"content-type": "application/pdf"},
    )

    with patch("backend.app.main.extract_pdf_text", return_value="Policy text"), patch("backend.app.main.chunk_text", return_value=["chunk"]), patch("backend.app.main.store_chunks") as store, patch("backend.app.main.delete_document_vectors") as cleanup:
        result = await ingest_pdf(file)

    assert result["status"] == "success"
    assert result["filename"] == "policy.pdf"
    store.assert_called_once()
    cleanup.assert_not_called()


@pytest.mark.anyio
async def test_pdf_storage_failure_triggers_cleanup_for_failed_document():
    file = UploadFile(
        filename="policy.pdf",
        file=BytesIO(b"%PDF"),
        headers={"content-type": "application/pdf"},
    )

    with patch("backend.app.main.uuid4", return_value="failed-pdf-document"), patch("backend.app.main.extract_pdf_text", return_value="Policy text"), patch("backend.app.main.chunk_text", return_value=["chunk"]), patch("backend.app.main.store_chunks", side_effect=RuntimeError("upsert failed")), patch("backend.app.main.delete_document_vectors") as cleanup:
        with pytest.raises(HTTPException) as caught:
            await ingest_pdf(file)

    assert caught.value.status_code == 503
    assert caught.value.detail == "Document storage failed. Please retry ingestion."
    cleanup.assert_called_once_with("failed-pdf-document")


@pytest.mark.anyio
async def test_pdf_cleanup_failure_keeps_original_error(caplog):
    file = UploadFile(
        filename="policy.pdf",
        file=BytesIO(b"%PDF"),
        headers={"content-type": "application/pdf"},
    )

    with patch("backend.app.main.uuid4", return_value="failed-pdf-cleanup"), patch("backend.app.main.extract_pdf_text", return_value="Policy text"), patch("backend.app.main.chunk_text", return_value=["chunk"]), patch("backend.app.main.store_chunks", side_effect=RuntimeError("original storage failure")), patch("backend.app.main.delete_document_vectors", side_effect=RuntimeError("cleanup failed")):
        with pytest.raises(HTTPException) as caught:
            await ingest_pdf(file)

    assert caught.value.status_code == 503
    assert caught.value.detail == "Document storage failed. Please retry ingestion."
    assert "failed-pdf-cleanup" in caplog.text
