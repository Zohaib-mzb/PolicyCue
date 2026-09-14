from unittest.mock import AsyncMock, patch

import pytest
from fastapi import HTTPException

from backend.app.main import URLRequest, ingest_url
from backend.app.ingestion.policy_categories import PolicyCategory
from backend.app.ingestion.policy_discovery import content_fingerprint


@pytest.mark.anyio
async def test_website_uses_one_document_and_unique_chunk_indexes():
    pages = [
        {"url": f"https://example.com/{name}", "source_urls": [f"https://example.com/{name}"],
         "text": f"{name} policy content", "categories": [category],
         "content_hash": content_fingerprint(name)}
        for name, category in [("privacy", PolicyCategory.PRIVACY), ("terms", PolicyCategory.TERMS)]
    ]
    document = {"url": "https://example.com/", "pages": pages, "warnings": [], "skipped": {}, "candidates": 2}
    with patch("backend.app.main.discover_website_policies", new=AsyncMock(return_value=document)), patch("backend.app.main.chunk_text", side_effect=[["p1", "p2"], ["t1", "t2"]]), patch("backend.app.main.URL_CHUNK_BATCH_SIZE", 2), patch("backend.app.main.store_chunks") as store:
        result = await ingest_url(URLRequest(url="https://example.com"))
    assert result["chunks"] == 4
    assert store.call_count == 2
    assert all(call.args[0] == result["document_id"] for call in store.call_args_list)
    assert [c.kwargs["chunk_index_offset"] for c in store.call_args_list] == [0, 2]
    assert store.call_args_list[0].kwargs["chunk_metadata"][0]["source_url"] == pages[0]["url"]
    assert store.call_args_list[1].kwargs["chunk_metadata"][0]["policy_categories"] == ["terms_of_service"]
    assert len(result["accepted_policies"]) == 2


@pytest.mark.anyio
async def test_successful_url_ingestion_does_not_cleanup_vectors():
    page = {"url": "https://example.com/privacy", "source_urls": ["https://example.com/privacy"], "text": "Policy content", "categories": [PolicyCategory.PRIVACY], "content_hash": "hash"}
    document = {"url": "https://example.com/", "pages": [page], "warnings": [], "skipped": {}, "candidates": 1}
    with patch("backend.app.main.discover_website_policies", new=AsyncMock(return_value=document)), patch("backend.app.main.chunk_text", return_value=["p1"]), patch("backend.app.main.store_chunks") as store, patch("backend.app.main.delete_document_vectors") as cleanup:
        result = await ingest_url(URLRequest(url="https://example.com"))
    assert result["status"] == "success"
    store.assert_called_once()
    cleanup.assert_not_called()


@pytest.mark.anyio
async def test_no_policies_does_not_claim_success_or_store_vectors():
    document = {"url": "https://example.com/", "pages": [], "warnings": ["Some pages unavailable"], "skipped": {}, "candidates": 5}
    with patch("backend.app.main.discover_website_policies", new=AsyncMock(return_value=document)), patch("backend.app.main.store_chunks") as store:
        result = await ingest_url(URLRequest(url="https://example.com"))
    assert result["status"] == "no_policies_found"
    assert result["warnings"] == document["warnings"]
    assert result["chunks"] == 0
    store.assert_not_called()


@pytest.mark.anyio
@pytest.mark.parametrize("code,expected_calls", [(429, 2), (400, 1)])
async def test_url_storage_retry_is_bounded_and_only_for_rate_limits(code, expected_calls):
    from google.genai.errors import ClientError
    page = {"url": "https://example.com/privacy", "source_urls": ["https://example.com/privacy"], "text": "Policy content", "categories": [PolicyCategory.PRIVACY], "content_hash": "hash"}
    document = {"url": "https://example.com/", "pages": [page], "warnings": [], "skipped": {}, "candidates": 1}
    error = ClientError(code, {"error": {"code": code, "message": "test error"}})
    with patch("backend.app.main.discover_website_policies", new=AsyncMock(return_value=document)), patch("backend.app.main.store_chunks", side_effect=error) as store, patch("backend.app.main.asyncio.sleep", new_callable=AsyncMock) as sleep, patch("backend.app.main.delete_document_vectors") as cleanup:
        with pytest.raises(HTTPException) as caught:
            await ingest_url(URLRequest(url="https://example.com/"))
    assert caught.value.status_code == 503
    assert caught.value.detail == "Policy storage failed. Please retry ingestion."
    assert store.call_count == expected_calls
    assert sleep.await_count == expected_calls - 1
    cleanup.assert_called_once()


@pytest.mark.anyio
async def test_url_quota_retry_reuses_same_document_and_chunk_offset():
    from google.genai.errors import ClientError
    page = {"url": "https://example.com/privacy", "source_urls": ["https://example.com/privacy"], "text": "Policy content", "categories": [PolicyCategory.PRIVACY], "content_hash": "hash"}
    document = {"url": "https://example.com/", "pages": [page], "warnings": [], "skipped": {}, "candidates": 1}
    error = ClientError(429, {"error": {"code": 429, "message": "quota"}})
    with patch("backend.app.main.discover_website_policies", new=AsyncMock(return_value=document)), patch("backend.app.main.store_chunks", side_effect=[error, None]) as store, patch("backend.app.main.asyncio.sleep", new_callable=AsyncMock) as sleep:
        result = await ingest_url(URLRequest(url="https://example.com/"))
    assert result["status"] == "success"
    assert store.call_args_list[0] == store.call_args_list[1]
    sleep.assert_awaited_once_with(60)


@pytest.mark.anyio
async def test_url_partial_storage_failure_triggers_cleanup_for_failed_document():
    pages = [
        {"url": "https://example.com/privacy", "source_urls": ["https://example.com/privacy"], "text": "Privacy policy", "categories": [PolicyCategory.PRIVACY], "content_hash": "one"},
        {"url": "https://example.com/terms", "source_urls": ["https://example.com/terms"], "text": "Terms policy", "categories": [PolicyCategory.TERMS], "content_hash": "two"},
    ]
    document = {"url": "https://example.com/", "pages": pages, "warnings": [], "skipped": {}, "candidates": 2}
    with patch("backend.app.main.uuid4", return_value="failed-url-document"), patch("backend.app.main.discover_website_policies", new=AsyncMock(return_value=document)), patch("backend.app.main.chunk_text", side_effect=[["p1"], ["t1"]]), patch("backend.app.main.URL_CHUNK_BATCH_SIZE", 1), patch("backend.app.main.store_chunks", side_effect=[None, RuntimeError("upsert failed")]) as store, patch("backend.app.main.delete_document_vectors") as cleanup:
        with pytest.raises(HTTPException) as caught:
            await ingest_url(URLRequest(url="https://example.com/"))

    assert caught.value.status_code == 503
    assert caught.value.detail == "Policy storage failed. Please retry ingestion."
    assert store.call_count == 2
    cleanup.assert_called_once_with("failed-url-document")


@pytest.mark.anyio
async def test_url_cleanup_failure_keeps_original_error(caplog):
    page = {"url": "https://example.com/privacy", "source_urls": ["https://example.com/privacy"], "text": "Policy content", "categories": [PolicyCategory.PRIVACY], "content_hash": "hash"}
    document = {"url": "https://example.com/", "pages": [page], "warnings": [], "skipped": {}, "candidates": 1}
    with patch("backend.app.main.uuid4", return_value="failed-cleanup-document"), patch("backend.app.main.discover_website_policies", new=AsyncMock(return_value=document)), patch("backend.app.main.store_chunks", side_effect=RuntimeError("original storage failure")), patch("backend.app.main.delete_document_vectors", side_effect=RuntimeError("cleanup failed")):
        with pytest.raises(HTTPException) as caught:
            await ingest_url(URLRequest(url="https://example.com/"))

    assert caught.value.status_code == 503
    assert caught.value.detail == "Policy storage failed. Please retry ingestion."
    assert "failed-cleanup-document" in caplog.text


@pytest.mark.anyio
async def test_url_failure_before_any_upsert_cleanup_is_safe():
    page = {"url": "https://example.com/privacy", "source_urls": ["https://example.com/privacy"], "text": "Policy content", "categories": [PolicyCategory.PRIVACY], "content_hash": "hash"}
    document = {"url": "https://example.com/", "pages": [page], "warnings": [], "skipped": {}, "candidates": 1}
    with patch("backend.app.main.uuid4", return_value="embedding-failed-document"), patch("backend.app.main.discover_website_policies", new=AsyncMock(return_value=document)), patch("backend.app.main.store_chunks", side_effect=RuntimeError("embedding failed")) as store, patch("backend.app.main.delete_document_vectors") as cleanup:
        with pytest.raises(HTTPException):
            await ingest_url(URLRequest(url="https://example.com/"))

    store.assert_called_once()
    cleanup.assert_called_once_with("embedding-failed-document")
