from unittest.mock import patch

from backend.app.retrieval.vector_store import (
    VECTOR_STORAGE_BATCH_SIZE,
    VectorStorageError,
    delete_document_vectors,
    search_chunks,
    store_chunks,
)


def test_store_chunks():
    embeddings = [
        [0.1] * 768,
        [0.2] * 768,
    ]

    with patch(
        "backend.app.retrieval.vector_store.create_document_embeddings",
        return_value=embeddings,
    ), patch(
        "backend.app.retrieval.vector_store.index"
    ) as mock_index:

        store_chunks(
            document_id="test-document",
            chunks=["First chunk", "Second chunk"],
            owner_id="owner-a",
        )

        mock_index.upsert.assert_called_once()

        vectors = mock_index.upsert.call_args.kwargs["vectors"]

        assert len(vectors) == 2
        assert vectors[0]["id"] == "test-document-0"
        assert vectors[1]["id"] == "test-document-1"
        assert vectors[0]["metadata"]["chunk_index"] == 0
        assert vectors[1]["metadata"]["chunk_index"] == 1
        assert vectors[0]["metadata"]["owner_id"] == "owner-a"
        assert vectors[1]["metadata"]["owner_id"] == "owner-a"


def test_store_chunks_rejects_embedding_count_mismatch():
    embeddings = [
        [0.1] * 768,
    ]

    with patch(
        "backend.app.retrieval.vector_store.create_document_embeddings",
        return_value=embeddings,
    ), patch(
        "backend.app.retrieval.vector_store.index"
    ) as mock_index:

        try:
            store_chunks(
                document_id="test-document",
                chunks=["First chunk", "Second chunk"],
            )
        except VectorStorageError as exc:
            assert exc.stage == "embedding"
            assert "Embedding count mismatch" in str(exc.__cause__)
        else:
            raise AssertionError(
                "Expected RuntimeError for embedding count mismatch."
            )

        mock_index.upsert.assert_not_called()


def test_store_chunks_batches_embeddings_and_upserts_with_global_indexes():
    chunks = [f"chunk-{index}" for index in range(VECTOR_STORAGE_BATCH_SIZE * 2 + 3)]
    metadata = [
        {"source_type": "text", "title": f"title-{index}"}
        for index in range(len(chunks))
    ]

    def embeddings(batch):
        return [[float(index)] * 768 for index in range(len(batch))]

    with patch(
        "backend.app.retrieval.vector_store.create_document_embeddings",
        side_effect=embeddings,
    ) as embed, patch("backend.app.retrieval.vector_store.index") as mock_index:
        store_chunks(
            "large-document",
            chunks,
            owner_id="owner-a",
            chunk_metadata=metadata,
        )

    assert [len(call.args[0]) for call in embed.call_args_list] == [32, 32, 3]
    batches = [call.kwargs["vectors"] for call in mock_index.upsert.call_args_list]
    assert [len(batch) for batch in batches] == [32, 32, 3]
    vectors = [vector for batch in batches for vector in batch]
    assert [vector["metadata"]["chunk_index"] for vector in vectors] == list(range(67))
    assert [vector["id"] for vector in vectors] == [
        f"large-document-{index}" for index in range(67)
    ]
    assert all(vector["metadata"]["document_id"] == "large-document" for vector in vectors)
    assert all(vector["metadata"]["owner_id"] == "owner-a" for vector in vectors)
    assert [vector["metadata"]["title"] for vector in vectors] == [
        f"title-{index}" for index in range(67)
    ]


def test_store_chunks_reports_later_batch_failure():
    chunks = [f"chunk-{index}" for index in range(VECTOR_STORAGE_BATCH_SIZE + 1)]

    with patch(
        "backend.app.retrieval.vector_store.create_document_embeddings",
        side_effect=lambda batch: [[0.1] * 768 for _ in batch],
    ), patch("backend.app.retrieval.vector_store.index") as mock_index:
        mock_index.upsert.side_effect = [None, RuntimeError("provider rejected batch")]
        try:
            store_chunks("partial-document", chunks, owner_id="owner-a")
        except VectorStorageError as exc:
            assert exc.stage == "upsert"
            assert exc.total_chunks == 33
            assert exc.batch_number == 2
            assert exc.batch_size == 1
        else:
            raise AssertionError("Expected the second batch to fail.")


def test_search_chunks_filters_by_document_id():
    mock_results = {
        "matches": [
            {
                "score": 0.95,
                "metadata": {
                    "document_id": "test-document",
                    "text": "First chunk",
                    "chunk_index": 0,
                    "source": "",
                    "filename": "test.pdf",
                },
            }
        ]
    }

    with patch(
        "backend.app.retrieval.vector_store.create_query_embedding",
        return_value=[0.1] * 768,
    ), patch(
        "backend.app.retrieval.vector_store.index"
    ) as mock_index:

        mock_index.query.return_value = mock_results

        results = search_chunks(
            query="What is this about?",
            document_id="test-document",
            owner_id="owner-a",
        )

        mock_index.query.assert_called_once()

        call_kwargs = mock_index.query.call_args.kwargs

        assert call_kwargs["vector"] == [0.1] * 768
        assert call_kwargs["top_k"] == 5
        assert call_kwargs["include_metadata"] is True
        assert call_kwargs["filter"] == {
            "$and": [
                {
                    "document_id": {
                        "$eq": "test-document",
                    }
                },
                {
                    "owner_id": {
                        "$eq": "owner-a",
                    }
                },
            ]
        }

        assert len(results) == 1
        assert results[0]["document_id"] == "test-document"
        assert "owner_id" not in results[0]
        assert results[0]["chunk_index"] == 0
        assert results[0]["text"] == "First chunk"
        assert results[0]["filename"] == "test.pdf"

def test_url_chunk_offsets_and_metadata_do_not_overwrite_other_pages():
    metadata = {"source_url": "https://example.com/privacy", "source_urls": ["https://example.com/privacy"], "policy_categories": ["privacy_policy"], "content_hash": "hash", "document_id": "must-not-override"}
    with patch("backend.app.retrieval.vector_store.create_document_embeddings", return_value=[[0.1] * 768]), patch("backend.app.retrieval.vector_store.index") as mock_index:
        store_chunks("website", ["policy"], chunk_index_offset=32, chunk_metadata=[metadata])
    vector = mock_index.upsert.call_args.kwargs["vectors"][0]
    assert vector["id"] == "website-32"
    assert vector["metadata"]["chunk_index"] == 32
    assert vector["metadata"]["document_id"] == "website"
    assert vector["metadata"]["source"] == metadata["source_url"]
    assert vector["metadata"]["policy_categories"] == ["privacy_policy"]
    assert vector["metadata"]["filename"] == ""


def test_delete_document_vectors_targets_only_document_id():
    with patch("backend.app.retrieval.vector_store.index") as mock_index:
        delete_document_vectors("failed-document", "owner-a")

    mock_index.delete.assert_called_once_with(
        filter={
            "$and": [
                {
                    "document_id": {
                        "$eq": "failed-document",
                    }
                },
                {
                    "owner_id": {
                        "$eq": "owner-a",
                    }
                },
            ]
        },
    )


def test_pinecone_transient_upsert_failure_is_retried(monkeypatch):
    class TransientError(Exception):
        status_code = 503

    class Settings:
        external_retry_attempts = 2
        external_retry_base_delay_seconds = 0.1
        external_retry_max_delay_seconds = 0.1

    with patch(
        "backend.app.retrieval.vector_store.create_document_embeddings",
        return_value=[[0.1] * 768],
    ), patch(
        "backend.app.core.external_retry.get_settings",
        return_value=Settings(),
    ), patch(
        "backend.app.core.external_retry.time.sleep"
    ), patch(
        "backend.app.retrieval.vector_store.index"
    ) as mock_index:
        mock_index.upsert.side_effect = [TransientError(), None]

        store_chunks("test-document", ["First chunk"], owner_id="owner-a")

    assert mock_index.upsert.call_count == 2


def test_pinecone_transient_delete_failure_is_retried(monkeypatch):
    class TransientError(Exception):
        status_code = 503

    class Settings:
        external_retry_attempts = 2
        external_retry_base_delay_seconds = 0.1
        external_retry_max_delay_seconds = 0.1

    with patch(
        "backend.app.core.external_retry.get_settings",
        return_value=Settings(),
    ), patch(
        "backend.app.core.external_retry.time.sleep"
    ), patch(
        "backend.app.retrieval.vector_store.index"
    ) as mock_index:
        mock_index.delete.side_effect = [TransientError(), None]

        delete_document_vectors("test-document", "owner-a")

    assert mock_index.delete.call_count == 2
