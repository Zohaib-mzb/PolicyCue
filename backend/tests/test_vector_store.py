from unittest.mock import patch

from backend.app.retrieval.vector_store import store_chunks


def test_store_chunks():
    embeddings = [[0.1] * 768, [0.2] * 768]

    with patch(
        "backend.app.retrieval.vector_store.create_embeddings",
        return_value=embeddings,
    ), patch(
        "backend.app.retrieval.vector_store.index"
    ) as mock_index:

        store_chunks(
            "test-document",
            ["First chunk", "Second chunk"],
        )

        mock_index.upsert.assert_called_once()