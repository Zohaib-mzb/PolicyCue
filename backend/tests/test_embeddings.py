from unittest.mock import patch

from backend.app.retrieval.embeddings import (
    create_document_embeddings,
    create_query_embedding,
)


class TransientError(Exception):
    status_code = 429


class Settings:
    pinecone_embedding_model = "llama-text-embed-v2"
    pinecone_embedding_dimension = 768
    external_retry_attempts = 2
    external_retry_base_delay_seconds = 0.1
    external_retry_max_delay_seconds = 0.1


def test_document_embedding_uses_passage_mode_and_dimension(monkeypatch):
    response = [
        {"values": [0.1] * 768},
        {"values": [0.2] * 768},
    ]

    monkeypatch.setattr(
        "backend.app.core.external_retry.get_settings",
        lambda: Settings(),
    )

    with patch(
        "backend.app.retrieval.embeddings.pinecone.inference.embed",
        return_value=response,
    ) as embed:
        embeddings = create_document_embeddings(["Privacy Policy", "Terms of Service"])

    assert len(embeddings) == 2
    assert len(embeddings[0]) == 768
    embed.assert_called_once_with(
        model="llama-text-embed-v2",
        inputs=["Privacy Policy", "Terms of Service"],
        parameters={
            "input_type": "passage",
            "truncate": "END",
            "dimension": 768,
        },
    )


def test_query_embedding_uses_query_mode_and_dimension(monkeypatch):
    response = [{"values": [0.3] * 768}]

    monkeypatch.setattr(
        "backend.app.core.external_retry.get_settings",
        lambda: Settings(),
    )

    with patch(
        "backend.app.retrieval.embeddings.pinecone.inference.embed",
        return_value=response,
    ) as embed:
        embedding = create_query_embedding("What does the privacy policy say?")

    assert len(embedding) == 768
    embed.assert_called_once_with(
        model="llama-text-embed-v2",
        inputs=["What does the privacy policy say?"],
        parameters={
            "input_type": "query",
            "truncate": "END",
            "dimension": 768,
        },
    )


def test_empty_document_embedding_input_does_not_call_pinecone():
    with patch(
        "backend.app.retrieval.embeddings.pinecone.inference.embed",
    ) as embed:
        assert create_document_embeddings([]) == []

    embed.assert_not_called()


def test_document_embedding_rejects_return_count_mismatch():
    with patch(
        "backend.app.retrieval.embeddings.pinecone.inference.embed",
        return_value=[{"values": [0.1] * 768}],
    ):
        try:
            create_document_embeddings(["Privacy Policy", "Terms of Service"])
        except RuntimeError as exc:
            assert "Expected 2 embeddings, received 1" in str(exc)
        else:
            raise AssertionError("Expected RuntimeError for embedding count mismatch.")


def test_document_embedding_retries_pinecone_429_and_succeeds(monkeypatch):
    monkeypatch.setattr(
        "backend.app.core.external_retry.get_settings",
        lambda: Settings(),
    )

    with patch(
        "backend.app.retrieval.embeddings.pinecone.inference.embed",
        side_effect=[TransientError(), [{"values": [0.1] * 768}]],
    ) as embed, patch("backend.app.core.external_retry.time.sleep") as sleep:
        embeddings = create_document_embeddings(["Privacy Policy"])

    assert len(embeddings) == 1
    assert embed.call_count == 2
    sleep.assert_called_once()


def test_document_embedding_failure_stops_at_retry_max(monkeypatch):
    monkeypatch.setattr(
        "backend.app.core.external_retry.get_settings",
        lambda: Settings(),
    )

    with patch(
        "backend.app.retrieval.embeddings.pinecone.inference.embed",
        side_effect=TransientError(),
    ) as embed, patch("backend.app.core.external_retry.time.sleep") as sleep:
        try:
            create_document_embeddings(["Privacy Policy"])
        except TransientError:
            pass
        else:
            raise AssertionError("Expected Pinecone inference error.")

    assert embed.call_count == 2
    sleep.assert_called_once()
