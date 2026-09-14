from unittest.mock import MagicMock, patch

from google.genai.errors import ClientError

from backend.app.retrieval.embeddings import (
    create_document_embeddings,
    create_query_embedding,
)


def test_embedding_dimensions():
    documents = create_document_embeddings(
        ["Privacy Policy"]
    )

    query = create_query_embedding(
        "What does the privacy policy say?"
    )

    assert len(documents) == 1
    assert len(documents[0]) == 768
    assert len(query) == 768


def test_document_embedding_retries_gemini_429_and_succeeds(monkeypatch):
    response = MagicMock()
    response.embeddings = [MagicMock(values=[0.1] * 768)]
    error = ClientError(429, {"error": {"code": 429, "message": "quota"}})

    class Settings:
        gemini_embedding_model = "gemini-embedding-2"
        gemini_embedding_dimension = 768
        external_retry_attempts = 2
        external_retry_base_delay_seconds = 0.1
        external_retry_max_delay_seconds = 0.1

    monkeypatch.setattr(
        "backend.app.core.external_retry.get_settings",
        lambda: Settings(),
    )

    with patch(
        "backend.app.retrieval.embeddings.client.models.embed_content",
        side_effect=[error, response],
    ) as embed, patch("backend.app.core.external_retry.time.sleep") as sleep:
        embeddings = create_document_embeddings(["Privacy Policy"])

    assert len(embeddings) == 1
    assert embed.call_count == 2
    sleep.assert_called_once()


def test_document_embedding_failure_stops_at_retry_max(monkeypatch):
    error = ClientError(429, {"error": {"code": 429, "message": "quota"}})

    class Settings:
        gemini_embedding_model = "gemini-embedding-2"
        gemini_embedding_dimension = 768
        external_retry_attempts = 2
        external_retry_base_delay_seconds = 0.1
        external_retry_max_delay_seconds = 0.1

    monkeypatch.setattr(
        "backend.app.core.external_retry.get_settings",
        lambda: Settings(),
    )

    with patch(
        "backend.app.retrieval.embeddings.client.models.embed_content",
        side_effect=error,
    ) as embed, patch("backend.app.core.external_retry.time.sleep") as sleep:
        try:
            create_document_embeddings(["Privacy Policy"])
        except ClientError:
            pass
        else:
            raise AssertionError("Expected Gemini ClientError.")

    assert embed.call_count == 2
    sleep.assert_called_once()
