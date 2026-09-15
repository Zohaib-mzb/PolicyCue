from pinecone import Pinecone

from backend.app.core.config import get_settings
from backend.app.core.external_retry import retry_external


settings = get_settings()
pinecone = Pinecone(
    api_key=settings.pinecone_api_key,
)


def _embedding_values(embedding) -> list[float]:
    if isinstance(embedding, dict):
        return embedding["values"]
    return embedding.values


def _embed(texts: list[str], input_type: str) -> list[list[float]]:
    if not texts:
        return []

    response = retry_external(
        lambda: pinecone.inference.embed(
            model=settings.pinecone_embedding_model,
            inputs=texts,
            parameters={
                "input_type": input_type,
                "truncate": "END",
                "dimension": settings.pinecone_embedding_dimension,
            },
        )
    )

    embeddings = [
        _embedding_values(embedding)
        for embedding in response
    ]

    if len(embeddings) != len(texts):
        raise RuntimeError(
            f"Expected {len(texts)} embeddings, "
            f"received {len(embeddings)}."
        )

    return embeddings


def create_document_embeddings(
    texts: list[str],
) -> list[list[float]]:
    return _embed(texts, input_type="passage")


def create_query_embedding(
    query: str,
) -> list[float]:
    return _embed([query], input_type="query")[0]
