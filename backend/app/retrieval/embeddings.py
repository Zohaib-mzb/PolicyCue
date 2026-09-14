from google import genai
from google.genai import types

from backend.app.core.config import get_settings
from backend.app.core.external_retry import retry_external


settings = get_settings()
client = genai.Client(api_key=settings.gemini_api_key)


def _embed(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []

    contents = [
        types.Content(
            parts=[types.Part.from_text(text=text)]
        )
        for text in texts
    ]

    response = retry_external(
        lambda: client.models.embed_content(
            model=settings.gemini_embedding_model,
            contents=contents,
            config={
                "output_dimensionality": settings.gemini_embedding_dimension,
            },
        )
    )

    embeddings = [
        embedding.values
        for embedding in response.embeddings
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
    prepared = [
        f"title: none | text: {text}"
        for text in texts
    ]

    return _embed(prepared)


def create_query_embedding(
    query: str,
) -> list[float]:
    prepared = (
        f"task: question answering | query: {query}"
    )

    return _embed([prepared])[0]
