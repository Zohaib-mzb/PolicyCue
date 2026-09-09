from google import genai

from backend.app.core.config import get_settings


settings = get_settings()
client = genai.Client(api_key=settings.gemini_api_key)


def create_embeddings(texts: list[str]) -> list[list[float]]:
    response = client.models.embed_content(
        model=settings.gemini_embedding_model,
        contents=texts,
        config={
            "output_dimensionality": settings.gemini_embedding_dimension,
        },
    )

    return [embedding.values for embedding in response.embeddings]