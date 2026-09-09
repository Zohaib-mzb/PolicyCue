from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_env: str = "development"

    gemini_api_key: str
    pinecone_api_key: str
    pinecone_index_name: str
    gemini_embedding_model: str
    gemini_embedding_dimension: int = 768
    secret_key: str
    gemini_model: str
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()