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
    ask_rate_limit: int = 20
    ingestion_rate_limit: int = 3
    rate_limit_window_seconds: int = 60
    rate_limit_max_keys: int = 10_000
    external_retry_attempts: int = 3
    external_retry_base_delay_seconds: float = 0.5
    external_retry_max_delay_seconds: float = 5.0
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
