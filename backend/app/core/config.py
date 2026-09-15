from functools import lru_cache

from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_env: str = "development"

    gemini_api_key: str
    pinecone_api_key: str
    pinecone_index_name: str
    pinecone_embedding_model: str = "llama-text-embed-v2"
    pinecone_embedding_dimension: int = 768
    gemini_embedding_model: str | None = None
    gemini_embedding_dimension: int = 768
    secret_key: str
    gemini_model: str
    cors_allowed_origins: list[str] = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ]
    session_cookie_samesite: str = "lax"
    ask_rate_limit: int = 20
    ingestion_rate_limit: int = 3
    ingestion_client_rate_limit: int = 3
    ingestion_owner_rate_limit: int = 3
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

    @field_validator("cors_allowed_origins", mode="before")
    @classmethod
    def parse_cors_allowed_origins(cls, value):
        if isinstance(value, str):
            return [
                origin.strip()
                for origin in value.split(",")
                if origin.strip()
            ]
        return value

    @field_validator("app_env", "session_cookie_samesite")
    @classmethod
    def normalize_lowercase(cls, value: str) -> str:
        return value.strip().lower()

    @field_validator(
        "gemini_api_key",
        "pinecone_api_key",
        "pinecone_index_name",
        "pinecone_embedding_model",
        "secret_key",
        "gemini_model",
    )
    @classmethod
    def required_settings_must_not_be_blank(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("Required configuration is missing.")
        return value.strip()

    @model_validator(mode="after")
    def validate_security_settings(self):
        if self.session_cookie_samesite not in {"lax", "strict", "none"}:
            raise ValueError("SESSION_COOKIE_SAMESITE must be lax, strict, or none.")

        if self.session_cookie_samesite == "none" and self.app_env != "production":
            raise ValueError("SESSION_COOKIE_SAMESITE=none requires production Secure cookies.")

        if self.app_env == "production":
            placeholder_values = {
                "changeme",
                "change-me",
                "your_key",
                "your-secret-key",
                "secret",
                "dev-secret",
            }
            if len(self.secret_key) < 32 or self.secret_key.lower() in placeholder_values:
                raise ValueError("Production SECRET_KEY is not strong enough.")

            if not self.cors_allowed_origins:
                raise ValueError("Production CORS origins must be explicitly configured.")

            if "*" in self.cors_allowed_origins:
                raise ValueError("Production CORS origins cannot use wildcard with credentials.")

            for value in (
                self.gemini_api_key,
                self.pinecone_api_key,
                self.pinecone_index_name,
                self.pinecone_embedding_model,
                self.gemini_model,
            ):
                if value.lower() in placeholder_values:
                    raise ValueError("Production external service configuration is incomplete.")

        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
