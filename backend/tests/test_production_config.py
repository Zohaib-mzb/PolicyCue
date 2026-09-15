import pytest
from pydantic import ValidationError

from backend.app.core.config import Settings


BASE_SETTINGS = {
    "gemini_api_key": "gemini-live-key",
    "pinecone_api_key": "pinecone-live-key",
    "pinecone_index_name": "policylens-prod",
    "pinecone_embedding_model": "gemini-embedding-2",
    "secret_key": "a" * 32,
    "gemini_model": "gemini-3.1-flash-lite",
}


def _settings(**overrides):
    values = {**BASE_SETTINGS, **overrides}
    return Settings(_env_file=None, **values)


def test_cors_origins_parse_from_comma_separated_env_value():
    settings = _settings(
        cors_allowed_origins="http://localhost:5173, https://policylens.vercel.app "
    )

    assert settings.cors_allowed_origins == [
        "http://localhost:5173",
        "https://policylens.vercel.app",
    ]


def test_production_accepts_explicit_origins_and_strong_secret():
    settings = _settings(
        app_env="production",
        cors_allowed_origins="https://policylens.vercel.app",
    )

    assert settings.app_env == "production"
    assert settings.cors_allowed_origins == ["https://policylens.vercel.app"]


@pytest.mark.parametrize(
    "secret_key",
    ["short", "change-me", "your-secret-key", "secret"],
)
def test_production_rejects_weak_or_placeholder_secret(secret_key):
    with pytest.raises(ValidationError):
        _settings(
            app_env="production",
            secret_key=secret_key,
            cors_allowed_origins="https://policylens.vercel.app",
        )


def test_production_rejects_wildcard_cors_with_credentials():
    with pytest.raises(ValidationError):
        _settings(app_env="production", cors_allowed_origins="*")


def test_production_rejects_missing_cors_origins():
    with pytest.raises(ValidationError):
        _settings(app_env="production", cors_allowed_origins="")


@pytest.mark.parametrize(
    "field_name",
    ["gemini_api_key", "pinecone_api_key", "pinecone_index_name"],
)
def test_production_rejects_placeholder_external_service_config(field_name):
    with pytest.raises(ValidationError):
        _settings(
            app_env="production",
            cors_allowed_origins="https://policylens.vercel.app",
            **{field_name: "YOUR_KEY"},
        )


def test_invalid_cookie_samesite_is_rejected():
    with pytest.raises(ValidationError):
        _settings(session_cookie_samesite="wide-open")


def test_samesite_none_requires_production_secure_cookies():
    with pytest.raises(ValidationError):
        _settings(app_env="development", session_cookie_samesite="none")


def test_production_allows_samesite_none_for_cross_site_cookie_use():
    settings = _settings(
        app_env="production",
        session_cookie_samesite="none",
        cors_allowed_origins="https://policylens.vercel.app",
    )

    assert settings.session_cookie_samesite == "none"
