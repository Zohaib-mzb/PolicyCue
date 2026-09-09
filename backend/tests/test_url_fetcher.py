import pytest

from backend.app.ingestion.url_fetcher import (
    URLFetchError,
    fetch_url,
)


@pytest.mark.anyio
async def test_invalid_url_is_rejected():
    with pytest.raises(URLFetchError):
        await fetch_url("http://127.0.0.1")


@pytest.mark.anyio
async def test_file_scheme_is_rejected():
    with pytest.raises(URLFetchError):
        await fetch_url("file:///etc/passwd")