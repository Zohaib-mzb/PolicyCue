"""Block network access, including client construction, in evaluation tests."""
import os
import socket
from unittest.mock import MagicMock, patch

import pytest

# These are dummy test values, never real service credentials.
for name, value in {"GEMINI_API_KEY": "offline-test-key", "GEMINI_MODEL": "gemini-test",
                    "PINECONE_API_KEY": "offline-test-key", "PINECONE_INDEX_NAME": "offline-test",
                    "SECRET_KEY": "offline-test-secret", "APP_ENV": "development"}.items():
    os.environ.setdefault(name, value)
os.environ["RAGAS_DO_NOT_TRACK"] = "true"


@pytest.fixture(autouse=True)
def no_external_calls(monkeypatch):
    def blocked(*args, **kwargs):
        raise AssertionError("Network access forbidden in evaluation tests")
    monkeypatch.setattr(socket.socket, "connect", blocked)
    with patch("pinecone.Pinecone", return_value=MagicMock()):
        yield
