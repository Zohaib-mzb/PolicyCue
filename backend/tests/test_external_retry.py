from unittest.mock import patch

import pytest
from google.genai.errors import ClientError

from backend.app.core.external_retry import (
    is_retryable_external_error,
    retry_external,
)


class TransientError(Exception):
    status_code = 503


class PermanentError(Exception):
    status_code = 400


class RetryAfterError(Exception):
    status_code = 429

    def __init__(self):
        self.response = type(
            "Response",
            (),
            {"headers": {"Retry-After": "30"}},
        )()


class Settings:
    external_retry_attempts = 3
    external_retry_base_delay_seconds = 0.5
    external_retry_max_delay_seconds = 2.0


def test_retryable_gemini_429_retries_and_succeeds(monkeypatch):
    calls = []
    error = ClientError(429, {"error": {"code": 429, "message": "quota"}})
    monkeypatch.setattr("backend.app.core.external_retry.get_settings", lambda: Settings())

    def operation():
        calls.append(1)
        if len(calls) == 1:
            raise error
        return "ok"

    with patch("backend.app.core.external_retry.time.sleep") as sleep, patch("backend.app.core.external_retry.random.uniform", return_value=0):
        assert retry_external(operation) == "ok"

    assert len(calls) == 2
    sleep.assert_called_once()


def test_retries_stop_at_configured_maximum(monkeypatch):
    monkeypatch.setattr("backend.app.core.external_retry.get_settings", lambda: Settings())

    with patch("backend.app.core.external_retry.time.sleep") as sleep:
        with pytest.raises(TransientError):
            retry_external(lambda: (_ for _ in ()).throw(TransientError()))

    assert sleep.call_count == 2


def test_backoff_is_bounded_and_respects_retry_after(monkeypatch):
    monkeypatch.setattr("backend.app.core.external_retry.get_settings", lambda: Settings())

    with patch("backend.app.core.external_retry.time.sleep") as sleep:
        with pytest.raises(RetryAfterError):
            retry_external(lambda: (_ for _ in ()).throw(RetryAfterError()))

    assert [call.args[0] for call in sleep.call_args_list] == [2.0, 2.0]


def test_permanent_errors_are_not_retried(monkeypatch):
    monkeypatch.setattr("backend.app.core.external_retry.get_settings", lambda: Settings())

    with patch("backend.app.core.external_retry.time.sleep") as sleep:
        with pytest.raises(PermanentError):
            retry_external(lambda: (_ for _ in ()).throw(PermanentError()))

    sleep.assert_not_called()


def test_retryable_status_detection():
    assert is_retryable_external_error(TransientError())
    assert not is_retryable_external_error(PermanentError())
