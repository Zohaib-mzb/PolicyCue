import random
import time
from collections.abc import Callable
from typing import TypeVar

from google.genai.errors import ClientError

from backend.app.core.config import get_settings


T = TypeVar("T")

RETRYABLE_STATUS_CODES = {408, 409, 429, 500, 502, 503, 504}


class ExternalServiceUnavailable(RuntimeError):
    pass


def _status_code(exc: Exception) -> int | None:
    code = getattr(exc, "code", None)
    if isinstance(code, int):
        return code

    status = getattr(exc, "status", None) or getattr(exc, "status_code", None)
    if isinstance(status, int):
        return status

    response = getattr(exc, "response", None)
    status = getattr(response, "status_code", None)
    if isinstance(status, int):
        return status

    return None


def _retry_after_seconds(exc: Exception) -> float | None:
    value = getattr(exc, "retry_after", None)
    if value is None:
        headers = getattr(exc, "headers", None)
        if not headers:
            response = getattr(exc, "response", None)
            headers = getattr(response, "headers", None)
        if headers:
            value = headers.get("retry-after") or headers.get("Retry-After")
    try:
        delay = float(value)
    except (TypeError, ValueError):
        return None

    if delay < 0:
        return None

    return delay


def is_retryable_external_error(exc: Exception) -> bool:
    if isinstance(exc, ClientError):
        return exc.code == 429

    status = _status_code(exc)
    return status in RETRYABLE_STATUS_CODES


def _retry_delay(exc: Exception, attempt: int) -> float:
    settings = get_settings()
    retry_after = _retry_after_seconds(exc)
    if retry_after is not None:
        return min(retry_after, settings.external_retry_max_delay_seconds)

    base = settings.external_retry_base_delay_seconds * (2 ** attempt)
    jitter = random.uniform(0, settings.external_retry_base_delay_seconds)
    return min(base + jitter, settings.external_retry_max_delay_seconds)


def retry_external(operation: Callable[[], T]) -> T:
    settings = get_settings()
    attempts = max(1, settings.external_retry_attempts)
    last_error: Exception | None = None

    for attempt in range(attempts):
        try:
            return operation()
        except Exception as exc:
            last_error = exc
            if not is_retryable_external_error(exc) or attempt == attempts - 1:
                raise
            time.sleep(_retry_delay(exc, attempt))

    raise ExternalServiceUnavailable() from last_error
