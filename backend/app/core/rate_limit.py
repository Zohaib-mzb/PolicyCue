import time
from collections import OrderedDict

from fastapi import HTTPException, Request

from backend.app.core.config import get_settings
from backend.app.core.session import require_owner_id


_buckets: OrderedDict[str, list[float]] = OrderedDict()


def _client_key(request: Request) -> str:
    owner_id = require_owner_id(request)
    if owner_id:
        return f"owner:{owner_id}"

    host = request.client.host if request.client else "unknown"
    return f"anonymous:{host}"


def _cleanup(now: float, window_seconds: int, max_keys: int) -> None:
    stale = [
        key
        for key, timestamps in _buckets.items()
        if not timestamps or timestamps[-1] <= now - window_seconds
    ]
    for key in stale:
        _buckets.pop(key, None)

    while len(_buckets) > max_keys:
        _buckets.popitem(last=False)


def reset_rate_limits() -> None:
    _buckets.clear()


def check_rate_limit(
    request: Request,
    *,
    scope: str,
    limit: int,
    window_seconds: int | None = None,
) -> None:
    settings = get_settings()
    window = window_seconds or settings.rate_limit_window_seconds
    now = time.monotonic()
    _cleanup(now, window, settings.rate_limit_max_keys)

    key = f"{scope}:{_client_key(request)}"
    timestamps = [
        timestamp
        for timestamp in _buckets.get(key, [])
        if timestamp > now - window
    ]

    if len(timestamps) >= limit:
        retry_after = max(1, int(timestamps[0] + window - now))
        headers = {"Retry-After": str(retry_after)}
        raise HTTPException(
            status_code=429,
            detail="Too many requests. Please retry later.",
            headers=headers,
        )

    timestamps.append(now)
    _buckets[key] = timestamps
    _buckets.move_to_end(key)
    while len(_buckets) > settings.rate_limit_max_keys:
        _buckets.popitem(last=False)
