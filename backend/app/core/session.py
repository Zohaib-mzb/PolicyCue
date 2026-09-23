import base64
import hashlib
import hmac
import json
import time
from uuid import UUID, uuid4

from fastapi import Request, Response

from backend.app.core.config import get_settings


SESSION_COOKIE_NAME = "policylens_session"
SESSION_HEADER_NAME = "X-PolicyCue-Session"
SESSION_MAX_AGE_SECONDS = 60 * 60 * 24 * 30


def _b64encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode().rstrip("=")


def _b64decode(data: str) -> bytes:
    padding = "=" * (-len(data) % 4)
    return base64.urlsafe_b64decode(data + padding)


def _sign(payload: str) -> str:
    secret = get_settings().secret_key.encode()
    return _b64encode(
        hmac.new(secret, payload.encode(), hashlib.sha256).digest()
    )


def create_session_token(owner_id: str) -> str:
    UUID(owner_id)
    payload = _b64encode(
        json.dumps(
            {"owner_id": owner_id, "iat": int(time.time())},
            separators=(",", ":"),
        ).encode()
    )
    return f"{payload}.{_sign(payload)}"


def read_owner_id_from_token(token: str | None) -> str | None:
    if not token or "." not in token:
        return None

    payload, signature = token.split(".", 1)
    expected_signature = _sign(payload)
    if not hmac.compare_digest(signature, expected_signature):
        return None

    try:
        data = json.loads(_b64decode(payload))
        owner_id = data["owner_id"]
        issued_at = int(data["iat"])
        if int(time.time()) - issued_at > SESSION_MAX_AGE_SECONDS:
            return None
        UUID(owner_id)
    except Exception:
        return None

    return owner_id


def _set_session_cookie(
    response: Response,
    owner_id: str,
    token: str | None = None,
) -> str:
    settings = get_settings()
    session_token = token or create_session_token(owner_id)
    response.set_cookie(
        key=SESSION_COOKIE_NAME,
        value=session_token,
        max_age=SESSION_MAX_AGE_SECONDS,
        httponly=True,
        secure=settings.app_env == "production",
        samesite=settings.session_cookie_samesite,
    )
    return session_token


def get_or_create_owner_session(
    request: Request,
    response: Response,
) -> tuple[str | None, str | None]:
    header_token = request.headers.get(SESSION_HEADER_NAME)
    if header_token is not None:
        owner_id = read_owner_id_from_token(header_token)
        return (owner_id, header_token) if owner_id else (None, None)

    cookie_token = request.cookies.get(SESSION_COOKIE_NAME)
    owner_id = read_owner_id_from_token(cookie_token)
    if owner_id:
        return owner_id, cookie_token

    owner_id = str(uuid4())
    token = _set_session_cookie(response, owner_id)
    return owner_id, token


def require_owner_id(request: Request) -> str | None:
    header_token = request.headers.get(SESSION_HEADER_NAME)
    if header_token is not None:
        return read_owner_id_from_token(header_token)
    return read_owner_id_from_token(request.cookies.get(SESSION_COOKIE_NAME))
