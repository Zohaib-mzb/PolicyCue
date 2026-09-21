"""Evaluation-only diagnostics, sequential request pacing and bounded retries.

Never persist exception text: SDK/Instructor errors can embed credentials and prompts.
Messages below are reconstructed from allowlisted classifications, not redacted guesses.
"""
import asyncio
import math
import re
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

PROVIDER_CODES = {
    'RESOURCE_EXHAUSTED', 'UNAVAILABLE', 'DEADLINE_EXCEEDED', 'INVALID_ARGUMENT',
    'PERMISSION_DENIED', 'UNAUTHENTICATED', 'NOT_FOUND', 'INTERNAL', 'UNKNOWN',
}


def exception_chain(exc: Exception) -> list[Exception]:
    found, seen = [], set()
    def visit(error):
        if not isinstance(error, BaseException) or id(error) in seen or len(found) >= 12:
            return
        seen.add(id(error))
        found.append(error)
        attempts = getattr(error, 'failed_attempts', None)
        if isinstance(attempts, (list, tuple)):
            for attempt in attempts[-1:]:
                visit(getattr(attempt, 'exception', None))
        last = getattr(error, 'last_attempt', None)
        if last is not None and hasattr(last, 'exception'):
            try:
                visit(last.exception())
            except Exception:
                pass
        visit(error.__cause__)
        visit(error.__context__)
    visit(exc)
    return found


def _status(exc):
    for value in (getattr(exc, 'code', None), getattr(exc, 'status_code', None),
                  getattr(exc, 'status', None), getattr(getattr(exc, 'response', None), 'status_code', None)):
        if type(value) is int and 100 <= value <= 599:
            return value
    return None


def _retry_after(exc):
    headers = getattr(getattr(exc, 'response', None), 'headers', None)
    value = headers.get('retry-after') or headers.get('Retry-After') if headers else None
    if isinstance(value, (str, int, float)):
        try:
            delay = float(value)
        except ValueError:
            try:
                parsed = parsedate_to_datetime(value)
                delay = (parsed - datetime.now(timezone.utc)).total_seconds()
            except (TypeError, ValueError, OverflowError):
                return None
        if math.isfinite(delay):
            return max(0.0, delay)
    return None


def diagnostic(exc: Exception, stage: str) -> dict:
    chain = exception_chain(exc)
    status = next((code for error in chain if (code := _status(error)) is not None), None)
    delay = next((delay for error in chain if (delay := _retry_after(error)) is not None), None)
    provider_code = None
    quota_scope = None
    for error in chain:
        direct_code = getattr(error, 'status', None)
        if isinstance(direct_code, str) and direct_code in PROVIDER_CODES:
            provider_code = direct_code
        payload = getattr(error, 'details', None) or getattr(error, 'response_json', None)
        if isinstance(payload, dict):
            body = payload.get('error', {})
            code = body.get('status') if isinstance(body, dict) else None
            if isinstance(code, str) and code in PROVIDER_CODES:
                provider_code = code
            details = body.get('details', []) if isinstance(body, dict) else []
            for detail in details if isinstance(details, list) else []:
                if not isinstance(detail, dict):
                    continue
                retry_delay = detail.get('retryDelay')
                if delay is None and isinstance(retry_delay, str) and re.fullmatch(r"[0-9]+(?:\.[0-9]+)?s", retry_delay):
                    parsed = float(retry_delay[:-1])
                    if math.isfinite(parsed):
                        delay = parsed
                violations = detail.get('violations', [])
                for violation in violations if isinstance(violations, list) else []:
                    identifier = violation.get('quotaId', '') if isinstance(violation, dict) else ''
                    if isinstance(identifier, str):
                        if 'perday' in identifier.lower():
                            quota_scope = 'daily'
                        elif 'perminute' in identifier.lower() and quota_scope != 'daily':
                            quota_scope = 'per_minute'
    names = {type(error).__name__ for error in chain}
    if status == 429 or provider_code == 'RESOURCE_EXHAUSTED':
        category = 'rate_or_quota'
    elif status == 503 or provider_code == 'UNAVAILABLE':
        category = 'capacity'
    elif status in {408, 504} or provider_code == 'DEADLINE_EXCEEDED' or any('Timeout' in name for name in names):
        category = 'timeout'
    elif status in {500, 502} or provider_code == 'INTERNAL':
        category = 'server_error'
    elif names & {'ValidationError', 'JSONDecodeError', 'IncompleteOutputException'}:
        category = 'structured_output_validation'
    elif status in {401, 403}:
        category = 'authentication_or_permission'
    elif status in {400, 404, 422} or provider_code == 'INVALID_ARGUMENT':
        category = 'invalid_request'
    elif 'InstructorRetryException' in names:
        category = 'instructor_retry_exhausted_unknown'
    else:
        category = 'unknown'
    root = next((error for error in reversed(chain) if type(error).__name__ not in
                 {'InstructorRetryException', 'RetryError', 'PipelineFailure'}), chain[-1])
    return {
        'stage': stage, 'exception_class': type(exc).__name__, 'exception_module': type(exc).__module__,
        'http_status': status, 'retry_after_seconds': delay, 'provider_code': provider_code,
        'category': category, 'quota_scope': quota_scope,
        'retryable': category in {'rate_or_quota', 'capacity', 'timeout', 'server_error'} and quota_scope != 'daily',
        'message': f"{category.replace('_', ' ')}" + (f' (HTTP {status})' if status else ''),
        'exception_chain': [{'class': type(error).__name__, 'module': type(error).__module__} for error in chain],
        'root_exception_class': type(root).__name__,
        'instructor_retry_exhausted': 'InstructorRetryException' in names,
    }


@dataclass(frozen=True)
class ReliabilityConfig:
    request_delay: float = 6.0
    judge_delay: float = 6.0
    max_retries: int = 2
    backoff_base: float = 60.0
    backoff_max: float = 120.0
    retry_after_limit: float = 600.0
    request_timeout: float = 90.0

    def __post_init__(self):
        for name, value in asdict(self).items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f'Invalid reliability option: {name}')
        if type(self.max_retries) is not int or self.max_retries > 5:
            raise ValueError('max_retries must be an integer between 0 and 5')
        if self.request_timeout <= 0:
            raise ValueError('request_timeout must be positive')


class Reliability:
    """Shared Gemini gate: generation and every judge subrequest share one clock."""
    def __init__(self, config=None, *, clock=time.monotonic, sleep=time.sleep, asleep=asyncio.sleep):
        self.config = config or ReliabilityConfig()
        self.clock, self.sleep, self.asleep = clock, sleep, asleep
        self.last_request = None
        self.events = []
        self.example_id = None

    def _delay(self, kind):
        spacing = self.config.judge_delay if kind == 'judge' else self.config.request_delay
        return 0 if self.last_request is None else max(0, self.last_request + spacing - self.clock())

    def _failure(self, exc, stage, attempt):
        item = diagnostic(exc, stage)
        item.update(attempt=attempt + 1, example_id=self.example_id)
        self.events.append(item)
        if not item['retryable'] or attempt >= self.config.max_retries:
            return None
        required = item['retry_after_seconds'] or 0
        if required > self.config.retry_after_limit:
            item['retry_suppressed'] = 'retry_after_exceeds_run_wait_limit'
            return None  # Never retry earlier than the server permits.
        delay = max(required, min(self.config.backoff_base * 2 ** attempt, self.config.backoff_max))
        item['scheduled_retry_seconds'] = delay
        return delay

    def call(self, operation, stage, kind=None):
        for attempt in range(self.config.max_retries + 1):
            if kind:
                delay = self._delay(kind)
                if delay:
                    self.sleep(delay)
                self.last_request = self.clock()
            try:
                return operation()
            except Exception as exc:
                delay = self._failure(exc, stage, attempt)
                if delay is None:
                    raise
                self.sleep(delay)

    async def acall(self, operation, stage, kind='judge'):
        for attempt in range(self.config.max_retries + 1):
            delay = self._delay(kind)
            if delay:
                await self.asleep(delay)
            self.last_request = self.clock()
            try:
                return await asyncio.wait_for(operation(), timeout=self.config.request_timeout)
            except Exception as exc:
                delay = self._failure(exc, stage, attempt)
                if delay is None:
                    raise
                await self.asleep(delay)
