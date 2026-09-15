import asyncio
import time
from urllib.parse import urlparse

import httpx

from backend.app.core.config import get_settings
from backend.app.ingestion.url_security import same_site, validate_url


APIFY_ACTOR_ID = "apify/website-content-crawler"
APIFY_MAX_CANDIDATES = 3
APIFY_TIMEOUT_SECONDS = 180.0
APIFY_POLL_SECONDS = 5.0
APIFY_MAX_RETURNED_TEXT = 2 * 1024 * 1024


FALLBACK_PATH_PRIORITY = (
    "/legal-portal/legal-terms/terms-of-service",
    "/legal-portal/privacy/privacy-policy",
    "/legal",
    "/privacy-policy",
    "/privacy",
    "/terms-of-service",
    "/terms",
    "/legal-portal/privacy/cookie-policy",
    "/cookie-policy",
    "/cookies",
    "/legal-portal/legal-terms/payment-terms-of-service",
)


def _candidate_priority(url: str) -> tuple[int, int, str]:
    path = (urlparse(url).path or "/").rstrip("/").casefold() or "/"
    for index, priority_path in enumerate(FALLBACK_PATH_PRIORITY):
        if path == priority_path or path.startswith(priority_path + "/"):
            return (index, len(path), url)
    return (len(FALLBACK_PATH_PRIORITY), len(path), url)


class ApifyFetchError(RuntimeError):
    pass


def _safe_canonical_url(item: dict, source_url: str, base_url: str) -> str:
    metadata = item.get("metadata")
    if not isinstance(metadata, dict):
        return source_url
    canonical = metadata.get("canonicalUrl")
    if isinstance(canonical, str) and validate_url(canonical) and same_site(canonical, base_url):
        return canonical
    return source_url


def _normalize_item(item: dict, base_url: str) -> dict | None:
    source_url = item.get("url")
    if not isinstance(source_url, str):
        return None
    if not validate_url(source_url) or not same_site(source_url, base_url):
        return None

    text = item.get("text") or item.get("markdown")
    if not isinstance(text, str):
        return None
    text = text.strip()
    if not text:
        return None
    if len(text) > APIFY_MAX_RETURNED_TEXT:
        text = text[:APIFY_MAX_RETURNED_TEXT]

    metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
    title = metadata.get("title") if isinstance(metadata.get("title"), str) else ""

    return {
        "source_url": source_url,
        "final_url": _safe_canonical_url(item, source_url, base_url),
        "title": title.strip(),
        "text": text,
        "acquisition_method": "apify",
    }


async def _run_apify_candidate(client: httpx.AsyncClient, url: str, host: str) -> dict:
    actor_input = {
        "startUrls": [{"url": url}],
        "crawlerType": "playwright:firefox",
        "maxCrawlPages": 1,
        "maxCrawlDepth": 0,
        "initialConcurrency": 1,
        "maxConcurrency": 1,
        "maxRequestRetries": 1,
        "saveMarkdown": True,
        "saveHtml": False,
        "includeUrlGlobs": [{"glob": f"https://{host}/**"}, {"glob": f"http://{host}/**"}],
        "excludeUrlGlobs": [
            {"glob": "*login*"},
            {"glob": "*signin*"},
            {"glob": "*signup*"},
        ],
    }
    created = await client.post(
        f"https://api.apify.com/v2/acts/{APIFY_ACTOR_ID.replace('/', '~')}/runs",
        params={"memory": 2048},
        json=actor_input,
    )
    if created.status_code >= 300:
        raise ApifyFetchError("Apify actor could not be started.")
    run = created.json().get("data", {})
    run_id = run.get("id")
    if not run_id:
        raise ApifyFetchError("Apify actor response was malformed.")

    started = time.perf_counter()
    while True:
        await asyncio.sleep(APIFY_POLL_SECONDS)
        response = await client.get(f"https://api.apify.com/v2/actor-runs/{run_id}")
        response.raise_for_status()
        data = response.json().get("data", {})
        status = data.get("status")
        if status in {"SUCCEEDED", "FAILED", "ABORTED", "TIMED-OUT"}:
            if status != "SUCCEEDED":
                raise ApifyFetchError("Apify actor did not complete successfully.")
            dataset_id = data.get("defaultDatasetId")
            if not dataset_id:
                raise ApifyFetchError("Apify actor did not return a dataset.")
            items_response = await client.get(
                f"https://api.apify.com/v2/datasets/{dataset_id}/items",
                params={"format": "json", "clean": "true", "limit": 1},
            )
            items_response.raise_for_status()
            items = items_response.json()
            if not isinstance(items, list):
                raise ApifyFetchError("Apify dataset response was malformed.")
            return {
                "items": items,
                "duration_seconds": time.perf_counter() - started,
                "usage_total_usd": data.get("usageTotalUsd"),
            }


async def fetch_policy_candidates_with_apify(urls: list[str], base_url: str) -> dict:
    settings = get_settings()
    token = settings.apify_api_token
    if not token:
        return {"pages": [], "used": False, "status": "not_configured", "sent": 0}

    safe_urls = []
    for url in urls:
        if validate_url(url) and same_site(url, base_url):
            safe_urls.append(url)

    safe_urls = sorted(dict.fromkeys(safe_urls), key=_candidate_priority)[:APIFY_MAX_CANDIDATES]

    if not safe_urls:
        return {"pages": [], "used": False, "status": "no_safe_candidates", "sent": 0}

    host = urlparse(base_url).hostname or ""
    headers = {"Authorization": f"Bearer {token}"}
    pages = []
    sent = 0
    total_duration = 0.0
    total_usage = 0.0
    usage_known = False
    try:
        async with asyncio.timeout(APIFY_TIMEOUT_SECONDS):
            async with httpx.AsyncClient(timeout=30.0, headers=headers) as client:
                for url in safe_urls:
                    sent += 1
                    try:
                        result = await _run_apify_candidate(client, url, host)
                    except (ApifyFetchError, httpx.HTTPError, ValueError, TypeError):
                        continue
                    total_duration += result.get("duration_seconds", 0.0)
                    usage = result.get("usage_total_usd")
                    if isinstance(usage, int | float):
                        usage_known = True
                        total_usage += usage
                    pages.extend(
                        page for item in result.get("items", [])
                        if (page := _normalize_item(item, base_url)) is not None
                    )
    except (httpx.HTTPError, TimeoutError, ValueError, TypeError):
        pass

    return {
        "pages": pages,
        "used": True,
        "status": "succeeded" if pages else "failed",
        "sent": sent,
        "returned": len(pages),
        "duration_seconds": round(total_duration, 3),
        **({"usage_total_usd": total_usage} if usage_known else {}),
    }
