import asyncio
from collections import deque
from hashlib import sha256
import re
from time import perf_counter
import unicodedata
from urllib.parse import parse_qsl, unquote, urlencode, urljoin, urlparse
from xml.etree import ElementTree

from bs4 import BeautifulSoup

from backend.app.ingestion.html_parser import extract_policy_content
from backend.app.ingestion.policy_categories import CATEGORY_PHRASES, PolicyCategory
from backend.app.ingestion.policy_validator import classify_policy, has_phrase, normalized_words
from backend.app.ingestion.apify_fetcher import fetch_policy_candidates_with_apify
from backend.app.ingestion.url_fetcher import URLFetchError, fetch_url
from backend.app.ingestion.url_security import same_site, valid_url_shape, validate_url


MAX_POLICY_CANDIDATES = 64
MAX_SITEMAP_FILES = 8
MAX_SITEMAP_DEPTH = 2
MAX_SITEMAP_URLS = 5000
MAX_SHALLOW_PAGES = 5
MAX_LINKS_INSPECTED = 2000
MAX_DISCOVERY_RESPONSE_SIZE = 1024 * 1024
MAX_POLICY_RESPONSE_SIZE = 2 * 1024 * 1024
MAX_CORPUS_CHARACTERS = 500_000
MAX_CONCURRENT_FETCHES = 4
DISCOVERY_DEADLINE = 180.0
SHALLOW_PATHS = ("/legal", "/help", "/support", "/policies", "/about")
TRACKING_PARAMETERS = {"trk", "trkinfo", "trackingid", "gclid", "fbclid", "msclkid"}
KNOWN_POLICY_HUB_PATHS = {"/legal", "/policies"}

COMMON_POLICY_PATHS = {
    PolicyCategory.PRIVACY: ("/legal-portal/privacy/privacy-policy", "/privacy", "/privacy-policy", "/legal/privacy", "/legal/privacy-policy"),
    PolicyCategory.TERMS: ("/legal", "/legal-portal/legal-terms/terms-of-service", "/terms", "/terms-of-service", "/terms-and-conditions", "/terms-of-use", "/legal/terms", "/legal/terms-of-service"),
    PolicyCategory.COOKIE: ("/legal-portal/privacy/cookie-policy", "/cookie-policy", "/cookies", "/legal/cookie-policy", "/legal/cookies"),
    PolicyCategory.REFUND: ("/refund-policy", "/returns", "/return-policy", "/refunds", "/legal/refund-policy"),
    PolicyCategory.SHIPPING: ("/shipping-policy", "/delivery-policy"),
    PolicyCategory.PAYMENT: ("/legal-portal/legal-terms/payment-terms-of-service", "/payment-terms", "/pricing"),
    PolicyCategory.ACCEPTABLE_USE: ("/acceptable-use-policy", "/aup"),
    PolicyCategory.ACCESSIBILITY: ("/accessibility", "/accessibility-statement"),
    PolicyCategory.DISCLAIMER: ("/disclaimer",),
    PolicyCategory.COPYRIGHT: ("/dmca", "/copyright"),
    PolicyCategory.DPA: ("/dpa", "/data-processing-agreement", "/legal/dpa"),
    PolicyCategory.COMMUNITY: ("/community-guidelines", "/legal/community-guidelines", "/legal/professional-community-policies"),
    PolicyCategory.AI: ("/ai-policy", "/legal/ai-policy"),
    PolicyCategory.CONTENT: ("/content-policy", "/legal/content-policy"),
    PolicyCategory.SAFETY: ("/safety-policy", "/legal/safety-policy"),
    PolicyCategory.USER_AGREEMENT: ("/user-agreement", "/legal/user-agreement"),
}


def normalize_url(url: str) -> str:
    if not valid_url_shape(url):
        return ""
    parsed = urlparse(url)
    host = parsed.hostname.lower().rstrip(".")
    if ":" in host:
        host = f"[{host}]"
    port = parsed.port
    if port and (parsed.scheme, port) not in {("http", 80), ("https", 443)}:
        host += f":{port}"
    query = urlencode([
        (key, value) for key, value in parse_qsl(parsed.query, keep_blank_values=True)
        if not key.lower().startswith("utm_") and key.lower() not in TRACKING_PARAMETERS
    ])
    # Preserve meaningful queries, path case, and non-root trailing slashes.
    return parsed._replace(netloc=host, path=parsed.path or "/", query=query, fragment="").geturl()


def match_categories(value: str) -> list[PolicyCategory]:
    words = normalized_words(unquote(value))
    return [category for category, phrases in CATEGORY_PHRASES.items()
            if any(has_phrase(words, phrase) for phrase in phrases)]


def is_policy_candidate_path(path: str) -> bool:
    normalized_path = unquote(path).rstrip("/").casefold() or "/"
    return normalized_path in KNOWN_POLICY_HUB_PATHS or bool(match_categories(path))


def resolve_url(value: str, base_url: str) -> str:
    try:
        return normalize_url(urljoin(base_url, value))
    except ValueError:
        return ""


def discover_policy_links(html: str, base_url: str) -> dict[PolicyCategory, list[str]]:
    results = {category: [] for category in PolicyCategory}
    soup = BeautifulSoup(html, "html.parser")
    for link in soup.find_all("a", href=True, limit=MAX_LINKS_INSPECTED):
        url = resolve_url(link["href"], base_url)
        if not same_site(url, base_url):
            continue
        for category in match_categories(f"{link.get_text(' ', strip=True)} {urlparse(url).path}"):
            if url not in results[category]:
                results[category].append(url)
    return results


def robots_sitemaps(text: str, base_url: str) -> list[str]:
    urls = []
    for line in text.splitlines():
        key, separator, value = line.partition(":")
        if separator and key.strip().lower() == "sitemap":
            url = resolve_url(value.split("#", 1)[0].strip(), base_url)
            if same_site(url, base_url) and url not in urls:
                urls.append(url)
                if len(urls) == MAX_SITEMAP_FILES:
                    break
    return urls


def parse_sitemap(xml: str, limit: int = MAX_SITEMAP_URLS) -> tuple[bool, list[str]]:
    # ElementTree does not load remote entities; reject DTD/entity declarations
    # as well to prevent internal entity expansion and ambiguous documents.
    if re.search(r"<!\s*(?:DOCTYPE|ENTITY)", xml, re.I):
        raise ValueError("Sitemap declarations are not supported.")
    try:
        root = ElementTree.fromstring(xml)
    except ElementTree.ParseError:
        raise ValueError("Malformed sitemap XML.") from None
    name = root.tag.rsplit("}", 1)[-1]
    if name not in {"sitemapindex", "urlset"}:
        raise ValueError("Unsupported sitemap format.")
    child_name = "sitemap" if name == "sitemapindex" else "url"
    urls = []
    for child in root:
        if child.tag.rsplit("}", 1)[-1] != child_name:
            continue
        for loc in child:
            if loc.tag.rsplit("}", 1)[-1] == "loc" and loc.text:
                urls.append(loc.text.strip())
                break
        if len(urls) >= limit:
            break
    return name == "sitemapindex", urls


def content_fingerprint(text: str) -> str:
    normalized = " ".join(unicodedata.normalize("NFKC", text).casefold().split())
    return sha256(normalized.encode()).hexdigest()


def _extract_policy_categories(html: str) -> tuple[str, list[PolicyCategory]]:
    text, headings = extract_policy_content(html)
    return text, classify_policy(text, headings)


async def discover_website_policies(base_url: str) -> dict:
    started_at = perf_counter()
    timings = {}
    validation_started = perf_counter()
    if not await asyncio.to_thread(validate_url, base_url):
        raise URLFetchError("URL is not allowed.")
    timings["input_validation"] = perf_counter() - validation_started
    base_url = normalize_url(base_url)
    origin = urlparse(base_url)._replace(path="/", params="", query="", fragment="").geturl()
    candidates = {}
    cached = {}
    pages = []
    fingerprints = {}
    warnings = []
    skipped = {"fetch_failed": 0, "not_policy": 0, "duplicate_content": 0}
    # Only URLs that the user supplied or that a site explicitly exposed are
    # material enough to justify the bounded fallback. Common paths are probes.
    failed_material_candidates = []
    material_coverage_limited = False
    fallback = {"used": False, "status": "not_needed", "sent": 0, "returned": 0}
    total_chars = 0
    processed_candidates = set()

    def warn(message):
        if message not in warnings:
            warnings.append(message)

    def add(url, *, material=False):
        nonlocal material_coverage_limited
        url = normalize_url(url)
        if not same_site(url, base_url):
            return
        if url in candidates:
            candidates[url] = candidates[url] or material
            return
        if len(candidates) >= MAX_POLICY_CANDIDATES:
            warn("Policy candidate limit reached; discovery is partial.")
            material_coverage_limited = True
            return
        candidates[url] = material

    def add_links(html, final_url):
        for urls in discover_policy_links(html, final_url).values():
            for url in urls:
                add(url, material=True)

    async def fetch(url, *, xml=False, policy=False):
        try:
            return await fetch_url(
                url, site_url=base_url, allow_xml=xml,
                max_size=MAX_POLICY_RESPONSE_SIZE if policy else MAX_DISCOVERY_RESPONSE_SIZE,
            )
        except URLFetchError:
            return None

    try:
        async with asyncio.timeout(DISCOVERY_DEADLINE):
            if is_policy_candidate_path(urlparse(base_url).path):
                add(base_url, material=True)

            homepage_started = perf_counter()
            # Input page and homepage (when different) are both useful seeds.
            for url in dict.fromkeys([base_url, origin]):
                result = await fetch(url)
                if result is None:
                    warn("A homepage/input page could not be fetched.")
                    continue
                final, html = result
                cached[normalize_url(final)] = result
                cached[url] = result
                add_links(html, final)
                _, categories = await asyncio.to_thread(_extract_policy_categories, html)
                if categories:
                    add(final, material=True)
            timings["homepage_input_fetch"] = perf_counter() - homepage_started

            robots_started = perf_counter()
            robots = await fetch(urljoin(origin, "robots.txt"))
            sitemap_urls = robots_sitemaps(robots[1], origin) if robots else []
            if robots is None:
                warn("robots.txt could not be fetched; other discovery sources were used.")
            timings["robots_fetch"] = perf_counter() - robots_started
            sitemap_started = perf_counter()
            queue = deque((url, 0) for url in dict.fromkeys(sitemap_urls + [urljoin(origin, "sitemap.xml")]))
            visited = set()
            inspected = 0
            while queue and len(visited) < MAX_SITEMAP_FILES and inspected < MAX_SITEMAP_URLS:
                url, depth = queue.popleft()
                if url in visited or not same_site(url, base_url):
                    continue
                visited.add(url)
                result = await fetch(url, xml=True)
                if result is None:
                    warn("One or more sitemaps could not be fetched.")
                    continue
                try:
                    is_index, urls = parse_sitemap(result[1], MAX_SITEMAP_URLS - inspected)
                except ValueError:
                    warn("One or more sitemaps were malformed or unsupported.")
                    continue
                inspected += len(urls)
                for loc in urls:
                    candidate = resolve_url(loc, result[0])
                    if not same_site(candidate, base_url):
                        continue
                    if is_index:
                        if depth < MAX_SITEMAP_DEPTH:
                            queue.append((candidate, depth + 1))
                        else:
                            warn("Sitemap depth limit reached; discovery is partial.")
                    elif is_policy_candidate_path(urlparse(candidate).path):
                        add(candidate, material=True)
            if queue or inspected >= MAX_SITEMAP_URLS:
                warn("Sitemap file or URL limit reached; discovery is partial.")
                material_coverage_limited = True
            timings["sitemap_processing"] = perf_counter() - sitemap_started

            shallow_started = perf_counter()
            for offset in range(0, MAX_SHALLOW_PAGES, MAX_CONCURRENT_FETCHES):
                urls = [urljoin(origin, path) for path in SHALLOW_PATHS[:MAX_SHALLOW_PAGES]][offset:offset + MAX_CONCURRENT_FETCHES]
                results = await asyncio.gather(*(fetch(url) for url in urls))
                for url, result in zip(urls, results):
                    if result is None:
                        warn("One or more shallow discovery pages could not be fetched.")
                        continue
                    final, html = result
                    cached[url] = cached[normalize_url(final)] = result
                    add_links(html, final)
            timings["shallow_discovery"] = perf_counter() - shallow_started

            async def process_candidates(urls):
                nonlocal total_chars, material_coverage_limited
                for offset in range(0, len(urls), MAX_CONCURRENT_FETCHES):
                    batch = urls[offset:offset + MAX_CONCURRENT_FETCHES]
                    async def get_candidate(url):
                        return cached.get(url) or await fetch(url, policy=True)
                    results = await asyncio.gather(*(get_candidate(url) for url in batch))
                    for url, result in zip(batch, results):
                        processed_candidates.add(url)
                        if result is None:
                            skipped["fetch_failed"] += 1
                            if candidates[url]:
                                failed_material_candidates.append(url)
                                material_coverage_limited = True
                            continue
                        final, html = result
                        text, categories = await asyncio.to_thread(_extract_policy_categories, html)
                        if not categories:
                            skipped["not_policy"] += 1
                            continue
                        fingerprint = content_fingerprint(text)
                        aliases = list(dict.fromkeys([normalize_url(final), url]))
                        if fingerprint in fingerprints:
                            page = fingerprints[fingerprint]
                            page["source_urls"] = list(dict.fromkeys(page["source_urls"] + aliases))
                            page["categories"] = sorted(set(page["categories"]) | set(categories))
                            skipped["duplicate_content"] += 1
                            continue
                        if total_chars + len(text) > MAX_CORPUS_CHARACTERS:
                            warn("Corpus size limit reached; some policy text was omitted.")
                            material_coverage_limited = True
                            remaining_chars = MAX_CORPUS_CHARACTERS - total_chars
                            if remaining_chars <= 0:
                                continue
                            text = text[:remaining_chars]
                        total_chars += len(text)
                        page = {"url": aliases[0], "source_urls": aliases, "text": text,
                                "categories": categories, "content_hash": fingerprint}
                        fingerprints[fingerprint] = page
                        pages.append(page)

            candidate_started = perf_counter()
            # Explicit links, direct input, and sitemap entries are evidence from
            # the site. Probe common paths only when that evidence produced fewer
            # than two useful policy pages.
            await process_candidates(list(candidates))
            if len(pages) < 2:
                for position in range(max(map(len, COMMON_POLICY_PATHS.values()))):
                    for paths in COMMON_POLICY_PATHS.values():
                        if position < len(paths):
                            add(urljoin(origin, paths[position]))
                await process_candidates([url for url in candidates if url not in processed_candidates])
            timings["candidate_fetch_and_validation"] = perf_counter() - candidate_started
    except TimeoutError:
        warn("Discovery time limit reached; results are partial.")
    fallback_started = perf_counter()
    if failed_material_candidates:
        fallback = await fetch_policy_candidates_with_apify(failed_material_candidates, base_url)
        if fallback.get("used"):
            if fallback.get("status") == "succeeded" and fallback.get("pages"):
                warn("Fallback acquisition was used for access-limited policy pages.")
            else:
                warn("Fallback acquisition could not retrieve usable policy pages.")
            for item in fallback.get("pages", []):
                text = item["text"]
                categories = await asyncio.to_thread(
                    classify_policy,
                    text,
                    f"{item.get('title', '')} {text[:500]}",
                )
                if not categories:
                    skipped["not_policy"] += 1
                    continue
                fingerprint = content_fingerprint(text)
                aliases = list(dict.fromkeys([item["final_url"], item["source_url"]]))
                if fingerprint in fingerprints:
                    page = fingerprints[fingerprint]
                    page["source_urls"] = list(dict.fromkeys(page["source_urls"] + aliases))
                    page["categories"] = sorted(set(page["categories"]) | set(categories))
                    skipped["duplicate_content"] += 1
                    continue
                if total_chars + len(text) > MAX_CORPUS_CHARACTERS:
                    warn("Corpus size limit reached; some policy text was omitted.")
                    remaining_chars = MAX_CORPUS_CHARACTERS - total_chars
                    if remaining_chars <= 0:
                        continue
                    text = text[:remaining_chars]
                total_chars += len(text)
                page = {
                    "url": aliases[0],
                    "source_urls": aliases,
                    "text": text,
                    "categories": categories,
                    "content_hash": fingerprint,
                    "acquisition_method": "apify",
                }
                fingerprints[fingerprint] = page
                pages.append(page)
    timings["apify_fallback"] = perf_counter() - fallback_started
    if skipped["fetch_failed"]:
        warn("Some policy candidates could not be fetched (missing, blocked, or unavailable).")
    if pages:
        coverage_status = "partial" if material_coverage_limited else "policies_found"
    elif failed_material_candidates:
        coverage_status = "access_limited"
    else:
        coverage_status = "no_policies_found"
    return {
        "url": base_url,
        "pages": pages,
        "warnings": warnings,
        "skipped": skipped,
        "candidates": len(candidates),
        "coverage_status": coverage_status,
        "fallback_used": "apify" if fallback.get("used") else "none",
        "fallback": {
            key: fallback.get(key)
            for key in ("status", "sent", "returned", "duration_seconds", "usage_total_usd")
            if key in fallback
        },
        "_timings": {key: round(value, 3) for key, value in timings.items()} | {
            "total": round(perf_counter() - started_at, 3),
        },
    }
