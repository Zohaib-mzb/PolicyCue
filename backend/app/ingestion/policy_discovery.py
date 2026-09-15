import asyncio
from collections import deque
from hashlib import sha256
import re
import unicodedata
from urllib.parse import parse_qsl, unquote, urlencode, urljoin, urlparse
from xml.etree import ElementTree

from bs4 import BeautifulSoup

from backend.app.ingestion.html_parser import extract_policy_content
from backend.app.ingestion.policy_categories import CATEGORY_PHRASES, PolicyCategory
from backend.app.ingestion.policy_validator import classify_policy, has_phrase, normalized_words
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

COMMON_POLICY_PATHS = {
    PolicyCategory.PRIVACY: ("/privacy", "/privacy-policy", "/legal/privacy", "/legal/privacy-policy"),
    PolicyCategory.TERMS: ("/terms", "/terms-of-service", "/terms-and-conditions", "/terms-of-use", "/legal/terms", "/legal/terms-of-service"),
    PolicyCategory.COOKIE: ("/cookie-policy", "/cookies", "/legal/cookie-policy", "/legal/cookies"),
    PolicyCategory.REFUND: ("/refund-policy", "/returns", "/return-policy", "/refunds", "/legal/refund-policy"),
    PolicyCategory.SHIPPING: ("/shipping-policy", "/delivery-policy"),
    PolicyCategory.PAYMENT: ("/payment-terms", "/pricing"),
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
    if not await asyncio.to_thread(validate_url, base_url):
        raise URLFetchError("URL is not allowed.")
    base_url = normalize_url(base_url)
    origin = urlparse(base_url)._replace(path="/", params="", query="", fragment="").geturl()
    candidates = {}
    cached = {}
    pages = []
    fingerprints = {}
    warnings = []
    skipped = {"fetch_failed": 0, "not_policy": 0, "duplicate_content": 0}
    total_chars = 0

    def warn(message):
        if message not in warnings:
            warnings.append(message)

    def add(url):
        url = normalize_url(url)
        if not same_site(url, base_url) or url in candidates:
            return
        if len(candidates) >= MAX_POLICY_CANDIDATES:
            warn("Policy candidate limit reached; discovery is partial.")
            return
        candidates[url] = None

    def add_links(html, final_url):
        for urls in discover_policy_links(html, final_url).values():
            for url in urls:
                add(url)

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
                    add(final)

            robots = await fetch(urljoin(origin, "robots.txt"))
            sitemap_urls = robots_sitemaps(robots[1], origin) if robots else []
            if robots is None:
                warn("robots.txt could not be fetched; other discovery sources were used.")
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
                    elif match_categories(urlparse(candidate).path):
                        add(candidate)
            if queue or inspected >= MAX_SITEMAP_URLS:
                warn("Sitemap file or URL limit reached; discovery is partial.")

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

            # Round-robin common paths prevents one category consuming the budget.
            for position in range(max(map(len, COMMON_POLICY_PATHS.values()))):
                for paths in COMMON_POLICY_PATHS.values():
                    if position < len(paths):
                        add(urljoin(origin, paths[position]))

            urls = list(candidates)
            for offset in range(0, len(urls), MAX_CONCURRENT_FETCHES):
                batch = urls[offset:offset + MAX_CONCURRENT_FETCHES]
                async def get_candidate(url):
                    return cached.get(url) or await fetch(url, policy=True)
                results = await asyncio.gather(*(get_candidate(url) for url in batch))
                for url, result in zip(batch, results):
                    if result is None:
                        skipped["fetch_failed"] += 1
                        continue
                    final, html = result
                    text, categories = await asyncio.to_thread(
                        _extract_policy_categories,
                        html,
                    )
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
                        warn("Corpus size limit reached; some policies were omitted.")
                        continue
                    total_chars += len(text)
                    page = {"url": aliases[0], "source_urls": aliases, "text": text,
                            "categories": categories, "content_hash": fingerprint}
                    fingerprints[fingerprint] = page
                    pages.append(page)
    except TimeoutError:
        warn("Discovery time limit reached; results are partial.")
    if skipped["fetch_failed"]:
        warn("Some policy candidates could not be fetched (missing, blocked, or unavailable).")
    return {"url": base_url, "pages": pages, "warnings": warnings,
            "skipped": skipped, "candidates": len(candidates)}
