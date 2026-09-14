from backend.app.ingestion.policy_discovery import (
    discover_policy_links,
)
from backend.app.ingestion.policy_categories import PolicyCategory


def test_discovers_policy_links():
    html = """
    <html>
        <body>
            <a href="/privacy-policy">Privacy Policy</a>
            <a href="/terms-of-service">Terms of Service</a>
            <a href="/community-guidelines">
                Community Guidelines
            </a>
            <a href="https://other-site.com/privacy">
                Other Privacy
            </a>
        </body>
    </html>
    """

    result = discover_policy_links(
        html,
        "https://example.com",
    )

    assert (
        "https://example.com/privacy-policy"
        in result[PolicyCategory.PRIVACY]
    )

    assert (
        "https://example.com/terms-of-service"
        in result[PolicyCategory.TERMS]
    )

    assert (
        "https://example.com/community-guidelines"
        in result[PolicyCategory.COMMUNITY]
    )

    assert all(
        "other-site.com" not in url
        for urls in result.values()
        for url in urls
    )

import pytest
from unittest.mock import AsyncMock, patch

from backend.app.ingestion import policy_discovery as discovery
from backend.app.ingestion.html_parser import extract_policy_content
from backend.app.ingestion.policy_categories import CATEGORY_PHRASES
from backend.app.ingestion.policy_validator import CATEGORY_EVIDENCE, CATEGORY_TITLES, classify_policy
from backend.app.ingestion.url_fetcher import URLFetchError


BASE = "https://example.com/"
POLICY_BODY = (
    "We collect personal information to provide our services and process your requests. "
    "We share personal data with service providers only for the purposes described here. "
    "You may contact us to request access, correction, or deletion of your records. "
    "We retain these records only as long as needed to fulfill our obligations. "
    "Our privacy practices apply when you use our website and create an account. "
    "We explain the available choices and how to exercise them. "
    "Please review this notice carefully before using our services and contact us with questions."
)
PRIVACY_HTML = f"<main><h1>Privacy Policy</h1><p>{POLICY_BODY}</p></main>"


@pytest.mark.parametrize("category", list(PolicyCategory))
def test_all_categories_match_links_and_substantive_content(category):
    phrase = CATEGORY_PHRASES[category][0]
    html = f'<a href="/legal/document">{phrase}</a>'
    assert discovery.discover_policy_links(html, BASE)[category] == [BASE + "legal/document"]
    evidence = ". ".join(CATEGORY_EVIDENCE[category])
    assert category in classify_policy(f"{phrase}. {evidence}. {POLICY_BODY}", CATEGORY_TITLES[category][0])
    assert category in discovery.match_categories("/legal/" + phrase.replace(" ", "-"))


@pytest.mark.parametrize("url", [
    "https://other.com/privacy", "https://example.com.attacker.com/privacy",
    "https://help.example.com/privacy", "http://127.0.0.1/privacy",
    "file:///privacy", "https://user:pass@example.com/privacy",
])
def test_discovery_rejects_outside_site(url):
    found = discovery.discover_policy_links(f'<a href="{url}">Privacy</a>', BASE)
    assert not any(found.values())


def test_www_alias_and_url_normalization():
    url = discovery.normalize_url("HTTPS://WWW.Example.COM:443/privacy?lang=en&utm_source=x&trk=a#section")
    assert url == "https://www.example.com/privacy?lang=en"
    assert discovery.normalize_url("https://example.com/legal/") == "https://example.com/legal/"
    assert discovery.normalize_url("https://example.com/Policy") != discovery.normalize_url("https://example.com/policy")
    html = '<a href="/privacy#one">Privacy</a><a href="/privacy?utm_source=x">Privacy</a>'
    assert discovery.discover_policy_links(html, BASE)[PolicyCategory.PRIVACY] == [BASE + "privacy"]
    assert discovery.discover_policy_links('<a href="https://www.example.com/privacy">Privacy</a>', BASE)[PolicyCategory.PRIVACY]


def test_robots_sitemap_extraction():
    robots = """User-agent: *
Disallow: /admin
Sitemap: https://example.com/maps.xml
sitemap: /second.xml
Sitemap: https://example.com/maps.xml
Sitemap: http://localhost/private.xml
Sitemap: https://other.com/map.xml
"""
    assert discovery.robots_sitemaps(robots, BASE) == [BASE + "maps.xml", BASE + "second.xml"]


@pytest.mark.parametrize("root,child,is_index", [("urlset", "url", False), ("sitemapindex", "sitemap", True)])
def test_sitemap_parsing(root, child, is_index):
    xml = f'<{root} xmlns="http://www.sitemaps.org/schemas/sitemap/0.9"><{child}><loc>{BASE}privacy</loc></{child}></{root}>'
    assert discovery.parse_sitemap(xml) == (is_index, [BASE + "privacy"])


@pytest.mark.parametrize("xml", ["<broken", "<html/>", '<!DOCTYPE a [<!ENTITY b "x">]><urlset/>'])
def test_unsafe_or_malformed_sitemaps_fail_safely(xml):
    with pytest.raises(ValueError):
        discovery.parse_sitemap(xml)


def test_non_policy_content_and_footer_are_rejected():
    html = f'<title>Welcome</title><main>{"Our products help your business grow. " * 30}</main><footer>Privacy Policy personal information collect share</footer>'
    text, headings = extract_policy_content(html)
    assert classify_policy(text, headings) == []
    assert classify_policy("Privacy Policy") == []
    assert classify_policy("Login " + "Sign in to continue. " * 30) == []
    assert classify_policy("Privacy Policy " + "Welcome to our store. " * 30) == []


def test_multi_category_policy():
    text = POLICY_BODY + " Cookies are saved in your browser. You can manage tracking consent and preferences."
    assert set(classify_policy(text, "Privacy Policy Cookie Policy")) == {PolicyCategory.PRIVACY, PolicyCategory.COOKIE}


@pytest.mark.anyio
async def test_multisource_discovery_deduplicates_and_survives_failures():
    responses = {
        BASE: '<a href="/privacy?utm_source=x#top">Privacy</a>',
        BASE + "robots.txt": "Sitemap: https://example.com/index.xml",
        BASE + "index.xml": f'<sitemapindex><sitemap><loc>{BASE}nested.xml</loc></sitemap></sitemapindex>',
        BASE + "nested.xml": f'<urlset><url><loc>{BASE}legal/privacy-copy</loc></url><url><loc>{BASE}ordinary-product</loc></url><url><loc>https://other.com/privacy</loc></url></urlset>',
        BASE + "legal": '<a href="/special/privacy-notice">Privacy Notice</a>',
        BASE + "privacy": PRIVACY_HTML,
        BASE + "legal/privacy-copy": PRIVACY_HTML.replace("We collect", "We   collect"),
        BASE + "special/privacy-notice": PRIVACY_HTML,
        BASE + "terms": "<h1>Sign in</h1>Please log in",
    }
    async def fake_fetch(url, **kwargs):
        if url not in responses:
            raise URLFetchError("not found")
        return url, responses[url]

    with patch.object(discovery, "validate_url", return_value=True), patch.object(discovery, "fetch_url", side_effect=fake_fetch) as fetch:
        result = await discovery.discover_website_policies(BASE)
    assert len(result["pages"]) == 1
    page = result["pages"][0]
    assert page["categories"] == [PolicyCategory.PRIVACY]
    assert set(page["source_urls"]) == {BASE + "privacy", BASE + "legal/privacy-copy", BASE + "special/privacy-notice"}
    assert result["skipped"]["duplicate_content"] == 2
    assert result["skipped"]["not_policy"] == 1
    assert result["warnings"]
    fetched = [call.args[0] for call in fetch.call_args_list]
    assert BASE + "ordinary-product" not in fetched
    assert "https://other.com/privacy" not in fetched
    assert fetched.count(BASE + "privacy") == 1
    assert all(call.kwargs["site_url"] == BASE for call in fetch.call_args_list)


@pytest.mark.anyio
async def test_candidate_limit():
    html = "".join(f'<a href="/privacy/{i}">Privacy</a>' for i in range(20))
    async def fake_fetch(url, **kwargs):
        if url == BASE:
            return url, html
        raise URLFetchError("unavailable")
    with patch.object(discovery, "validate_url", return_value=True), patch.object(discovery, "MAX_POLICY_CANDIDATES", 3), patch.object(discovery, "fetch_url", side_effect=fake_fetch) as fetch:
        result = await discovery.discover_website_policies(BASE)
    assert result["candidates"] == 3
    fetched_candidates = [c for c in fetch.call_args_list if c.kwargs["max_size"] == discovery.MAX_POLICY_RESPONSE_SIZE]
    assert len(fetched_candidates) == 3
    assert any("candidate limit" in warning for warning in result["warnings"])


@pytest.mark.anyio
async def test_sitemap_file_depth_and_inspection_limits():
    async def fake_fetch(url, **kwargs):
        if url.endswith("robots.txt"):
            return url, "Sitemap: https://example.com/map0.xml"
        if url.endswith(".xml"):
            return url, '<sitemapindex>' + ''.join(f'<sitemap><loc>{BASE}map{i}.xml</loc></sitemap>' for i in range(20)) + '</sitemapindex>'
        return url, "<html></html>"
    with patch.object(discovery, "validate_url", return_value=True), patch.object(discovery, "MAX_SITEMAP_FILES", 2), patch.object(discovery, "MAX_SITEMAP_URLS", 3), patch.object(discovery, "fetch_url", side_effect=fake_fetch) as fetch:
        result = await discovery.discover_website_policies(BASE)
    assert len([c for c in fetch.call_args_list if c.kwargs["allow_xml"]]) <= 2
    assert any("Sitemap file or URL limit" in w for w in result["warnings"])


@pytest.mark.anyio
async def test_private_input_rejected_before_fetch():
    with patch.object(discovery, "fetch_url", new_callable=AsyncMock) as fetch:
        with pytest.raises(URLFetchError):
            await discovery.discover_website_policies("http://127.0.0.1/")
    fetch.assert_not_called()


def test_privacy_help_index_and_overview_are_not_policies():
    assert classify_policy(POLICY_BODY, "Data and Privacy | LinkedIn Help Tagged in Data and Privacy") == []
    assert classify_policy(POLICY_BODY, "Privacy Our privacy philosophy") == []
    assert classify_policy(POLICY_BODY, "Privacy Policy") == [PolicyCategory.PRIVACY]


def test_malformed_links_are_ignored_without_losing_valid_links():
    html = '<a href="http://[broken">Privacy</a><a href="/privacy">Privacy Policy</a>'
    assert discovery.discover_policy_links(html, BASE)[PolicyCategory.PRIVACY] == [BASE + "privacy"]
    assert discovery.robots_sitemaps("Sitemap: http://[broken\nSitemap: /map.xml", BASE) == [BASE + "map.xml"]


@pytest.mark.anyio
async def test_sitemap_depth_limit_stops_recursive_index():
    async def fake_fetch(url, **kwargs):
        if url.endswith("robots.txt"):
            return url, "Sitemap: https://example.com/depth0.xml"
        if "/depth" in url:
            depth = int(url.rsplit("depth", 1)[1].split(".")[0])
            return url, f'<sitemapindex><sitemap><loc>{BASE}depth{depth + 1}.xml</loc></sitemap></sitemapindex>'
        return url, "<urlset/>" if url.endswith(".xml") else "<html/>"
    with patch.object(discovery, "validate_url", return_value=True), patch.object(discovery, "fetch_url", side_effect=fake_fetch) as fetch:
        result = await discovery.discover_website_policies(BASE)
    fetched = [c.args[0] for c in fetch.call_args_list]
    assert BASE + "depth2.xml" in fetched
    assert BASE + "depth3.xml" not in fetched
    assert any("depth limit" in w for w in result["warnings"])


def test_sitemap_locations_are_capped():
    xml = '<urlset>' + ''.join(f'<url><loc>{BASE}privacy/{i}</loc></url>' for i in range(10)) + '</urlset>'
    assert discovery.parse_sitemap(xml, limit=2) == (False, [BASE + "privacy/0", BASE + "privacy/1"])


@pytest.mark.anyio
async def test_discovery_deadline_reports_partial_results():
    import asyncio
    async def slow_fetch(*args, **kwargs):
        await asyncio.sleep(1)
    with patch.object(discovery, "validate_url", return_value=True), patch.object(discovery, "DISCOVERY_DEADLINE", 0.01), patch.object(discovery, "fetch_url", side_effect=slow_fetch):
        result = await discovery.discover_website_policies(BASE)
    assert result["pages"] == []
    assert any("time limit" in w for w in result["warnings"])
