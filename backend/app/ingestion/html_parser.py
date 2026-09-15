from bs4 import BeautifulSoup


def extract_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")

    for element in soup(
        ["script", "style", "noscript", "svg"]
    ):
        element.decompose()

    text = soup.get_text(
        separator=" ",
        strip=True,
    )

    return text

def extract_policy_content(html: str) -> tuple[str, str]:
    """Prefer the article body so legal footer links cannot validate a home page."""
    soup = BeautifulSoup(html, "html.parser")
    title = soup.title.get_text(" ", strip=True) if soup.title else ""
    for element in soup(["script", "style", "noscript", "svg", "nav", "footer", "header", "form", "aside"]):
        element.decompose()
    body = soup.find("main") or soup.find("article") or soup
    headings = " ".join(el.get_text(" ", strip=True) for el in body.find_all(["h1", "h2", "h3"]))
    return body.get_text(" ", strip=True), f"{title} {headings}".strip()
