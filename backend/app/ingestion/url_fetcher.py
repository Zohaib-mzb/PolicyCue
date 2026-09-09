import httpx

from backend.app.ingestion.url_security import validate_url


MAX_RESPONSE_SIZE = 5 * 1024 * 1024
REQUEST_TIMEOUT = 10.0
MAX_REDIRECTS = 5


class URLFetchError(Exception):
    pass


async def fetch_url(url: str) -> str:
    if not validate_url(url):
        raise URLFetchError("URL is not allowed.")

    timeout = httpx.Timeout(REQUEST_TIMEOUT)

    async with httpx.AsyncClient(
        timeout=timeout,
        follow_redirects=False,
    ) as client:
        current_url = url

        for _ in range(MAX_REDIRECTS + 1):
            response = await client.get(
                current_url,
                headers={
                    "User-Agent": "PolicyLens/1.0"
                },
            )

            if response.is_redirect:
                location = response.headers.get("location")

                if not location:
                    raise URLFetchError("Invalid redirect.")

                next_url = str(
                    httpx.URL(current_url).join(location)
                )

                if not validate_url(next_url):
                    raise URLFetchError(
                        "Redirect destination is not allowed."
                    )

                current_url = next_url
                continue

            if response.status_code >= 400:
                raise URLFetchError(
                    f"Website returned HTTP {response.status_code}."
                )

            content_length = response.headers.get("content-length")

            if content_length and int(content_length) > MAX_RESPONSE_SIZE:
                raise URLFetchError("Response is too large.")

            content_type = response.headers.get(
                "content-type",
                ""
            ).lower()

            if not (
                "text/html" in content_type
                or "text/plain" in content_type
            ):
                raise URLFetchError(
                    "Unsupported content type."
                )

            if len(response.content) > MAX_RESPONSE_SIZE:
                raise URLFetchError("Response is too large.")

            return response.text

        raise URLFetchError("Too many redirects.")