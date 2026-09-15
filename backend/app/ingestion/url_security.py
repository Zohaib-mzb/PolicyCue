from ipaddress import ip_address
from urllib.parse import urlparse
import socket


ALLOWED_SCHEMES = {"http", "https"}
MAX_URL_LENGTH = 2048


def public_addresses(hostname: str) -> list[str]:
    """Reject the entire DNS answer if any address is not public."""
    try:
        addresses = [str(ip_address(hostname))]
    except ValueError:
        try:
            addresses = list(dict.fromkeys(
                row[4][0] for row in socket.getaddrinfo(
                    hostname, None, type=socket.SOCK_STREAM,
                )
            ))
        except (OSError, UnicodeError):
            return []
    for address in addresses:
        ip = ip_address(address)
        if not ip.is_global or ip.is_multicast or ip.is_reserved:
            return []
    return addresses


def is_public_hostname(hostname: str) -> bool:
    return bool(public_addresses(hostname))


def valid_url_shape(url: str) -> bool:
    try:
        parsed = urlparse(url)
        return bool(
            len(url) <= MAX_URL_LENGTH
            and parsed.scheme.lower() in ALLOWED_SCHEMES
            and parsed.hostname
            and parsed.username is None
            and parsed.password is None
            and parsed.port in (None, 80, 443)
            and not any(ord(c) < 32 for c in url)
            and "\\" not in url
            and "%" not in parsed.hostname
        )
    except ValueError:
        return False


def same_site(url: str, base_url: str) -> bool:
    # Exact host plus its www alias only; never guess registrable domains
    # by taking the last two labels (unsafe for co.uk and hosted tenants).
    if not valid_url_shape(url) or not valid_url_shape(base_url):
        return False
    host = urlparse(url).hostname.lower().rstrip(".")
    base = urlparse(base_url).hostname.lower().rstrip(".")
    return host.removeprefix("www.") == base.removeprefix("www.")


def validate_url(url: str) -> bool:
    return valid_url_shape(url) and is_public_hostname(urlparse(url).hostname)
