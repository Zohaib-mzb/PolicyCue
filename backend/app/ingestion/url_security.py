from ipaddress import ip_address
from urllib.parse import urlparse
import socket


ALLOWED_SCHEMES = {"http", "https"}


def is_public_hostname(hostname: str) -> bool:
    try:
        ip = ip_address(hostname)
        return ip.is_global
    except ValueError:
        pass

    try:
        results = socket.getaddrinfo(hostname, None)
    except socket.gaierror:
        return False

    for result in results:
        resolved_ip = result[4][0]

        try:
            if not ip_address(resolved_ip).is_global:
                return False
        except ValueError:
            return False

    return True


def validate_url(url: str) -> bool:
    parsed = urlparse(url)

    if parsed.scheme.lower() not in ALLOWED_SCHEMES:
        return False

    if not parsed.hostname:
        return False

    return is_public_hostname(parsed.hostname)