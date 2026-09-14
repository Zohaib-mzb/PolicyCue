from backend.app.ingestion.url_security import validate_url


def test_public_https_url():
    assert validate_url("https://example.com") is True


def test_public_http_url():
    assert validate_url("http://example.com") is True


def test_localhost_is_blocked():
    assert validate_url("http://localhost") is False


def test_loopback_ip_is_blocked():
    assert validate_url("http://127.0.0.1") is False


def test_private_ip_is_blocked():
    assert validate_url("http://192.168.1.1") is False


def test_private_10_network_is_blocked():
    assert validate_url("http://10.0.0.1") is False


def test_file_scheme_is_blocked():
    assert validate_url("file:///etc/passwd") is False


def test_javascript_scheme_is_blocked():
    assert validate_url("javascript:alert(1)") is False

import pytest
from unittest.mock import patch
from backend.app.ingestion.url_security import public_addresses


@pytest.mark.parametrize("url", [
    "http://169.254.169.254/latest/meta-data/", "http://[::1]/",
    "http://224.0.0.1/", "http://[ff02::1]/", "http://100.64.0.1/",
    "http://192.0.2.1/", "http://user:pass@example.com/", "http://example.com:8080/",
    "http://[broken", "http://localhost/", "http://127.1/",
])
def test_unsafe_addresses_and_url_shapes(url):
    assert validate_url(url) is False


def test_mixed_public_private_dns_is_rejected():
    with patch("backend.app.ingestion.url_security.socket.getaddrinfo", return_value=[
        (2, 1, 6, "", ("93.184.216.34", 0)),
        (2, 1, 6, "", ("127.0.0.1", 0)),
    ]):
        assert public_addresses("example.com") == []
