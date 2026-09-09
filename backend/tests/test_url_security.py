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