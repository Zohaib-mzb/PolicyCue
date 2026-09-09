import pytest

from backend.app.ingestion.policy_fetcher import _is_private_host


@pytest.mark.parametrize(
    "hostname",
    [
        "127.0.0.1",
        "localhost",
        "10.0.0.1",
        "172.16.0.1",
        "192.168.1.1",
        "::1",
    ],
)
def test_private_hosts_are_blocked(hostname):
    assert _is_private_host(hostname) is True


@pytest.mark.parametrize(
    "hostname",
    [
        "google.com",
        "example.com",
    ],
)
def test_public_hosts_are_allowed(hostname):
    assert _is_private_host(hostname) is False