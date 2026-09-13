"""Tests use local fixtures; any Python network connection is a test failure."""

import socket
import pytest


@pytest.fixture(autouse=True)
def prohibit_network(monkeypatch):
    """Fail any test that attempts an external network connection."""

    original = socket.socket.connect
    attempts = []

    def connect(sock, address):
        """Reject external network connections while allowing local Qt communication."""

        # Reject internet sockets so no test can spend vendor credit; local Qt sockets remain
        # usable.
        if sock.family in (socket.AF_INET, socket.AF_INET6):
            attempts.append(True)
            raise AssertionError("Tests must not contact network services")

        return original(sock, address)

    monkeypatch.setattr(socket.socket, "connect", connect)
    yield

    assert not attempts, "A test attempted network access"
