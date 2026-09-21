"""Tests use local fixtures; any Python network connection is a test failure."""

import os
import socket

import pytest

# Let every GUI test share one headless Qt application instead of recreating module fixtures.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(scope="session")
def qt_application():
    """Keep the single process-wide Qt application alive for all GUI tests."""

    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


@pytest.fixture
def candlestick_data():
    """Provide one short, complete session for annotation-window tests."""

    import pandas as pd

    return pd.DataFrame(
        {
            "candlestick_id": ["candle-1", "candle-2", "candle-3"],
            "session_date": ["2026-09-09"] * 3,
            "ts_event": pd.date_range(
                "2026-09-09 13:30:00",
                periods=3,
                freq="5min",
                tz="UTC",
            ),
            "open_gap": [0.001, -0.002, 0.0],
            "open": [100.0, 101.0, 100.5],
            "high": [101.5, 102.0, 101.0],
            "low": [99.5, 100.0, 99.0],
            "close": [101.0, 100.5, 99.5],
        }
    )


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
