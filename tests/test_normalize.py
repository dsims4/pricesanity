from datetime import datetime, timezone

import pytest

from pricesanity.data.normalize import normalize_candlestick
from pricesanity.data.schemas import RawCandlestick


def test_normalize_candlestick_calculates_relative_geometry() -> None:
    previous = RawCandlestick(
        timestamp=datetime(2026, 9, 9, 13, 30, tzinfo=timezone.utc),
        instrument="ES",
        open=99.0,
        high=101.0,
        low=98.0,
        close=100.0,
    )
    current = RawCandlestick(
        timestamp=datetime(2026, 9, 9, 13, 35, tzinfo=timezone.utc),
        instrument="ES",
        open=102.0,
        high=106.0,
        low=101.0,
        close=104.0,
    )

    normalized = normalize_candlestick(previous, current)

    assert normalized.open_gap == pytest.approx(0.02)
    assert normalized.body == pytest.approx(0.02)
    assert normalized.high_from_close == pytest.approx(0.02)
    assert normalized.low_from_close == pytest.approx(-0.03)


def test_normalize_candlestick_rejects_reversed_order() -> None:
    previous = RawCandlestick(
        timestamp=datetime(2026, 9, 9, 13, 35, tzinfo=timezone.utc),
        instrument="ES",
        open=100.0,
        high=102.0,
        low=99.0,
        close=101.0,
    )
    current = RawCandlestick(
        timestamp=datetime(2026, 9, 9, 13, 30, tzinfo=timezone.utc),
        instrument="ES",
        open=101.0,
        high=103.0,
        low=100.0,
        close=102.0,
    )

    with pytest.raises(ValueError, match="must occur before"):
        normalize_candlestick(previous, current)
        