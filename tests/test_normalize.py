from datetime import datetime, timezone

import pytest

from pricesanity.data.normalize import normalize_candlestick
from pricesanity.data.schemas import RawCandlestick


def test_normalize_candlestick_calculates_relative_geometry() -> None:
    # Create the earlier candlestick that supplies the reference close.
    previous_candlestick = RawCandlestick(
        timestamp=datetime(2026, 9, 9, 13, 30, tzinfo=timezone.utc),
        instrument="ES",
        open=99.0,
        high=101.0,
        low=98.0,
        close=100.0,
    )

    # Create the current candlestick with easy-to-check price differences.
    current_candlestick = RawCandlestick(
        timestamp=datetime(2026, 9, 9, 13, 35, tzinfo=timezone.utc),
        instrument="ES",
        open=102.0,
        high=106.0,
        low=101.0,
        close=104.0,
    )

    # Normalize the current geometry against the earlier close of 100.
    normalized_candlestick = normalize_candlestick(
        previous_candlestick,
        current_candlestick,
    )

    # Verify all four ratios use the expected signs and scale.
    assert normalized_candlestick.open_gap == pytest.approx(0.02)
    assert normalized_candlestick.body == pytest.approx(0.02)
    assert normalized_candlestick.high_from_close == pytest.approx(0.02)
    assert normalized_candlestick.low_from_close == pytest.approx(-0.03)


def test_normalize_candlestick_rejects_reversed_order() -> None:
    # Create a supposed previous candlestick with the later timestamp.
    previous_candlestick = RawCandlestick(
        timestamp=datetime(2026, 9, 9, 13, 35, tzinfo=timezone.utc),
        instrument="ES",
        open=100.0,
        high=102.0,
        low=99.0,
        close=101.0,
    )

    # Create a supposed current candlestick with the earlier timestamp.
    current_candlestick = RawCandlestick(
        timestamp=datetime(2026, 9, 9, 13, 30, tzinfo=timezone.utc),
        instrument="ES",
        open=101.0,
        high=103.0,
        low=100.0,
        close=102.0,
    )

    # Verify normalization rejects this noncausal ordering.
    with pytest.raises(ValueError, match="must occur before"):
        normalize_candlestick(previous_candlestick, current_candlestick)
