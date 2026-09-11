from datetime import datetime, timezone

import pandas as pd
import pytest

from pricesanity.data.normalize import (
    normalize_candlestick,
    normalize_candlestick_data,
)
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


def test_normalize_candlestick_data_preserves_causal_alignment() -> None:
    # Use three simple candles so each expected ratio can be traced to the close
    # in the row immediately before it.
    candlestick_data = pd.DataFrame(
        {
            "ts_event": pd.date_range(
                "2026-09-08 19:55:00",
                periods=3,
                freq="5min",
                tz="UTC",
            ),
            "open": [99.0, 102.0, 104.0],
            "high": [101.0, 106.0, 109.0],
            "low": [98.0, 101.0, 103.0],
            "close": [100.0, 104.0, 108.0],
        }
    )
    original_candlestick_data = candlestick_data.copy(deep=True)

    # Normalize the final two candles without giving the first candle a
    # nonexistent earlier close.
    normalized_candlestick_data = normalize_candlestick_data(
        candlestick_data,
        timestamp_column="ts_event",
        instrument="ES",
    )

    # Each row remains aligned with its current candle and uses only the prior
    # close as the common scale for all four features.
    assert normalized_candlestick_data["ts_event"].tolist() == (
        candlestick_data["ts_event"].iloc[1:].tolist()
    )
    assert normalized_candlestick_data["instrument"].tolist() == ["ES", "ES"]
    assert normalized_candlestick_data["open_gap"].tolist() == pytest.approx(
        [0.02, 0.0]
    )
    assert normalized_candlestick_data["body"].tolist() == pytest.approx(
        [0.02, 4.0 / 104.0]
    )
    assert normalized_candlestick_data["high_from_close"].tolist() == (
        pytest.approx([0.02, 1.0 / 104.0])
    )
    assert normalized_candlestick_data["low_from_close"].tolist() == (
        pytest.approx([-0.03, -5.0 / 104.0])
    )

    # Normalization must not change the trusted candlesticks used to calculate
    # or later audit these features.
    pd.testing.assert_frame_equal(candlestick_data, original_candlestick_data)


def test_normalize_candlestick_data_rejects_naive_timestamps() -> None:
    # Naive timestamps cannot prove the absolute order needed for causal
    # previous-close references.
    candlestick_data = pd.DataFrame(
        {
            "ts_event": pd.to_datetime(
                ["2026-09-09 09:30:00", "2026-09-09 09:35:00"]
            ),
            "open": [100.0, 101.0],
            "high": [102.0, 103.0],
            "low": [99.0, 100.0],
            "close": [101.0, 102.0],
        }
    )

    # Reject the table instead of silently assuming which timezone it uses.
    with pytest.raises(ValueError, match="timezone-aware"):
        normalize_candlestick_data(
            candlestick_data,
            timestamp_column="ts_event",
            instrument="ES",
        )


def test_normalize_candlestick_data_rejects_zero_previous_close() -> None:
    # A zero close in the first row becomes an undefined denominator for the
    # next candle's relative geometry.
    candlestick_data = pd.DataFrame(
        {
            "ts_event": pd.date_range(
                "2026-09-09 13:30:00",
                periods=2,
                freq="5min",
                tz="UTC",
            ),
            "open": [0.0, 1.0],
            "high": [0.0, 2.0],
            "low": [0.0, 1.0],
            "close": [0.0, 2.0],
        }
    )

    # Stop before division rather than allowing infinite model features.
    with pytest.raises(ValueError, match="Previous closes cannot be zero"):
        normalize_candlestick_data(
            candlestick_data,
            timestamp_column="ts_event",
            instrument="ES",
        )
