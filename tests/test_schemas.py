from datetime import datetime, timezone

import pytest

from pricesanity.data.schemas import RawCandlestick


def test_raw_candlestick_accepts_valid_ohlc() -> None:
    # Create a valid timezone-aware candlestick.
    raw_candlestick = RawCandlestick(
        timestamp=datetime(2026, 9, 9, 13, 30, tzinfo=timezone.utc),
        instrument="ES",
        open=6500.0,
        high=6502.0,
        low=6498.0,
        close=6501.0,
    )

    # Verify its ID combines the instrument with the exact UTC timestamp.
    assert raw_candlestick.candlestick_id == (
        "ES:5min:2026-09-09T13:30:00+00:00"
    )


def test_raw_candlestick_rejects_invalid_high() -> None:
    # Verify a high below the open and close violates OHLC geometry.
    with pytest.raises(ValueError, match="High cannot be below"):
        RawCandlestick(
            timestamp=datetime(2026, 9, 9, 13, 30, tzinfo=timezone.utc),
            instrument="ES",
            open=6500.0,
            high=6499.0,
            low=6498.0,
            close=6501.0,
        )


def test_raw_candlestick_rejects_naive_timestamp() -> None:
    # Verify a timestamp without timezone information is rejected.
    with pytest.raises(ValueError, match="timezone-aware"):
        RawCandlestick(
            timestamp=datetime(2026, 9, 9, 13, 30),
            instrument="ES",
            open=6500.0,
            high=6502.0,
            low=6498.0,
            close=6501.0,
        )


def test_intervals_distinguish_candles_and_preserve_constructor_compatibility():
    from dataclasses import replace
    from pricesanity.data.normalize import normalize_candlestick

    raw = RawCandlestick(datetime(2026, 9, 9, 13, 30, tzinfo=timezone.utc),
                         "ES", 100., 102., 99., 101.)
    assert raw.interval == "5min"
    minute = replace(raw, interval="1min")
    assert minute.candlestick_id != raw.candlestick_id
    assert replace(raw, interval="300s").candlestick_id == raw.candlestick_id
    later = replace(minute, timestamp=datetime(2026, 9, 9, 13, 31, tzinfo=timezone.utc))
    normalized = normalize_candlestick(minute, later)
    assert normalized.interval == "1min"
    assert normalized.candlestick_id == later.candlestick_id
    with pytest.raises(ValueError, match="same interval"):
        normalize_candlestick(raw, later)


@pytest.mark.parametrize("interval", ["0min", "-1min", "NaT", "nonsense"])
def test_candle_rejects_invalid_interval(interval):
    Raw = RawCandlestick
    with pytest.raises(ValueError):
        Raw(datetime(2026, 9, 9, tzinfo=timezone.utc), "ES", 100, 100, 100, 100,
            interval=interval)


def test_identifier_preserves_timezone_equivalence_and_nanoseconds():
    import pandas as pd
    from pricesanity.data.identifiers import build_candlestick_id

    timestamp = pd.Timestamp("2026-09-09T13:30:00.123456789Z")
    key = build_candlestick_id("ES", timestamp, "60s")
    assert key == build_candlestick_id("ES", timestamp.tz_convert("America/New_York"), "1min")
    assert key == "ES:1min:2026-09-09T13:30:00.123456789+00:00"
