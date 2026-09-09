from datetime import datetime, timezone

import pytest

from pricesanity.data.schemas import RawCandlestick


def test_raw_candlestick_accepts_valid_ohlc() -> None:
    candlestick = RawCandlestick(
        timestamp=datetime(2026, 9, 9, 13, 30, tzinfo=timezone.utc),
        instrument="ES",
        open=6500.0,
        high=6502.0,
        low=6498.0,
        close=6501.0,
    )

    assert candlestick.candlestick_id == (
        "ES:2026-09-09T13:30:00+00:00"
    )


def test_raw_candlestick_rejects_invalid_high() -> None:
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
    with pytest.raises(ValueError, match="timezone-aware"):
        RawCandlestick(
            timestamp=datetime(2026, 9, 9, 13, 30),
            instrument="ES",
            open=6500.0,
            high=6502.0,
            low=6498.0,
            close=6501.0,
        )