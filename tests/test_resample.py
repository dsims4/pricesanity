import pandas as pd
import pytest

from pricesanity.data.resample import resample_ohlc


def test_resample_ohlc_combines_source_candlesticks() -> None:
    # Create five one-minute candles with distinct OHLC values.
    candlestick_data = pd.DataFrame(
        {
            "ts_event": pd.date_range(
                "2026-09-09 13:30:00",
                periods=5,
                freq="1min",
                tz="UTC",
            ),
            "open": [100.0, 103.0, 102.0, 99.0, 101.0],
            "high": [104.0, 105.0, 103.0, 102.0, 106.0],
            "low": [99.0, 101.0, 98.0, 97.0, 100.0],
            "close": [103.0, 102.0, 99.0, 101.0, 105.0],
        }
    )

    # Combine the five source candles into one complete five-minute candle.
    resampled_candlestick_data = resample_ohlc(
        candlestick_data,
        timestamp_column="ts_event",
        source_interval="1min",
        target_interval="5min",
        require_complete_candlesticks=True,
    )

    # Verify the new candle uses the group's starting timestamp and OHLC values.
    assert len(resampled_candlestick_data) == 1
    assert resampled_candlestick_data.loc[0, "ts_event"] == pd.Timestamp(
        "2026-09-09 13:30:00",
        tz="UTC",
    )
    assert resampled_candlestick_data.loc[0, "open"] == 100.0
    assert resampled_candlestick_data.loc[0, "high"] == 106.0
    assert resampled_candlestick_data.loc[0, "low"] == 97.0
    assert resampled_candlestick_data.loc[0, "close"] == 105.0


def test_resample_ohlc_discards_incomplete_candlesticks() -> None:
    # Create one complete five-minute group followed by one partial group.
    candlestick_data = pd.DataFrame(
        {
            "ts_event": pd.date_range(
                "2026-09-09 13:30:00",
                periods=6,
                freq="1min",
                tz="UTC",
            ),
            "open": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0],
            "high": [101.0, 102.0, 103.0, 104.0, 105.0, 106.0],
            "low": [99.0, 100.0, 101.0, 102.0, 103.0, 104.0],
            "close": [100.5, 101.5, 102.5, 103.5, 104.5, 105.5],
        }
    )

    # Require each five-minute candle to contain five one-minute candles.
    resampled_candlestick_data = resample_ohlc(
        candlestick_data,
        timestamp_column="ts_event",
        source_interval="1min",
        target_interval="5min",
        require_complete_candlesticks=True,
    )

    # Keep the complete 13:30 group and discard the partial 13:35 group.
    assert len(resampled_candlestick_data) == 1
    assert resampled_candlestick_data.loc[0, "ts_event"] == pd.Timestamp(
        "2026-09-09 13:30:00",
        tz="UTC",
    )


def test_resample_ohlc_keeps_incomplete_candlesticks_when_allowed() -> None:
    # Create one complete five-minute group followed by one partial group.
    candlestick_data = pd.DataFrame(
        {
            "ts_event": pd.date_range(
                "2026-09-09 13:30:00",
                periods=6,
                freq="1min",
                tz="UTC",
            ),
            "open": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0],
            "high": [101.0, 102.0, 103.0, 104.0, 105.0, 106.0],
            "low": [99.0, 100.0, 101.0, 102.0, 103.0, 104.0],
            "close": [100.5, 101.5, 102.5, 103.5, 104.5, 105.5],
        }
    )

    # Allow a target candle even when it contains fewer than five source candles.
    resampled_candlestick_data = resample_ohlc(
        candlestick_data,
        timestamp_column="ts_event",
        source_interval="1min",
        target_interval="5min",
        require_complete_candlesticks=False,
    )

    # Keep both the complete 13:30 group and the partial 13:35 group.
    assert len(resampled_candlestick_data) == 2
    assert resampled_candlestick_data.loc[1, "ts_event"] == pd.Timestamp(
        "2026-09-09 13:35:00",
        tz="UTC",
    )


def test_resample_ohlc_rejects_nonmultiple_target_interval() -> None:
    # Create one source candlestick for testing interval validation.
    candlestick_data = pd.DataFrame(
        {
            "ts_event": pd.to_datetime(["2026-09-09 13:30:00Z"]),
            "open": [100.0],
            "high": [101.0],
            "low": [99.0],
            "close": [100.5],
        }
    )

    # Verify a ninety-second target cannot be built from whole one-minute rows.
    with pytest.raises(ValueError, match="integer multiple"):
        resample_ohlc(
            candlestick_data,
            timestamp_column="ts_event",
            source_interval="1min",
            target_interval="90s",
            require_complete_candlesticks=True,
        )
