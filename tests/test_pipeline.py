import pandas as pd

from pricesanity.config import (
    AppConfig,
    DataConfig,
    NormalizationConfig,
    ProjectConfig,
    SessionConfig,
)
from pricesanity.data.pipeline import prepare_candlestick_sessions


def test_prepare_candlestick_sessions_preserves_opening_gap_reference() -> None:
    # Use one complete reference date followed by one complete eligible date so
    # the second opening candle can be checked against the first session's close.
    reference_timestamps = pd.date_range(
        "2026-01-05 09:30:00",
        "2026-01-05 16:00:00",
        freq="1min",
        inclusive="left",
        tz="America/New_York",
    )
    eligible_timestamps = pd.date_range(
        "2026-01-06 09:30:00",
        "2026-01-06 16:00:00",
        freq="1min",
        inclusive="left",
        tz="America/New_York",
    )

    # Keep each minute flat within its date so resampling remains simple and the
    # two-percent overnight opening gap is the only price change.
    candlestick_data = pd.DataFrame(
        {
            "ts_event": reference_timestamps.append(
                eligible_timestamps
            ).tz_convert("UTC"),
            "open": (
                [100.0] * len(reference_timestamps)
                + [102.0] * len(eligible_timestamps)
            ),
            "high": (
                [100.0] * len(reference_timestamps)
                + [102.0] * len(eligible_timestamps)
            ),
            "low": (
                [100.0] * len(reference_timestamps)
                + [102.0] * len(eligible_timestamps)
            ),
            "close": (
                [100.0] * len(reference_timestamps)
                + [102.0] * len(eligible_timestamps)
            ),
        }
    )

    # Include the scheduled open and close surrounding each RTH date so status
    # extraction can build two complete session schedules.
    status_data = pd.DataFrame(
        {
            "ts_event": [
                "2026-01-04 23:00:00Z",
                "2026-01-05 22:00:00Z",
                "2026-01-05 23:00:00Z",
                "2026-01-06 22:00:00Z",
            ],
            "reason": ["scheduled"] * 4,
            "trading_event": ["none"] * 4,
            "is_trading": ["Y", "N", "Y", "N"],
        }
    )

    # Mark both dates available so the first can establish a trustworthy close
    # and the second can become eligible model data.
    data_conditions = pd.DataFrame(
        {
            "date": ["2026-01-05", "2026-01-06"],
            "condition": ["available", "available"],
        }
    )

    # Keep every pipeline setting explicit so the integration test documents
    # the exact one-minute to five-minute RTH transformation.
    config = AppConfig(
        project=ProjectConfig(name="pricesanity", random_seed=42),
        data=DataConfig(
            instrument="ES",
            source_timezone="UTC",
            session_timezone="America/New_York",
            timestamp_column="ts_event",
            source_interval="1min",
            target_interval="5min",
            require_complete_candlesticks=True,
        ),
        session=SessionConfig(
            start_time="09:30",
            end_time="16:00",
            trading_weekdays=(0, 1, 2, 3, 4),
        ),
        normalization=NormalizationConfig(scheme="relative_ohlc_v1"),
    )

    # Prepare the complete second date while retaining the first date only long
    # enough to normalize the eligible opening candle.
    prepared_candlestick_data = prepare_candlestick_sessions(
        candlestick_data,
        status_data,
        data_conditions,
        config=config,
    )

    # One RTH session contains seventy-eight five-minute candles, and its first
    # normalized row must preserve the two-percent move from the prior close.
    assert len(prepared_candlestick_data) == 78
    assert prepared_candlestick_data.loc[0, "ts_event"] == pd.Timestamp(
        "2026-01-06 14:30:00Z"
    )
    assert prepared_candlestick_data.loc[0, "open_gap"] == 0.02
    assert prepared_candlestick_data["body"].eq(0.0).all()
