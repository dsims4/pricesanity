"""Cleaning and session filtering for intraday candlesticks."""

from datetime import time

import pandas as pd


def filter_regular_trading_hours(
    frame: pd.DataFrame,
    *,
    timestamp_column: str,
    market_timezone: str,
    session_start: str,
    session_end: str,
    weekdays: tuple[int, ...],
) -> pd.DataFrame:
    """Return candlesticks within the configured regular session."""
    timestamps = frame[timestamp_column]

    if timestamps.dt.tz is None:
        raise ValueError("Timestamps must be timezone-aware.")

    start = time.fromisoformat(session_start)
    end = time.fromisoformat(session_end)

    start_minute = start.hour * 60 + start.minute
    end_minute = end.hour * 60 + end.minute

    if start_minute >= end_minute:
        raise ValueError("RTH session start must occur before its end.")

    local_timestamps = timestamps.dt.tz_convert(market_timezone)
    local_minutes = (
        local_timestamps.dt.hour * 60
        + local_timestamps.dt.minute
    )

    within_weekday = local_timestamps.dt.weekday.isin(weekdays)
    within_session = (
        (local_minutes >= start_minute)
        & (local_minutes < end_minute)
    )

    return frame.loc[within_weekday & within_session].copy()
