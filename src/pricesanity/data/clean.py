"""Cleaning and session filtering for intraday candlesticks."""

from datetime import time

import pandas as pd


def filter_regular_trading_hours(
    candlestick_data: pd.DataFrame,
    *,
    timestamp_column: str,
    session_timezone: str,
    session_start_time: str,
    session_end_time: str,
    trading_weekdays: tuple[int, ...],
) -> pd.DataFrame:
    """Filter candlestick data to the maximum configured RTH boundaries.

    Args:
        candlestick_data: Candlestick data to filter.
        timestamp_column: Column containing timezone-aware timestamps.
        session_timezone: Session timezone.
        session_start_time: Session start time.
        session_end_time: Session end time.
        trading_weekdays: Weekday indices eligible for trading sessions.

    Returns:
        A copy of the candlestick data within the configured weekday and time boundaries.

    Raises:
        ValueError: If timestamps lack a timezone or the session times are invalid.
    """

    # Keep one reference to the timestamps because weekday and time boundaries
    # are both calculated from the same session-local values.
    candlestick_timestamps = candlestick_data[timestamp_column]

    # Naive timestamps cannot be converted reliably because their absolute
    # timezone, trading date, and local session time are unknown.
    if candlestick_timestamps.dt.tz is None:
        raise ValueError("Timestamps must be timezone-aware.")

    # Convert the configured clock text into time values before comparing those
    # boundaries with each candlestick.
    parsed_session_start_time = time.fromisoformat(session_start_time)
    parsed_session_end_time = time.fromisoformat(session_end_time)

    # Express the opening boundary as one number so candlestick hours and
    # minutes can be compared without separate conditions.
    session_start_minute_of_day = (
        parsed_session_start_time.hour * 60
        + parsed_session_start_time.minute
    )

    # Express the closing boundary in the same unit so the configured window
    # can be validated and measured directly.
    session_end_minute_of_day = (
        parsed_session_end_time.hour * 60
        + parsed_session_end_time.minute
    )

    # This coarse filter supports only same-day intraday windows, so the closing
    # minute must occur after the opening minute.
    if session_start_minute_of_day >= session_end_minute_of_day:
        raise ValueError("RTH session start must occur before its end.")

    # Measure the maximum session window so each candle can later be located
    # relative to zero at the configured open.
    session_duration_minutes = session_end_minute_of_day - session_start_minute_of_day

    # Convert timestamps only for the local weekday and clock checks while
    # leaving their stored UTC values unchanged for later pipeline stages.
    session_timestamps = candlestick_timestamps.dt.tz_convert(
        session_timezone
    )

    # Measure every candle from the configured open so one numeric range can
    # include the open and exclude the closing boundary.
    session_relative_minutes = (
        session_timestamps.dt.hour * 60
        + session_timestamps.dt.minute
        - session_start_minute_of_day
    )

    # Remove obvious non-trading weekdays before the exact Databento-derived
    # schedule checks each remaining date.
    is_trading_weekday = session_timestamps.dt.weekday.isin(
        trading_weekdays
    )

    # Keep possible intraday candles from the opening boundary up to, but not
    # including, the time at which the next candle interval would begin.
    is_within_session = (session_relative_minutes >= 0) & (
        session_relative_minutes < session_duration_minutes
    )

    # Return only candles eligible for later schedule validation without
    # allowing changes to the result to modify the original table.
    return candlestick_data.loc[
        is_trading_weekday & is_within_session
    ].copy()
