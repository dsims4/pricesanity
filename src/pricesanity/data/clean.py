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
    """Filter candlestick data to regular trading hours.

    Args:
        candlestick_data: Candlestick data to filter.
        timestamp_column: Column containing timezone-aware timestamps.
        session_timezone: Session timezone.
        session_start_time: Session start time.
        session_end_time: Session end time.
        trading_weekdays: Weekday indices eligible for trading sessions.

    Returns:
        A copy of the candlestick data within the configured weekday and time
        boundaries.

    Raises:
        ValueError: If timestamps lack a timezone or the session times are
            invalid.
    """
    # Select the timestamp column for the timezone and boundary checks below.
    candlestick_timestamps = candlestick_data[timestamp_column]

    # Reject timestamps that cannot be converted into the session timezone.
    if candlestick_timestamps.dt.tz is None:
        raise ValueError("Timestamps must be timezone-aware.")

    # Parse the configured boundaries so their hours and minutes can be used in
    # session calculations.
    parsed_session_start_time = time.fromisoformat(session_start_time)
    parsed_session_end_time = time.fromisoformat(session_end_time)

    # Convert the session start into minutes after midnight for comparison with
    # each candlestick.
    session_start_minute_of_day = (
        parsed_session_start_time.hour * 60
        + parsed_session_start_time.minute
    )

    # Convert the session end into the same unit as the session start.
    session_end_minute_of_day = (
        parsed_session_end_time.hour * 60
        + parsed_session_end_time.minute
    )

    # Reject a session that ends at or before it starts on the same day.
    if session_start_minute_of_day >= session_end_minute_of_day:
        raise ValueError("RTH session start must occur before its end.")

    # Calculate session length for the later relative-minute boundary check.
    session_duration_minutes = (
        session_end_minute_of_day - session_start_minute_of_day
    )

    # Create local timestamps for session checks while preserving stored UTC
    # timestamps for later pipeline stages.
    session_timestamps = candlestick_timestamps.dt.tz_convert(
        session_timezone
    )

    # Subtract the opening minute so zero represents the session start and the
    # session length represents its exclusive end.
    session_relative_minutes = (
        session_timestamps.dt.hour * 60
        + session_timestamps.dt.minute
        - session_start_minute_of_day
    )

    # Create a weekday mask so configured non-trading weekdays are removed.
    is_trading_weekday = session_timestamps.dt.weekday.isin(
        trading_weekdays
    )

    # Create a time mask that includes the open but excludes the closing boundary.
    is_within_session = (
        (session_relative_minutes >= 0)
        & (session_relative_minutes < session_duration_minutes)
    )

    # Combine both masks and copy the matching RTH candlesticks into new data.
    return candlestick_data.loc[
        is_trading_weekday & is_within_session
    ].copy()
