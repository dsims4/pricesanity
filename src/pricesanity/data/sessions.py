"""Validate candlestick sessions against Databento-derived schedules."""

import pandas as pd


# Define the schedule fields required to validate each trading session.
REQUIRED_SESSION_SCHEDULE_COLUMNS = (
    "session_open",
    "session_close",
    "data_condition",
)

# Accept only dates for which Databento reports no known data issue.
AVAILABLE_DATA_CONDITION = "available"


def filter_complete_sessions(
    candlestick_data: pd.DataFrame,
    session_schedule: pd.DataFrame,
    *,
    timestamp_column: str,
    target_interval: str,
) -> pd.DataFrame:
    """Keep candles from complete sessions with available data.

    Args:
        candlestick_data: Resampled candlestick data to validate.
        session_schedule: Date-specific session boundaries and data conditions.
        timestamp_column: Column containing timezone-aware candle timestamps.
        target_interval: Expected interval between candlesticks.

    Returns:
        A copy containing complete regular and scheduled half-sessions.

    Raises:
        ValueError: If timestamps, schedule fields, boundaries, or the target
            interval are invalid.
    """
    # Select candle timestamps for schedule comparisons below.
    candlestick_timestamps = candlestick_data[timestamp_column]

    # Reject timestamps that cannot identify absolute moments in time.
    if candlestick_timestamps.dt.tz is None:
        raise ValueError("Candlestick timestamps must be timezone-aware.")

    # Find schedule fields that are missing before inspecting any session rows.
    missing_schedule_columns = set(
        REQUIRED_SESSION_SCHEDULE_COLUMNS
    ) - set(session_schedule.columns)

    # Reject a schedule that cannot describe boundaries and data quality.
    if missing_schedule_columns:
        # Sort and join missing names so the error is stable and readable.
        missing_column_names = ", ".join(sorted(missing_schedule_columns))
        raise ValueError(
            f"Session schedule is missing required columns: "
            f"{missing_column_names}"
        )

    # Copy the schedule so timestamp parsing does not change the caller's data.
    parsed_session_schedule = session_schedule.copy()

    # Parse both boundary columns into a shared UTC representation.
    for boundary_column in ("session_open", "session_close"):
        parsed_session_schedule[boundary_column] = pd.to_datetime(
            parsed_session_schedule[boundary_column],
            errors="coerce",
            utc=True,
        )

    # Reject a schedule containing a boundary that could not be parsed.
    if parsed_session_schedule[
        ["session_open", "session_close"]
    ].isna().any().any():
        raise ValueError("Session schedule contains invalid boundaries.")

    # Subtract the boundary columns with the supported "pandas" Series method
    # so each schedule row carries its duration into later validation.
    parsed_session_schedule["session_duration"] = parsed_session_schedule[
        "session_close"
    ].sub(parsed_session_schedule["session_open"])

    # Convert the configured interval into a duration for schedule generation.
    target_duration = pd.Timedelta(target_interval)

    # Reject zero or negative intervals before dividing session durations.
    if target_duration <= pd.Timedelta(0):
        raise ValueError("Target interval must be greater than zero.")

    # Create a mask that will collect candles from every accepted session.
    is_in_complete_session = pd.Series(
        False,
        index=candlestick_data.index,
    )

    # Validate every scheduled date against its own regular or early close.
    for session in parsed_session_schedule.itertuples(index=False):
        # Skip dates that Databento marks degraded, pending, or missing.
        if session.data_condition != AVAILABLE_DATA_CONDITION:
            continue

        # Select the precomputed duration for target-interval validation.
        session_duration = session.session_duration

        # Reject boundaries that do not describe a positive trading session.
        if session_duration <= pd.Timedelta(0):
            raise ValueError("Session close must occur after session open.")

        # Calculate the target candle count expected within this session.
        duration_ratio = session_duration / target_duration

        # Require every session boundary to align with whole target candles.
        if not duration_ratio.is_integer():
            raise ValueError(
                "Session duration must be an integer multiple of target "
                "interval."
            )

        # Generate every start timestamp expected for this scheduled session.
        expected_timestamps = pd.date_range(
            start=session.session_open,
            end=session.session_close,
            freq=target_duration,
            inclusive="left",
        )

        # Mark actual candles within this session's inclusive-open interval.
        is_in_scheduled_session = (
            (candlestick_timestamps >= session.session_open)
            & (candlestick_timestamps < session.session_close)
        )

        # Select actual timestamps for exact comparison with the schedule.
        actual_timestamps = pd.DatetimeIndex(
            candlestick_timestamps.loc[is_in_scheduled_session]
        )

        # Accept the session only when no timestamp is missing or unexpected.
        if actual_timestamps.equals(expected_timestamps):
            is_in_complete_session |= is_in_scheduled_session

    # Copy accepted sessions so filtering cannot mutate the original data.
    return candlestick_data.loc[is_in_complete_session].copy()
