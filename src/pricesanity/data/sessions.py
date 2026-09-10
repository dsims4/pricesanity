"""Validate candlestick sessions against Databento-derived schedules."""

import pandas as pd


# Session boundaries define which timestamps should exist, while data condition
# determines whether those timestamps are trustworthy enough to keep.
REQUIRED_SESSION_SCHEDULE_COLUMNS = (
    "session_open",
    "session_close",
    "data_condition",
)

# A session is eligible only when Databento reports no known problem with that
# date's historical data.
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
    # Keep one reference to the candle timestamps because every scheduled date
    # will be compared against this same chronological series.
    candlestick_timestamps = candlestick_data[timestamp_column]

    # Timezone-aware candles are required because the schedule is stored in UTC
    # and naive values could refer to different absolute moments.
    if candlestick_timestamps.dt.tz is None:
        raise ValueError("Candlestick timestamps must be timezone-aware.")

    # Find all missing schedule fields before inspecting dates because partial
    # schedule data cannot prove either completeness or data quality.
    missing_schedule_columns = set(
        REQUIRED_SESSION_SCHEDULE_COLUMNS
    ) - set(session_schedule.columns)

    # Do not filter candles when the schedule cannot define both expected
    # timestamps and whether the underlying data is usable.
    if missing_schedule_columns:
        # Include every missing field in one stable message so the schedule can
        # be corrected without repeated validation attempts.
        missing_column_names = ", ".join(sorted(missing_schedule_columns))
        raise ValueError(
            f"Session schedule is missing required columns: "
            f"{missing_column_names}"
        )

    # Work from a copy because parsing boundaries should not change the schedule
    # supplied by the caller.
    parsed_session_schedule = session_schedule.copy()

    # Convert both boundaries to UTC so each candle and schedule timestamp uses
    # the same reference during exact comparisons.
    for boundary_column in ("session_open", "session_close"):
        parsed_session_schedule[boundary_column] = pd.to_datetime(
            parsed_session_schedule[boundary_column],
            errors="coerce",
            utc=True,
        )

    # An invalid open or close makes it impossible to determine which candles
    # belong to that session.
    if parsed_session_schedule[
        ["session_open", "session_close"]
    ].isna().any().any():
        raise ValueError("Session schedule contains invalid boundaries.")

    # Store each session's duration because regular days and scheduled
    # half-days require different numbers of target candles.
    parsed_session_schedule["session_duration"] = parsed_session_schedule[
        "session_close"
    ].sub(parsed_session_schedule["session_open"])

    # Convert the configured target interval to a "pandas" duration so it can
    # divide session lengths and generate expected candle timestamps.
    target_duration = pd.Timedelta(target_interval)

    # A zero or negative interval cannot divide a session into chronological
    # candlestick groups.
    if target_duration <= pd.Timedelta(0):
        raise ValueError("Target interval must be greater than zero.")

    # Begin with every candle rejected, then admit a session only after its data
    # condition and complete timestamp sequence have both been verified.
    is_in_complete_session = pd.Series(
        False,
        index=candlestick_data.index,
    )

    # Check each date separately because its schedule may describe either a
    # regular session or an official early close.
    for session in parsed_session_schedule.itertuples(index=False):
        # Degraded, pending, missing, and unknown dates are skipped because a
        # complete timestamp sequence does not guarantee trustworthy prices.
        if session.data_condition != AVAILABLE_DATA_CONDITION:
            continue

        # Use this date's duration because a half-day must not be measured
        # against the expected candle count of a full session.
        session_duration = session.session_duration

        # A close at or before the open cannot describe a chronological trading
        # session.
        if session_duration <= pd.Timedelta(0):
            raise ValueError("Session close must occur after session open.")

        # Divide this date's duration by the target interval to determine
        # whether its boundaries can contain only whole candlesticks.
        duration_ratio = session_duration / target_duration

        # Partial target intervals cannot produce a complete final candle, so
        # the schedule must divide into a whole number of intervals.
        if not duration_ratio.is_integer():
            raise ValueError(
                "Session duration must be an integer multiple of target "
                "interval."
            )

        # Generate the exact candle openings that a complete session should
        # contain, excluding the session close itself.
        expected_timestamps = pd.date_range(
            start=session.session_open,
            end=session.session_close,
            freq=target_duration,
            inclusive="left",
        )

        # Isolate actual candles from this date so they can be compared with the
        # expected sequence without involving neighboring sessions.
        is_in_scheduled_session = (
            (candlestick_timestamps >= session.session_open)
            & (candlestick_timestamps < session.session_close)
        )

        # Convert the selected times to the same index type produced by
        # "pandas" date_range for an exact sequence comparison.
        actual_timestamps = pd.DatetimeIndex(
            candlestick_timestamps.loc[is_in_scheduled_session]
        )

        # Admit every candle from this date only when its timestamps match the
        # schedule in value, order, and count.
        if actual_timestamps.equals(expected_timestamps):
            is_in_complete_session |= is_in_scheduled_session

    # Return a separate table so later processing cannot modify the supplied
    # candlestick data through this filtered result.
    return candlestick_data.loc[is_in_complete_session].copy()
