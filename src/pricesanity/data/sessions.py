"""Validate candlestick sessions against Databento-derived schedules."""

from datetime import timedelta

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
    """Keep trustworthy sessions with a complete, trustworthy predecessor.

    Args:
        candlestick_data: Resampled candlestick data to validate.
        session_schedule: Date-specific session boundaries and data conditions.
        timestamp_column: Column containing timezone-aware candle timestamps.
        target_interval: Expected interval between candlesticks.

    Returns:
        A copy containing eligible regular and scheduled half-sessions.

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

    # Work from a copy because parsing and ordering boundaries should not change
    # the schedule supplied by the caller.
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

    # Put sessions into chronological order because each date's opening gap
    # depends on the trustworthy close from the scheduled session before it.
    parsed_session_schedule = parsed_session_schedule.sort_values(
        "session_open",
        kind="stable",
    )

    # Convert the parsed boundary columns into standard Python datetimes so the
    # editor and runtime agree on the types used for session arithmetic.
    session_opens = pd.DatetimeIndex(
        parsed_session_schedule["session_open"]
    ).to_pydatetime()
    session_closes = pd.DatetimeIndex(
        parsed_session_schedule["session_close"]
    ).to_pydatetime()

    # Normalize condition values before the loop so every date uses plain text
    # when its data quality is compared.
    data_conditions = parsed_session_schedule["data_condition"].astype(
        str
    ).tolist()

    # Convert the configured target interval into a standard duration so all
    # division and boundary comparisons below use known Python types.
    target_duration = pd.Timedelta(target_interval).to_pytimedelta()

    # A zero or negative interval cannot divide a session into chronological
    # candlestick groups.
    if target_duration <= timedelta(0):
        raise ValueError("Target interval must be greater than zero.")

    # Begin with every candle rejected, then admit a session only after its data
    # condition and complete timestamp sequence have both been verified.
    is_in_complete_session = pd.Series(
        False,
        index=candlestick_data.index,
    )

    # The first scheduled date has no verified predecessor inside the supplied
    # data, so it can establish a reference but cannot enter the model dataset.
    previous_session_is_trustworthy = False

    # Check each date separately because its schedule may describe either a
    # regular session or an official early close.
    for session_open, session_close, data_condition in zip(
        session_opens,
        session_closes,
        data_conditions,
        strict=True,
    ):

        # Degraded, pending, missing, and unknown dates cannot provide model
        # input or a trustworthy closing reference for the following session.
        if data_condition != AVAILABLE_DATA_CONDITION:
            previous_session_is_trustworthy = False
            continue

        # Measure this date independently because regular and half-day sessions
        # require different numbers of target candlesticks.
        session_duration = session_close - session_open

        # A close at or before the open cannot describe a chronological trading
        # session.
        if session_duration <= timedelta(0):
            raise ValueError("Session close must occur after session open.")

        # Divide this date's duration by the target interval to determine
        # whether its boundaries can contain only whole candlesticks.
        duration_ratio = (
            session_duration.total_seconds()
            / target_duration.total_seconds()
        )

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
            start=session_open,
            end=session_close,
            freq=target_duration,
            inclusive="left",
        )

        # Isolate actual candles from this date so they can be compared with the
        # expected sequence without involving neighboring sessions.
        is_in_scheduled_session = candlestick_timestamps.between(
            session_open,
            session_close,
            inclusive="left",
        )

        # Convert the selected times to the same index type produced by
        # "pandas" date_range for an exact sequence comparison.
        actual_timestamps = pd.DatetimeIndex(
            candlestick_timestamps.loc[is_in_scheduled_session]
        )

        # Record whether this date supplies every expected candle and therefore
        # ends with a trustworthy close for the following scheduled session.
        current_session_is_trustworthy = actual_timestamps.equals(
            expected_timestamps
        )

        # Admit this date only when its own candles and the preceding session's
        # closing reference have both been verified.
        if (
            current_session_is_trustworthy
            and previous_session_is_trustworthy
        ):
            is_in_complete_session.loc[is_in_scheduled_session] = True

        # Even an excluded session can establish the next session's reference
        # when its own timestamps and data condition are trustworthy.
        previous_session_is_trustworthy = current_session_is_trustworthy

    # Return a separate table so later processing cannot modify the supplied
    # candlestick data through this filtered result.
    return candlestick_data.loc[is_in_complete_session].copy()
