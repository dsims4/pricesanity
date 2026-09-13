"""Validate candlestick sessions and their normalization references."""

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

REQUIRED_SESSION_SCHEDULE_COLUMNS = (
    "session_open",
    "session_close",
    "data_condition",
)
AVAILABLE_DATA_CONDITION = "available"


@dataclass(frozen=True)
class ValidatedSessions:
    """Eligible candles and trustworthy history including reference-only days."""

    eligible: pd.DataFrame
    reference_history: pd.DataFrame


def validate_sessions(
    candlestick_data: pd.DataFrame,
    session_schedule: pd.DataFrame,
    *,
    timestamp_column: str,
    target_interval: str,
    session_timezone: str = "America/New_York",
    observed_session_dates: Iterable[date] = (),
) -> ValidatedSessions:
    """Select sessions using one trust chain for eligibility and normalization.

    ``observed_session_dates`` includes dates seen before incomplete bars were
    discarded, and dates with adverse quality evidence. A date with evidence
    but no schedule breaks the chain. Calendar gaps with no evidence are not
    guessed to be trading days (weekends and holidays can be legitimate gaps).

    Only candles inside complete, available sessions enter reference_history.
    A session enters eligible only when the preceding evidenced session is also
    trustworthy. Thus every eligible opening uses that session's actual close.

    Args:
        candlestick_data: Chronological candlestick prices to validate or prepare.
        session_schedule: Opening, closing, and quality evidence for each session.
        timestamp_column: Column containing the candlestick or event timestamps.
        target_interval: Fixed duration of each resulting candlestick.
        session_timezone: Timezone used to identify trading-session dates.
        observed_session_dates: Dates with evidence that must participate in the session trust
            chain.

    Returns:
        Eligible candlesticks and their trustworthy reference history.

    Raises:
        ValueError: If timestamps, session boundaries, or interval alignment are invalid.
    """

    # Index timestamps once so each session can find its candles without scanning the entire
    # corpus.
    timestamps = pd.DatetimeIndex(candlestick_data[timestamp_column])

    # Session boundaries need absolute timestamps rather than ambiguous local clock readings.
    if timestamps.tz is None:
        raise ValueError("Candlestick timestamps must be timezone-aware.")

    # Invalid, repeated, or unordered timestamps cannot establish session completeness.
    if timestamps.hasnans or not timestamps.is_unique or not timestamps.is_monotonic_increasing:
        raise ValueError(
            "Candlestick timestamps must be valid, unique, and chronological."
        )

    missing_columns = set(REQUIRED_SESSION_SCHEDULE_COLUMNS) - set(
        session_schedule.columns
    )

    # A schedule must provide both boundaries and quality evidence for each session.
    if missing_columns:
        raise ValueError(
            "Session schedule is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    validated_schedule = session_schedule.copy()

    # Normalize both boundaries to UTC before comparing the schedule with candle timestamps.
    for column in ("session_open", "session_close"):
        validated_schedule[column] = pd.to_datetime(
            validated_schedule[column], errors="coerce", utc=True
        )

    # Unparseable boundaries cannot be used to decide which candles belong to a session.
    if validated_schedule[["session_open", "session_close"]].isna().any().any():
        raise ValueError("Session schedule contains invalid boundaries.")

    validated_schedule = validated_schedule.sort_values("session_open", kind="stable")

    candlestick_duration = pd.Timedelta(target_interval)

    # A positive fixed interval is required to construct the expected candle grid.
    if pd.isna(candlestick_duration) or candlestick_duration <= pd.Timedelta(0):
        raise ValueError("Target interval must be greater than zero.")

    session_durations = validated_schedule["session_close"] - validated_schedule["session_open"]

    # A closing boundary must follow its corresponding opening boundary.
    if (session_durations <= pd.Timedelta(0)).any():
        raise ValueError("Session close must occur after session open.")

    # A partial final interval would make exact candlestick completeness impossible.
    if (session_durations % candlestick_duration != pd.Timedelta(0)).any():
        raise ValueError(
            "Session duration must be an integer multiple of target interval."
        )

    session_dates = validated_schedule["session_open"].dt.tz_convert(session_timezone).dt.date
    closing_dates = (
        (validated_schedule["session_close"] - pd.Timedelta(nanoseconds=1))
        .dt.tz_convert(session_timezone)
        .dt.date
    )

    # This session model permits one same-day schedule per local trading date.
    if session_dates.duplicated().any() or not session_dates.equals(closing_dates):
        raise ValueError("Schedules must describe one same-day session per date.")

    schedule_by_date = dict(
        zip(session_dates, validated_schedule.itertuples(index=False), strict=True)
    )

    # Index once. Each scheduled session then uses two binary searches instead
    # of comparing every candle with every pair of boundaries.
    evidenced_dates = set(timestamps.tz_convert(session_timezone).date)
    evidenced_dates.update(observed_session_dates)
    evidenced_dates.update(schedule_by_date)
    eligible_candlestick_mask = np.zeros(len(candlestick_data), dtype=bool)
    trustworthy_candlestick_mask = np.zeros(len(candlestick_data), dtype=bool)
    previous_is_trustworthy = False

    # Walk all evidenced dates in order because a missing or degraded session breaks the next
    # opening reference.
    for session_date in sorted(evidenced_dates):
        session = schedule_by_date.get(session_date)

        # Missing schedules or adverse quality evidence break the preceding-session trust chain.
        if session is None or str(session.data_condition) != AVAILABLE_DATA_CONDITION:
            previous_is_trustworthy = False
            continue

        session_start_index = int(timestamps.searchsorted(session.session_open))
        session_stop_index = int(timestamps.searchsorted(session.session_close))
        expected_timestamps = pd.date_range(
            session.session_open,
            session.session_close,
            freq=candlestick_duration,
            inclusive="left",
        )
        current_is_trustworthy = (
            timestamps[session_start_index:session_stop_index]
            .tz_convert("UTC")
            .equals(expected_timestamps)
        )

        # Only a complete validated grid can become a trustworthy future opening-gap reference.
        if current_is_trustworthy:
            trustworthy_candlestick_mask[session_start_index:session_stop_index] = True

            # Training eligibility also requires a trustworthy preceding evidenced session.
            if previous_is_trustworthy:
                eligible_candlestick_mask[session_start_index:session_stop_index] = True

        previous_is_trustworthy = current_is_trustworthy

    return ValidatedSessions(
        eligible=candlestick_data.loc[eligible_candlestick_mask].copy(),
        reference_history=candlestick_data.loc[trustworthy_candlestick_mask].copy(),
    )


def filter_complete_sessions(
    candlestick_data: pd.DataFrame,
    session_schedule: pd.DataFrame,
    *,
    timestamp_column: str,
    target_interval: str,
) -> pd.DataFrame:
    """Return eligible sessions for callers that do not need reference history.

    Args:
        candlestick_data: Chronological candlestick prices to validate or prepare.
        session_schedule: Opening, closing, and quality evidence for each session.
        timestamp_column: Column containing the candlestick or event timestamps.
        target_interval: Fixed duration of each resulting candlestick.

    Returns:
        Candlesticks eligible for use with trustworthy preceding closes.
    """

    # Use the same trust-chain implementation even when the caller only needs eligible candles.
    return validate_sessions(
        candlestick_data,
        session_schedule,
        timestamp_column=timestamp_column,
        target_interval=target_interval,
    ).eligible
