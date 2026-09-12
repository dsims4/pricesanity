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
    """
    timestamps = pd.DatetimeIndex(candlestick_data[timestamp_column])
    if timestamps.tz is None:
        raise ValueError("Candlestick timestamps must be timezone-aware.")
    if (
        timestamps.hasnans
        or not timestamps.is_unique
        or not timestamps.is_monotonic_increasing
    ):
        raise ValueError(
            "Candlestick timestamps must be valid, unique, and chronological."
        )

    missing_columns = set(REQUIRED_SESSION_SCHEDULE_COLUMNS) - set(
        session_schedule.columns
    )
    if missing_columns:
        raise ValueError(
            "Session schedule is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )
    schedule = session_schedule.copy()
    for column in ("session_open", "session_close"):
        schedule[column] = pd.to_datetime(schedule[column], errors="coerce", utc=True)
    if schedule[["session_open", "session_close"]].isna().any().any():
        raise ValueError("Session schedule contains invalid boundaries.")
    schedule = schedule.sort_values("session_open", kind="stable")

    duration = pd.Timedelta(target_interval)
    if pd.isna(duration) or duration <= pd.Timedelta(0):
        raise ValueError("Target interval must be greater than zero.")
    lengths = schedule["session_close"] - schedule["session_open"]
    if (lengths <= pd.Timedelta(0)).any():
        raise ValueError("Session close must occur after session open.")
    if (lengths % duration != pd.Timedelta(0)).any():
        raise ValueError(
            "Session duration must be an integer multiple of target interval."
        )

    session_dates = schedule["session_open"].dt.tz_convert(session_timezone).dt.date
    closing_dates = (
        (schedule["session_close"] - pd.Timedelta(nanoseconds=1))
        .dt.tz_convert(session_timezone).dt.date
    )
    if session_dates.duplicated().any() or not session_dates.equals(closing_dates):
        raise ValueError("Schedules must describe one same-day session per date.")
    schedule_by_date = dict(
        zip(session_dates, schedule.itertuples(index=False), strict=True)
    )

    # Index once. Each scheduled session then uses two binary searches instead
    # of comparing every candle with every pair of boundaries.
    evidenced_dates = set(timestamps.tz_convert(session_timezone).date)
    evidenced_dates.update(observed_session_dates)
    evidenced_dates.update(schedule_by_date)
    eligible = np.zeros(len(candlestick_data), dtype=bool)
    trustworthy = np.zeros(len(candlestick_data), dtype=bool)
    previous_is_trustworthy = False
    for session_date in sorted(evidenced_dates):
        session = schedule_by_date.get(session_date)
        if session is None or str(session.data_condition) != AVAILABLE_DATA_CONDITION:
            previous_is_trustworthy = False
            continue
        start = int(timestamps.searchsorted(session.session_open))
        stop = int(timestamps.searchsorted(session.session_close))
        expected = pd.date_range(
            session.session_open, session.session_close, freq=duration, inclusive="left"
        )
        current_is_trustworthy = (
            timestamps[start:stop].tz_convert("UTC").equals(expected)
        )
        if current_is_trustworthy:
            trustworthy[start:stop] = True
            if previous_is_trustworthy:
                eligible[start:stop] = True
        previous_is_trustworthy = current_is_trustworthy

    return ValidatedSessions(
        eligible=candlestick_data.loc[eligible].copy(),
        reference_history=candlestick_data.loc[trustworthy].copy(),
    )


def filter_complete_sessions(
    candlestick_data: pd.DataFrame,
    session_schedule: pd.DataFrame,
    *,
    timestamp_column: str,
    target_interval: str,
) -> pd.DataFrame:
    """Return eligible sessions for callers that do not need reference history."""
    return validate_sessions(
        candlestick_data,
        session_schedule,
        timestamp_column=timestamp_column,
        target_interval=target_interval,
    ).eligible
