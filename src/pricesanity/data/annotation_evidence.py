"""Preserve validated session references across the preparation-to-GUI boundary."""

import numpy as np
import pandas as pd

from pricesanity.config import AppConfig


EVIDENCE_ATTRIBUTE = "pricesanity_session_validation"
EVIDENCE_POLICY = "scheduled_available_v1"


def attach_annotation_evidence(
    ohlc: pd.DataFrame,
    normalized: pd.DataFrame,
    reference_history: pd.DataFrame,
    session_schedule: pd.DataFrame,
    *,
    config: AppConfig,
) -> None:
    """Attach session evidence after the complete-session trust chain has passed.

    Args:
        ohlc: Eligible candles whose boundaries must survive export.
        normalized: Aligned model features receiving identical metadata.
        reference_history: Complete, available sessions including reference-only days.
        session_schedule: Status-derived boundaries and dataset conditions.
        config: Configuration used to prepare the two tables.
    """

    # Metadata survives the Parquet round trip without becoming a numeric model
    # feature or requiring reference-only candles to become annotation targets.
    timestamp_column = config.data.timestamp_column
    timezone = config.data.session_timezone
    history_times = pd.DatetimeIndex(reference_history[timestamp_column])
    schedule_by_date = session_schedule.set_index("session_date")
    eligible_dates = ohlc[timestamp_column].dt.tz_convert(timezone).dt.date
    records = []

    # Save one record per eligible session, including the close of a preceding
    # trustworthy day that may deliberately be absent from the displayed table.
    for session_date, candles in ohlc.groupby(eligible_dates, sort=True):
        session = schedule_by_date.loc[session_date]
        reference_position = history_times.searchsorted(candles[timestamp_column].iloc[0]) - 1

        # A validator regression must fail preparation rather than publishing a
        # session whose opening reference has to be guessed by the GUI.
        if reference_position < 0:
            raise ValueError("Eligible session has no validated preceding close.")

        reference = reference_history.iloc[reference_position]
        reference_date = reference[timestamp_column].tz_convert(timezone).date()
        previous_session = schedule_by_date.loc[reference_date]
        records.append(
            {
                "session_open": session.session_open.isoformat(),
                "session_close": session.session_close.isoformat(),
                "condition": session.data_condition,
                "previous_session_open": previous_session.session_open.isoformat(),
                "previous_session_close": previous_session.session_close.isoformat(),
                "previous_condition": previous_session.data_condition,
                "reference_timestamp": reference[timestamp_column].isoformat(),
                "reference_close": float(reference.close),
            }
        )

    evidence = {
        "policy": EVIDENCE_POLICY,
        "instrument": config.data.instrument,
        "interval": str(pd.Timedelta(config.data.target_interval)),
        "timezone": timezone,
        "session_start": config.session.start_time,
        "session_end": config.session.end_time,
        "sessions": records,
    }
    ohlc.attrs[EVIDENCE_ATTRIBUTE] = evidence
    normalized.attrs[EVIDENCE_ATTRIBUTE] = evidence


def validate_annotation_evidence(
    ohlc: pd.DataFrame,
    normalized: pd.DataFrame,
    *,
    config: AppConfig,
) -> None:
    """Reject incomplete displayed sessions or gaps inconsistent with their references.

    Args:
        ohlc: Chronological, finite OHLC candles loaded from the prepared artifact.
        normalized: Timestamp-aligned features containing finite opening gaps.
        config: Configuration used to interpret the displayed sessions.

    Raises:
        ValueError: If metadata, complete grids, reference closes, or opening gaps disagree.
    """

    evidence = ohlc.attrs.get(EVIDENCE_ATTRIBUTE)

    # Old fallback exports cannot establish this policy. Require a fresh prepare
    # instead of silently treating a finite feature value as validity evidence.
    if (
        not isinstance(evidence, dict)
        or evidence.get("policy") != EVIDENCE_POLICY
        or evidence != normalized.attrs.get(EVIDENCE_ATTRIBUTE)
    ):
        raise ValueError("Missing or mismatched session validation evidence; rerun prepare.")

    duration = pd.Timedelta(config.data.target_interval)
    expected_settings = {
        "instrument": config.data.instrument,
        "interval": str(duration),
        "timezone": config.data.session_timezone,
        "session_start": config.session.start_time,
        "session_end": config.session.end_time,
    }

    # Configuration changes must not relabel old session or interval evidence.
    if any(evidence.get(name) != value for name, value in expected_settings.items()):
        raise ValueError("Session validation settings differ from configuration; rerun prepare.")

    timestamp_column = config.data.timestamp_column
    timestamps = pd.DatetimeIndex(ohlc[timestamp_column])
    dates = timestamps.tz_convert(config.data.session_timezone).date
    sessions = list(ohlc.groupby(dates, sort=True).indices.values())
    records = evidence.get("sessions")

    # Extract arrays once. Slicing a DataFrame per session would repeatedly
    # copy the full corpus metadata while checking only a few dozen prices.
    open_prices = ohlc["open"].to_numpy()
    close_prices = ohlc["close"].to_numpy()
    opening_gaps = normalized["open_gap"].to_numpy()

    # Exact coverage rejects trimmed sessions and metadata left over from a
    # different export, including a removed opening or closing candle.
    if not isinstance(records, list) or len(records) != len(sessions):
        raise ValueError("Session validation evidence does not cover the displayed sessions.")

    for positions, record in zip(sessions, records, strict=True):
        # Malformed saved metadata is an input error, not an invitation to infer
        # a replacement close from the first candle that happens to be visible.
        try:
            session_open = pd.Timestamp(record["session_open"])
            session_close = pd.Timestamp(record["session_close"])
            previous_open = pd.Timestamp(record["previous_session_open"])
            previous_close = pd.Timestamp(record["previous_session_close"])
            reference_timestamp = pd.Timestamp(record["reference_timestamp"])
            reference_price = float(record["reference_close"])
            available = record["condition"] == record["previous_condition"] == "available"
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("Invalid session validation evidence; rerun prepare.") from error

        # Both sessions must have authoritative quality evidence and absolute,
        # ordered boundaries. The reference must be their last complete candle.
        boundaries = (session_open, session_close, previous_open, previous_close)
        if (
            not available
            or any(pd.isna(value) or value.tzinfo is None for value in boundaries)
            or pd.isna(reference_timestamp)
            or reference_timestamp.tzinfo is None
            or not previous_open < previous_close <= session_open < session_close
            or reference_timestamp != previous_close - duration
            or (session_close - session_open) % duration != pd.Timedelta(0)
            or (previous_close - previous_open) % duration != pd.Timedelta(0)
            or not np.isfinite(reference_price)
            or reference_price == 0
        ):
            raise ValueError("Session lacks a valid preceding session close; rerun prepare.")

        expected_times = pd.date_range(
            session_open,
            session_close,
            freq=duration,
            inclusive="left",
        )

        # A regular cadence alone would accept a truncated day. Compare both
        # endpoints and every candle against the status-derived session grid.
        # ISO metadata can be parsed at microsecond precision while Parquet
        # keeps nanoseconds. Compare the same instants at one shared resolution.
        if not timestamps[positions].as_unit("ns").equals(expected_times.as_unit("ns")):
            raise ValueError("Displayed session is not full; rerun prepare.")

        # Use the preceding session close for the opening candle, then each
        # candle's actual close for the next gap. Check every displayed row.
        previous_prices = np.r_[reference_price, close_prices[positions][:-1]]
        if np.any(previous_prices == 0):
            raise ValueError("Opening gaps cannot use a zero previous close.")

        expected_gaps = (open_prices[positions] - previous_prices) / previous_prices
        actual_gaps = opening_gaps[positions]
        if not np.allclose(actual_gaps, expected_gaps, rtol=1e-12, atol=1e-12):
            raise ValueError("Opening gaps do not match validated candle closes; rerun prepare.")
