"""Prepare trustworthy intraday candlesticks for model dataset construction."""

from dataclasses import dataclass

import pandas as pd

from pricesanity.config import AppConfig
from pricesanity.data.clean import filter_regular_trading_hours
from pricesanity.data.normalize import normalize_candlestick_data
from pricesanity.data.resample import resample_ohlc
from pricesanity.data.sessions import validate_sessions
from pricesanity.data.status import (
    build_session_schedule,
    extract_session_transitions,
)

# Only the implemented relative-geometry scheme can currently produce model
# features without silently ignoring the user's configuration.
SUPPORTED_NORMALIZATION_SCHEME = "relative_ohlc_v1"


@dataclass(frozen=True)
class PreparedCandlestickSessions:
    """Validated OHLC candles and their aligned normalized features."""

    ohlc: pd.DataFrame
    normalized: pd.DataFrame


def prepare_candlestick_session_tables(
    candlestick_data: pd.DataFrame,
    status_data: pd.DataFrame,
    data_conditions: pd.DataFrame,
    *,
    config: AppConfig,
) -> PreparedCandlestickSessions:
    """Prepare aligned OHLC and normalized trustworthy intraday sessions.

    Args:
        candlestick_data: Chronological OHLC candlesticks from Databento.
        status_data: Raw Databento status records.
        data_conditions: Databento data condition for each trading date.
        config: Project data, session, and normalization settings.

    Returns:
        Validated OHLC candles and their aligned normalized features.

    Raises:
        ValueError: If price data, status data, session schedules, or
            configuration values are invalid.
    """
    # Reject an unsupported scheme before processing data because returning the
    # current features under a different configured name would mislabel them.
    if config.normalization.scheme != SUPPORTED_NORMALIZATION_SCHEME:
        raise ValueError(
            f"Unsupported normalization scheme: "
            f"{config.normalization.scheme}"
        )

    # The broad configured window removes obvious overnight and weekend data
    # before the exact Databento schedule evaluates each trading date.
    candidate_candlestick_data = filter_regular_trading_hours(
        candlestick_data,
        timestamp_column=config.data.timestamp_column,
        session_timezone=config.data.session_timezone,
        session_start_time=config.session.start_time,
        session_end_time=config.session.end_time,
        trading_weekdays=config.session.trading_weekdays,
    )

    # Source candles are combined before session validation because completeness
    # must be measured using the same target interval the model will receive.
    resampled_candlestick_data = resample_ohlc(
        candidate_candlestick_data,
        timestamp_column=config.data.timestamp_column,
        source_interval=config.data.source_interval,
        target_interval=config.data.target_interval,
        require_complete_candlesticks=(
            config.data.require_complete_candlesticks
        ),
    )

    # Raw status records contain snapshots, halts, and unrelated events, so only
    # scheduled opening and closing transitions continue into schedule building.
    session_transitions = extract_session_transitions(
        status_data,
        timestamp_column=config.data.timestamp_column,
    )

    # Date-specific boundaries preserve official early closes while attaching
    # the quality condition needed to reject unreliable historical data.
    session_schedule = build_session_schedule(
        session_transitions,
        data_conditions,
        timestamp_column=config.data.timestamp_column,
        session_timezone=config.data.session_timezone,
        session_start_time=config.session.start_time,
        session_end_time=config.session.end_time,
    )

    # Preserve date evidence before resampling can discard every incomplete bar
    # of a date. Adverse conditions also break the chain without price records.
    observed_dates = set(
        candidate_candlestick_data[config.data.timestamp_column]
        .dt.tz_convert(config.data.session_timezone).dt.date
    )
    adverse_conditions = ~(
        data_conditions["condition"].astype(str).str.lower().eq("available")
    )
    observed_dates.update(
        pd.to_datetime(data_conditions.loc[adverse_conditions, "date"]).dt.date
    )
    validated_sessions = validate_sessions(
        resampled_candlestick_data,
        session_schedule,
        timestamp_column=config.data.timestamp_column,
        target_interval=config.data.target_interval,
        session_timezone=config.data.session_timezone,
        observed_session_dates=observed_dates,
    )
    eligible_candlestick_data = validated_sessions.eligible

    # Shift only within trustworthy scheduled history. Reference-only days stay
    # available for the next eligible opening; after-close candles never do.
    normalized_candlestick_data = normalize_candlestick_data(
        validated_sessions.reference_history,
        timestamp_column=config.data.timestamp_column,
        instrument=config.data.instrument,
    )

    # Timestamps provide a stable link between validated OHLC candles and their
    # normalized forms without allowing identifiers to become model features.
    eligible_timestamps = eligible_candlestick_data[
        config.data.timestamp_column
    ]
    is_eligible_candlestick = normalized_candlestick_data[
        config.data.timestamp_column
    ].isin(eligible_timestamps)

    # Keep the human-readable OHLC candles separate from normalized model
    # features while preserving identical row and timestamp alignment.
    return PreparedCandlestickSessions(
        ohlc=eligible_candlestick_data.reset_index(drop=True),
        normalized=normalized_candlestick_data.loc[
            is_eligible_candlestick
        ].reset_index(drop=True),
    )


def prepare_candlestick_sessions(
    candlestick_data: pd.DataFrame,
    status_data: pd.DataFrame,
    data_conditions: pd.DataFrame,
    *,
    config: AppConfig,
) -> pd.DataFrame:
    """Prepare normalized candlesticks from trustworthy intraday sessions.

    Args:
        candlestick_data: Chronological OHLC candlesticks from Databento.
        status_data: Raw Databento status records.
        data_conditions: Databento data condition for each trading date.
        config: Project data, session, and normalization settings.

    Returns:
        Normalized candlesticks from complete sessions with trustworthy
        preceding closes.
    """
    # Preserve the original normalized-only interface for callers that do not
    # need to render the validated OHLC candles used during annotation.
    prepared_sessions = prepare_candlestick_session_tables(
        candlestick_data,
        status_data,
        data_conditions,
        config=config,
    )
    return prepared_sessions.normalized
