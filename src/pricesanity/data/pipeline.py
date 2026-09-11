"""Prepare trustworthy intraday candlesticks for model dataset construction."""

import pandas as pd

from pricesanity.config import AppConfig
from pricesanity.data.clean import filter_regular_trading_hours
from pricesanity.data.normalize import normalize_candlestick_data
from pricesanity.data.resample import resample_ohlc
from pricesanity.data.sessions import filter_complete_sessions
from pricesanity.data.status import (
    build_session_schedule,
    extract_session_transitions,
)

# Only the implemented relative-geometry scheme can currently produce model
# features without silently ignoring the user's configuration.
SUPPORTED_NORMALIZATION_SCHEME = "relative_ohlc_v1"


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

    # Validate raw OHLC timestamps before normalization so each accepted session
    # and its preceding closing reference are known to be complete.
    eligible_candlestick_data = filter_complete_sessions(
        resampled_candlestick_data,
        session_schedule,
        timestamp_column=config.data.timestamp_column,
        target_interval=config.data.target_interval,
    )

    # Normalize the complete resampled history before removing reference-only
    # sessions, allowing each eligible opening candle to use the preceding close.
    normalized_candlestick_data = normalize_candlestick_data(
        resampled_candlestick_data,
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

    # Return only normalized model candidates while preserving a separate table
    # that later sequence construction can group into complete sessions.
    return normalized_candlestick_data.loc[
        is_eligible_candlestick
    ].reset_index(drop=True)
