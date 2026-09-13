"""Prepare trustworthy intraday candlesticks for model dataset construction."""

from dataclasses import dataclass
from collections.abc import Iterable
from datetime import date
from pathlib import Path

import pandas as pd

from pricesanity.config import AppConfig
from pricesanity.data.clean import filter_regular_trading_hours
from pricesanity.data.databento_ingest import iter_ohlc_csv
from pricesanity.data.normalize import normalize_candlestick_data
from pricesanity.data.resample import resample_ohlc
from pricesanity.data.sessions import validate_sessions
from pricesanity.data.status import (
    add_historical_fallback_sessions,
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
        ValueError: If price data, status data, session schedules, or configuration values are
            invalid.
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

    observed_dates = set(
        candidate_candlestick_data[config.data.timestamp_column]
        .dt.tz_convert(config.data.session_timezone)
        .dt.date
    )

    # Apply session trust and normalization globally after reduction so CSV boundaries cannot
    # reset history.
    return prepare_resampled_session_tables(
        resampled_candlestick_data,
        status_data,
        data_conditions,
        config=config,
        observed_session_dates=observed_dates,
    )


def prepare_resampled_session_tables(
    resampled_candlestick_data: pd.DataFrame,
    status_data: pd.DataFrame,
    data_conditions: pd.DataFrame,
    *,
    config: AppConfig,
    observed_session_dates: Iterable[date],
) -> PreparedCandlestickSessions:
    """Apply the global trust chain after price reduction, including date evidence.

    Args:
        resampled_candlestick_data: Candlestick prices already combined into the target
            interval.
        status_data: Databento status records describing session transitions.
        data_conditions: Daily dataset-quality evidence used to determine eligibility.
        config: Project data, session, and normalization settings.
        observed_session_dates: Dates with evidence that must participate in the session trust
            chain.

    Returns:
        Aligned OHLC and normalized tables for eligible sessions.
    """

    # Reject unsupported normalization before producing features with the wrong declared meaning.
    if config.normalization.scheme != SUPPORTED_NORMALIZATION_SCHEME:
        raise ValueError(f"Unsupported normalization scheme: {config.normalization.scheme}")

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

    observed_dates = set(observed_session_dates)

    # Databento's older CME status feed lacks the scheduled closes needed to
    # build sessions. Before the first authoritative status-derived session,
    # use only the configured normal RTH grid; this recovers complete normal
    # dates without guessing historical early closes.
    session_schedule = add_historical_fallback_sessions(
        session_schedule,
        data_conditions,
        observed_dates,
        session_timezone=config.data.session_timezone,
        session_start_time=config.session.start_time,
        session_end_time=config.session.end_time,
    )

    # Dataset availability does not prove that this instrument's candles were
    # downloaded. Retain condition dates on configured trading weekdays even
    # when no prices or historical status survived, so a missing Friday cannot
    # let Monday borrow an older close. Weekend metadata alone adds no session.
    condition_dates = pd.to_datetime(data_conditions["date"], errors="raise")
    is_configured_weekday = condition_dates.dt.weekday.isin(config.session.trading_weekdays)
    adverse_conditions = ~(data_conditions["condition"].astype(str).str.lower().eq("available"))
    observed_dates.update(
        condition_dates.loc[is_configured_weekday | adverse_conditions].dt.date
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
    ].isin(
        eligible_timestamps
    )

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
        Normalized candlesticks from complete sessions with trustworthy preceding closes.
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


def prepare_csv_session_tables(
    csv_path: str | Path,
    status_data: pd.DataFrame,
    data_conditions: pd.DataFrame,
    *,
    config: AppConfig,
    csv_chunk_rows: int = 100_000,
) -> PreparedCandlestickSessions:
    """Reduce bounded CSV reads before applying global session validation.

    Only the unfinished target bucket crosses a read boundary. Global alignment
    uses the first candidate day's midnight, matching the eager resampler even
    for intervals that do not divide a calendar day evenly.

    Args:
        csv_path: Source candlestick CSV read without loading the entire file.
        status_data: Databento status records describing session transitions.
        data_conditions: Daily dataset-quality evidence used to determine eligibility.
        config: Project data, session, and normalization settings.
        csv_chunk_rows: Maximum number of source rows loaded in one CSV read.

    Returns:
        Aligned OHLC and normalized tables after bounded-memory preparation.
    """

    # Reuse the configured field consistently across source reads, aggregation, and global
    # session validation.
    timestamp_column = config.data.timestamp_column
    target_interval_duration = pd.Timedelta(config.data.target_interval)

    # A positive duration is needed to locate the final unfinished aggregation bucket.
    if target_interval_duration <= pd.Timedelta(0) or pd.isna(target_interval_duration):
        raise ValueError("Target interval must be positive")

    pending_candlesticks = None
    resampling_origin = None
    resampled_chunks = []
    observed_dates = set()

    def resample_complete_candlesticks(candlestick_chunk: pd.DataFrame) -> pd.DataFrame:
        """Reduce complete source groups using the shared session alignment.

        Args:
            candlestick_chunk: Complete groups of source candlesticks ready to resample.

        Returns:
            Candlesticks aggregated into the configured target interval.
        """

        # Apply the same aggregation rules and fixed origin to every completed source-read
        # segment.
        return resample_ohlc(
            candlestick_chunk,
            timestamp_column=timestamp_column,
            source_interval=config.data.source_interval,
            target_interval=config.data.target_interval,
            require_complete_candlesticks=config.data.require_complete_candlesticks,
            origin=resampling_origin,
        )

    # Reduce bounded source reads while carrying unfinished buckets across arbitrary CSV
    # boundaries.
    for raw_candlestick_chunk in iter_ohlc_csv(
        csv_path,
        timestamp_column=timestamp_column,
        source_timezone=config.data.source_timezone,
        chunk_rows=csv_chunk_rows,
    ):
        candidate_candlesticks = filter_regular_trading_hours(
            raw_candlestick_chunk,
            timestamp_column=timestamp_column,
            session_timezone=config.data.session_timezone,
            session_start_time=config.session.start_time,
            session_end_time=config.session.end_time,
            trading_weekdays=config.session.trading_weekdays,
        )

        # A read containing only out-of-session data contributes no candidate candles.
        if candidate_candlesticks.empty:
            continue

        # Retain dates before discarding incomplete bars so missing session evidence still
        # breaks the trust chain.
        observed_dates.update(
            candidate_candlesticks[timestamp_column]
            .dt.tz_convert(config.data.session_timezone)
            .dt.date
        )

        # Anchor every read to the same origin so chunk boundaries cannot shift resampling
        # alignment.
        if resampling_origin is None:
            resampling_origin = candidate_candlesticks[timestamp_column].iloc[0].normalize()

        # Carry the unfinished group forward rather than dropping or processing its rows twice.
        if pending_candlesticks is not None:
            candidate_candlesticks = pd.concat(
                [pending_candlesticks, candidate_candlesticks], ignore_index=True
            )

        # Locate the last bucket against the shared origin; the following CSV read may still
        # contain its missing minutes.
        last_timestamp = candidate_candlesticks[timestamp_column].iloc[-1]
        unfinished_bucket_start = (
            resampling_origin
            + ((last_timestamp - resampling_origin) // target_interval_duration)
            * target_interval_duration
        )
        unfinished_candlestick_mask = (
            candidate_candlesticks[timestamp_column] >= unfinished_bucket_start
        )
        complete_candlesticks = candidate_candlesticks.loc[~unfinished_candlestick_mask]

        # Only buckets known to be complete are reduced before the next source read.
        if not complete_candlesticks.empty:
            resampled_chunks.append(resample_complete_candlesticks(complete_candlesticks))

        # Carry only the unfinished bucket so memory remains bounded while aggregation preserves
        # continuity.
        pending_candlesticks = candidate_candlesticks.loc[unfinished_candlestick_mask].copy()

    # The final read may leave one bucket that still needs the normal completeness checks.
    if pending_candlesticks is not None:
        resampled_chunks.append(resample_complete_candlesticks(pending_candlesticks))

    # Combine only the reduced candles; keep the large one-minute source reads bounded.
    if resampled_chunks:
        resampled_candlesticks = pd.concat(resampled_chunks, ignore_index=True)
    else:
        resampled_candlesticks = pd.DataFrame(
            {
                timestamp_column: pd.Series([], dtype="datetime64[ns, UTC]"),
                **{name: pd.Series([], dtype=float) for name in ("open", "high", "low", "close")},
            }
        )

    # Apply session trust and normalization globally after reduction so CSV boundaries cannot
    # reset history.
    return prepare_resampled_session_tables(
        resampled_candlesticks,
        status_data,
        data_conditions,
        config=config,
        observed_session_dates=observed_dates,
    )
