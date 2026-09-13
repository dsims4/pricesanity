"""Resample intraday OHLC candlesticks to a larger interval."""

import pandas as pd

from pricesanity.data.databento_ingest import REQUIRED_OHLC_COLUMNS


def resample_ohlc(
    candlestick_data: pd.DataFrame,
    *,
    timestamp_column: str,
    source_interval: str,
    target_interval: str,
    require_complete_candlesticks: bool,
    origin: str | pd.Timestamp = "start_day",
) -> pd.DataFrame:
    """Resample OHLC candlestick data to a larger interval.

    Args:
        candlestick_data: Chronological OHLC candlestick data.
        timestamp_column: Column containing candlestick timestamps.
        source_interval: Current candlestick interval.
        target_interval: Desired candlestick interval.
        require_complete_candlesticks: Whether to discard incomplete results.
        origin: Shared bin alignment for eager or incremental aggregation.

    Returns:
        Resampled OHLC candlestick data labeled by interval start.

    Raises:
        ValueError: If the target interval is not a whole multiple of the source interval.
    """

    # Convert the source interval string into a "pandas" duration for interval
    # arithmetic.
    source_duration = pd.Timedelta(source_interval)

    # Convert the target interval string into the same duration type for direct
    # comparison with the source interval.
    target_duration = pd.Timedelta(target_interval)

    # Divide the durations to find how many source candles should form each
    # target candle.
    duration_ratio = target_duration / source_duration

    # Reject an invalid ratio because equal target groups require a positive
    # whole number of source intervals.
    if duration_ratio < 1 or not duration_ratio.is_integer():
        raise ValueError(
            "Target interval must be an integer multiple "
            "of source interval."
        )

    # Convert the whole-number ratio into the expected source candle count for
    # later resampling validation.
    expected_source_candlesticks = int(duration_ratio)

    # Move timestamps into the index because "pandas" resampling groups data
    # through a time-based index.
    indexed_candlestick_data = candlestick_data.set_index(timestamp_column)

    # Group source candles and use each group's start as its output timestamp.
    candlestick_resampler = indexed_candlestick_data.resample(
        target_interval,
        closed="left",
        label="left",
        origin=origin,
    )

    # Combine each group into a new candle using its first open, highest high,
    # lowest low, and last close.
    resampled_candlestick_data = candlestick_resampler.agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
        }
    )

    # Count source closes in each group for the later completeness comparison.
    source_candlestick_counts = candlestick_resampler["close"].count()

    # Remove empty groups created by "pandas" so gaps do not become NaN candles.
    resampled_candlestick_data = resampled_candlestick_data.dropna(
        subset=list(REQUIRED_OHLC_COLUMNS)
    )

    # Apply completeness validation only when partial target candles are
    # forbidden by the configuration.
    if require_complete_candlesticks:
        # Compare each group count with the expected count to mark complete
        # resampled candles.
        is_complete_candlestick = source_candlestick_counts == expected_source_candlesticks

        # Use the completeness mask to remove partial target candles.
        resampled_candlestick_data = resampled_candlestick_data.loc[
            is_complete_candlestick
        ]

    # Restore the timestamp column so the result matches the pipeline's table
    # structure.
    return resampled_candlestick_data.reset_index()
