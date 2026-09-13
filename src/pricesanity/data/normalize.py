"""Causal normalization for OHLC candlesticks."""

import numpy as np
import pandas as pd

from pricesanity.data.databento_ingest import REQUIRED_OHLC_COLUMNS
from pricesanity.data.schemas import (
    NormalizedCandlestick,
    RawCandlestick,
)


def normalize_candlestick(
    previous_candlestick: RawCandlestick,
    current_candlestick: RawCandlestick,
) -> NormalizedCandlestick:
    """Normalize one candlestick against the previous close.

    Args:
        previous_candlestick: Earlier candlestick used for the reference close.
        current_candlestick: Candlestick to normalize.

    Returns:
        The current candlestick as normalized OHLC geometry.

    Raises:
        ValueError: If instruments or intervals differ, timestamps are out of order, or the
            previous close is zero.
    """

    # Require one instrument so both candlesticks share the same price scale.
    if previous_candlestick.instrument != current_candlestick.instrument:
        raise ValueError("Candlesticks must belong to the same instrument.")

    # Candles with different durations must not share a normalization reference.
    if previous_candlestick.interval != current_candlestick.interval:
        raise ValueError("Candlesticks must use the same interval.")

    # Require chronological order so normalization cannot use future prices.
    if previous_candlestick.timestamp >= current_candlestick.timestamp:
        raise ValueError(
            "The previous candlestick must occur before the current one."
        )

    # Select the previous close as a causal, shared scale for all four ratios.
    previous_close = previous_candlestick.close

    # Reject a zero denominator, which would make the ratios undefined.
    if previous_close == 0:
        raise ValueError("The previous close cannot be zero.")

    # Build dimensionless features so candles at different price levels remain
    # comparable to the model.
    return NormalizedCandlestick(
        timestamp=current_candlestick.timestamp,
        instrument=current_candlestick.instrument,
        interval=current_candlestick.interval,

        # Preserve the overnight move by measuring open from previous close.
        open_gap=(current_candlestick.open - previous_close) / previous_close,

        # Preserve candle direction by measuring the signed open-to-close body.
        body=(
            current_candlestick.close - current_candlestick.open
        ) / previous_close,

        # Preserve upper price geometry by measuring high from current close.
        high_from_close=(
            current_candlestick.high - current_candlestick.close
        ) / previous_close,

        # Preserve lower price geometry by measuring low from current close.
        low_from_close=(
            current_candlestick.low - current_candlestick.close
        ) / previous_close,
    )


def normalize_candlestick_data(
    candlestick_data: pd.DataFrame,
    *,
    timestamp_column: str,
    instrument: str,
) -> pd.DataFrame:
    """Normalize each candlestick against the previous chronological close.

    Args:
        candlestick_data: Chronological OHLC candlestick data.
        timestamp_column: Column containing candlestick timestamps.
        instrument: Instrument shared by every candlestick.

    Returns:
        Timestamps, instrument names, and normalized OHLC geometry for every candlestick that
            has a previous close.

    Raises:
        ValueError: If required fields, timestamps, prices, ordering, the instrument, or
            previous closes are invalid.
    """

    # Each output candle needs its timestamp and complete OHLC geometry before
    # it can be measured against the preceding close.
    required_candlestick_columns = (
        timestamp_column,
        *REQUIRED_OHLC_COLUMNS,
    )

    # Find all missing fields before normalization because partial candle
    # geometry cannot produce the four model features reliably.
    missing_candlestick_columns = set(required_candlestick_columns) - set(
        candlestick_data.columns
    )

    # Stop before changing any values when the input cannot identify and
    # reconstruct complete OHLC candlesticks.
    if missing_candlestick_columns:
        # Include every missing field in one stable message so the input can be
        # corrected without repeated validation attempts.
        missing_column_names = ", ".join(
            sorted(missing_candlestick_columns)
        )
        raise ValueError(
            f"Candlestick data is missing required columns: "
            f"{missing_column_names}"
        )

    # Every output row needs a stable instrument identity for later alignment
    # with annotations, predictions, and the original market data.
    if not instrument.strip():
        raise ValueError("Instrument cannot be empty.")

    # Copy only the required source fields because parsing and numeric conversion
    # must not alter the validated candlestick data supplied by the caller.
    parsed_candlestick_data = candlestick_data.loc[
        :, list(required_candlestick_columns)
    ].copy()

    # Parse timestamps without assuming a timezone because a missing timezone
    # must be rejected instead of silently interpreting local time as UTC.
    parsed_timestamps = pd.to_datetime(
        parsed_candlestick_data[timestamp_column],
        errors="coerce",
    )

    # Invalid timestamps cannot be ordered or aligned with sessions,
    # annotations, and later candle-by-candle predictions.
    if parsed_timestamps.isna().any():
        raise ValueError("Candlestick data contains invalid timestamps.")

    # Causal calculations need absolute moments, so timestamps without timezone
    # information cannot establish a reliable chronological order.
    if parsed_timestamps.dt.tz is None:
        raise ValueError("Candlestick timestamps must be timezone-aware.")

    # Store timestamps in UTC so session filtering and annotation alignment use
    # the same absolute time representation.
    parsed_candlestick_data[timestamp_column] = parsed_timestamps.dt.tz_convert("UTC")

    # Convert all OHLC fields to numbers so text or malformed prices cannot
    # produce misleading normalized ratios.
    for price_column in REQUIRED_OHLC_COLUMNS:
        parsed_candlestick_data[price_column] = pd.to_numeric(
            parsed_candlestick_data[price_column],
            errors="coerce",
        )

    # Every price must be finite because missing or infinite values would pass
    # undefined geometry into the model.
    has_only_finite_prices = np.isfinite(
        parsed_candlestick_data[list(REQUIRED_OHLC_COLUMNS)].to_numpy(dtype=float)
    ).all()

    # Relative geometry cannot be calculated from infinite or missing prices.
    if not has_only_finite_prices:
        raise ValueError("OHLC prices must be finite numbers.")

    # More than one candle at the same time would make the previous-close
    # reference ambiguous.
    if parsed_candlestick_data[timestamp_column].duplicated().any():
        raise ValueError("Candlestick data contains duplicate timestamps.")

    # Strict chronological order guarantees that every denominator comes only
    # from information available before the candle being normalized.
    if not parsed_candlestick_data[
        timestamp_column
    ].is_monotonic_increasing:
        raise ValueError("Candlestick timestamps must be in chronological order.")

    # A high below the open or close cannot describe valid OHLC geometry and
    # would give the model a physically inconsistent upper range.
    has_invalid_high = parsed_candlestick_data["high"] < (
        parsed_candlestick_data[["open", "close"]].max(axis="columns")
    )

    # A candle high must enclose both its opening and closing prices.
    if has_invalid_high.any():
        raise ValueError("High cannot be below the open or close.")

    # A low above the open or close would likewise create an impossible lower
    # candlestick range.
    has_invalid_low = parsed_candlestick_data["low"] > (
        parsed_candlestick_data[["open", "close"]].min(axis="columns")
    )

    # A candle low must enclose both its opening and closing prices.
    if has_invalid_low.any():
        raise ValueError("Low cannot be above the open or close.")

    # Shift closes down one row so each candle is paired only with the close
    # known immediately before its own timestamp.
    previous_closes = parsed_candlestick_data["close"].shift(1)

    # The first candle has no earlier row, so normalization begins with the
    # second candle and preserves the first close only as its causal reference.
    has_previous_close = previous_closes.notna()

    # A zero previous close cannot provide the shared scale required by any of
    # the four relative geometry calculations.
    if (previous_closes.loc[has_previous_close] == 0).any():
        raise ValueError("Previous closes cannot be zero.")

    # Isolate rows with a causal reference so every output feature has a valid
    # denominator and remains aligned with its original timestamp.
    current_candlestick_data = parsed_candlestick_data.loc[
        has_previous_close
    ]
    previous_closes = previous_closes.loc[has_previous_close]

    # Build one model-ready row per current candle while retaining timestamp and
    # instrument fields for session, annotation, and prediction alignment.
    normalized_candlestick_data = pd.DataFrame(
        {
            timestamp_column: current_candlestick_data[timestamp_column],
            "instrument": instrument,

            # Measure the opening move from the preceding close so the first
            # candle of a session preserves its overnight gap.
            "open_gap": (
                current_candlestick_data["open"] - previous_closes
            ) / previous_closes,

            # Measure the signed body on the same scale so positive and negative
            # values preserve bullish and bearish candle direction.
            "body": (
                current_candlestick_data["close"]
                - current_candlestick_data["open"]
            ) / previous_closes,

            # Measure the high above or below the current close to preserve the
            # candle's upper price geometry.
            "high_from_close": (
                current_candlestick_data["high"]
                - current_candlestick_data["close"]
            )
            / previous_closes,

            # Measure the low from the current close on the same scale to
            # preserve the candle's lower price geometry.
            "low_from_close": (
                current_candlestick_data["low"]
                - current_candlestick_data["close"]
            ) / previous_closes,
        }
    )

    # Replace inherited row labels with a continuous index while timestamps
    # continue to preserve exact links to the original candlesticks.
    return normalized_candlestick_data.reset_index(drop=True)
