"""Causal normalization for OHLC candlesticks."""

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
        ValueError: If the instruments differ, timestamps are out of order, or
            the previous close is zero.
    """
    # Require one instrument so both candlesticks share the same price scale.
    if previous_candlestick.instrument != current_candlestick.instrument:
        raise ValueError("Candlesticks must belong to the same instrument.")

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
