"""Causal normalization for OHLC candlesticks."""

from pricesanity.data.schemas import (
    NormalizedCandlestick,
    RawCandlestick,
)


def normalize_candlestick(
        previous: RawCandlestick,
        current: RawCandlestick,
) -> NormalizedCandlestick:
    """Normalize one candlestick relative to the previous' close."""
    if previous.instrument != current.instrument:
        raise ValueError("Candlesticks must belong to the same instrument.")

    if previous.timestamp >= current.timestamp:
        raise ValueError(
            "The previous candlestick must occur before the current one."
        )

    previous_close = previous.close
    if previous_close == 0:
        raise ValueError("The previous close cannot be zero.")

    return NormalizedCandlestick(
        timestamp=current.timestamp,
        instrument=current.instrument,
        open_gap=(current.open - previous_close) / previous_close,
        body=(current.close - current.open) / previous_close,
        high_from_close=(current.high - current.close) / previous_close,
        low_from_close=(current.low - current.close) / previous_close,
    )
