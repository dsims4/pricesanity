"""Typed data structures for price-action candlesticks."""

from dataclasses import dataclass
from datetime import datetime
from math import isfinite

from pricesanity.data.identifiers import (
    DEFAULT_CANDLE_INTERVAL,
    build_candlestick_id,
    canonical_interval,
)


@dataclass(frozen=True)
class RawCandlestick:
    """One unnormalized candlestick loaded from source data."""

    timestamp: datetime
    instrument: str
    open: float
    high: float
    low: float
    close: float
    interval: str = DEFAULT_CANDLE_INTERVAL

    def __post_init__(self) -> None:
        """Validate raw candlestick fields.

        Raises:
            ValueError: If the timestamp, instrument, or OHLC prices are invalid.
        """

        # Canonicalize the interval during frozen-object initialization so equivalent durations
        # share an identifier.
        object.__setattr__(self, "interval", canonical_interval(self.interval))

        # Reject a timestamp that cannot identify one absolute moment in time.
        if self.timestamp.tzinfo is None or self.timestamp.utcoffset() is None:
            raise ValueError("Raw candlestick timestamps must be timezone-aware.")

        # Reject an empty or whitespace-only instrument name.
        if not self.instrument.strip():
            raise ValueError("Instrument cannot be empty.")

        # Collect OHLC prices into one tuple so one finite-value check covers all
        # four fields.
        ohlc_prices = (self.open, self.high, self.low, self.close)

        # Reject any OHLC price that is infinite or not a number.
        if not all(isfinite(price) for price in ohlc_prices):
            raise ValueError("OHLC prices must be finite numbers.")

        # Reject a high below the greater of the open and close.
        if self.high < max(self.open, self.close):
            raise ValueError("High cannot be below the open or close.")

        # Reject a low above the lesser of the open and close.
        if self.low > min(self.open, self.close):
            raise ValueError("Low cannot be above the open or close.")

    @property
    def candlestick_id(self) -> str:
        """Return the interval-qualified string identifier."""

        # Use the shared builder so raw and normalized representations identify exactly the same
        # candle.
        return build_candlestick_id(self.instrument, self.timestamp, self.interval)


@dataclass(frozen=True)
class NormalizedCandlestick:
    """One candlestick represented as relative OHLC geometry."""

    timestamp: datetime
    instrument: str
    open_gap: float
    body: float
    high_from_close: float
    low_from_close: float
    interval: str = DEFAULT_CANDLE_INTERVAL

    def __post_init__(self) -> None:
        """Validate normalized candlestick fields.

        Raises:
            ValueError: If the timestamp, instrument, or normalized features are invalid.
        """

        # Canonicalize the interval during frozen-object initialization so equivalent durations
        # share an identifier.
        object.__setattr__(self, "interval", canonical_interval(self.interval))

        # Reject a timestamp that cannot identify one absolute moment in time.
        if self.timestamp.tzinfo is None or self.timestamp.utcoffset() is None:
            raise ValueError(
                "Normalized candlestick timestamps must be timezone-aware."
            )

        # Reject an empty or whitespace-only instrument name.
        if not self.instrument.strip():
            raise ValueError("Instrument cannot be empty.")

        # Collect normalized ratios into one tuple so one finite-value check
        # covers all four model features.
        normalized_features = (
            self.open_gap,
            self.body,
            self.high_from_close,
            self.low_from_close,
        )

        # Reject any normalized ratio that is infinite or not a number.
        if not all(isfinite(feature) for feature in normalized_features):
            raise ValueError("Normalized OHLC features must be finite numbers.")

    @property
    def candlestick_id(self) -> str:
        """Return the interval-qualified string identifier."""

        # Use the shared builder so raw and normalized representations identify exactly the same
        # candle.
        return build_candlestick_id(self.instrument, self.timestamp, self.interval)
