"""Typed data structures for price-action candlesticks."""

from dataclasses import dataclass
from datetime import datetime, timezone
from math import isfinite


@dataclass(frozen=True)
class RawCandlestick:
    """One unnormalized candlestick loaded from source data."""

    timestamp: datetime
    instrument: str
    open: float
    high: float
    low: float
    close: float

    def __post_init__(self) -> None:
        """Validate timestamp and OHLC geometry."""
        if self.timestamp.tzinfo is None or self.timestamp.utcoffset() is None:
            raise ValueError("Raw candlestick timestamps must be timezone-aware.")

        if not self.instrument.strip():
            raise ValueError("Instrument cannot be empty.")

        prices = (self.open, self.high, self.low, self.close)
        if not all(isfinite(price) for price in prices):
            raise ValueError("OHLC prices must be finite numbers.")

        if self.high < max(self.open, self.close):
            raise ValueError("High cannot be below the open or close.")

        if self.low > min(self.open, self.close):
            raise ValueError("Low cannot be above the open or close.")

    @property
    def candlestick_id(self) -> str:
        """Return a deterministic identifier based on instrument and time."""
        utc_timestamp = self.timestamp.astimezone(timezone.utc)
        return f"{self.instrument}:{utc_timestamp.isoformat()}"


@dataclass(frozen=True)
class NormalizedCandlestick:
    """One candlestick represented as relative OHLC geometry."""

    timestamp: datetime
    instrument: str
    open_gap: float
    body: float
    high_from_close: float
    low_from_close: float

    def __post_init__(self) -> None:
        """Validate normalized candlestick metadata and features."""
        if self.timestamp.tzinfo is None or self.timestamp.utcoffset() is None:
            raise ValueError(
                "Normalized candlestick timestamps must be timezone-aware."
            )

        if not self.instrument.strip():
            raise ValueError("Instrument cannot be empty.")

        features = (
            self.open_gap,
            self.body,
            self.high_from_close,
            self.low_from_close,
        )
        if not all(isfinite(feature) for feature in features):
            raise ValueError("Normalized OHLC features must be finite numbers.")

    @property
    def candlestick_id(self) -> str:
        """Return the stable identifier used by the raw candlestick."""
        utc_timestamp = self.timestamp.astimezone(timezone.utc)
        return f"{self.instrument}:{utc_timestamp.isoformat()}"
