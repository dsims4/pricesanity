"""Shared candle identity across typed candles, charts, and stored annotations."""

from datetime import datetime
from functools import lru_cache

import pandas as pd

# Match the default five-minute candle configuration.
DEFAULT_CANDLE_INTERVAL = "5min"


@lru_cache(maxsize=64)
def canonical_interval(interval: str) -> str:
    """Give equivalent fixed durations (such as 300s and 5min) the same key.

    Args:
        interval: Fixed candlestick duration used to qualify the identifier.

    Returns:
        Canonical duration string shared by equivalent fixed intervals.

    Raises:
        ValueError: If the interval is not a positive fixed duration.
    """

    # Normalize equivalent duration spellings before using the interval in a persistent
    # identifier.
    duration = pd.Timedelta(interval)

    # Only positive fixed intervals can distinguish usable candlestick durations.
    if pd.isna(duration) or duration <= pd.Timedelta(0):
        raise ValueError("Candle interval must be a positive fixed duration.")

    nanoseconds = duration.value

    # Choose the largest exact unit so equivalent intervals produce the same readable
    # identifier.
    for unit, scale in (
        ("min", 60_000_000_000),
        ("s", 1_000_000_000),
        ("ms", 1_000_000),
        ("us", 1_000),
        ("ns", 1),
    ):
        # Use the largest exact unit so equivalent durations share the same readable identifier.
        if nanoseconds % scale == 0:
            return f"{nanoseconds // scale}{unit}"

    raise AssertionError("Every duration has an integer nanosecond representation.")


def build_candlestick_id(
    instrument: str,
    timestamp: datetime,
    interval: str = DEFAULT_CANDLE_INTERVAL,
) -> str:
    """Identify an instrument, fixed candle interval, and absolute opening time.

    Args:
        instrument: Instrument symbol associated with the candlestick.
        timestamp: Timezone-aware opening timestamp of the candlestick.
        interval: Fixed candlestick duration used to qualify the identifier.

    Returns:
        Identifier combining instrument, interval, and absolute opening time.

    Raises:
        ValueError: If the instrument is empty or the timestamp does not identify an absolute
            moment.
    """

    # An empty instrument would let unrelated series share the same identifier.
    if not instrument.strip():
        raise ValueError("Instrument cannot be empty.")

    timestamp = pd.Timestamp(timestamp)

    # An identifier must represent one absolute opening time.
    if pd.isna(timestamp) or timestamp.tzinfo is None:
        raise ValueError("Candlestick timestamps must be valid and timezone-aware.")

    return (
        f"{instrument}:{canonical_interval(interval)}:"
        f"{timestamp.tz_convert('UTC').isoformat()}"
    )
