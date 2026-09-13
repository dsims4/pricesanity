"""Typed structures for human price-action annotations."""

from dataclasses import dataclass
from enum import StrEnum


class MarketRegime(StrEnum):
    """Market structures available for each candlestick annotation."""

    BULL = "bull"
    BEAR = "bear"
    RANGE = "range"


@dataclass(frozen=True)
class CandlestickAnnotation:
    """Current and future regime targets aligned to one candlestick.

    Args:
        candlestick_id: Stable identifier connecting the annotation to price data.
        current_regime: Market regime after the current candle.
        anticipated_regime: Regime the current candle is most likely to produce.
    """

    candlestick_id: str
    current_regime: MarketRegime
    anticipated_regime: MarketRegime

    def __post_init__(self) -> None:
        """Validate the annotation fields.

        Raises:
            ValueError: If the candlestick identifier is empty.
        """

        # An annotation without a candle identifier could not be aligned
        # with its candlestick during training or chart replay.
        if not self.candlestick_id.strip():
            raise ValueError("Candlestick identifier cannot be empty.")
