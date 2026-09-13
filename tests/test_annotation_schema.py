import pytest

from pricesanity.annotation.schema import (
    CandlestickAnnotation,
    MarketRegime,
)


def test_candlestick_annotation_preserves_model_targets() -> None:
    """Verify candlestick annotation preserves model targets."""

    # Record both regime targets against the same stable identifier used by raw
    # and normalized candlestick data.
    annotation = CandlestickAnnotation(
        candlestick_id="ES.v.0:2026-09-09T13:30:00+00:00",
        current_regime=MarketRegime.RANGE,
        anticipated_regime=MarketRegime.BEAR,
    )

    # Both supervised targets remain attached to the exact source candlestick.
    assert annotation.candlestick_id == "ES.v.0:2026-09-09T13:30:00+00:00"
    assert annotation.current_regime is MarketRegime.RANGE
    assert annotation.anticipated_regime is MarketRegime.BEAR


def test_candlestick_annotation_rejects_empty_identifier() -> None:
    """Verify candlestick annotation rejects empty identifier."""

    # Reject an annotation that could not later be matched to model inputs.
    # Reject unknown regime names so stored judgments always use the supported training
    # vocabulary.
    with pytest.raises(
        ValueError,
        match="Candlestick identifier cannot be empty",
    ):
        CandlestickAnnotation(
            candlestick_id=" ",
            current_regime=MarketRegime.BULL,
            anticipated_regime=MarketRegime.RANGE,
        )
