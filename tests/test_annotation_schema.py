import pytest

from pricesanity.annotation.schema import (
    CandlestickAnnotation,
    DirectionalOutlook,
    MarketRegime,
)


def test_candlestick_annotation_preserves_model_targets() -> None:
    # Record one causal interpretation against the same stable identifier used
    # by raw and normalized candlestick data.
    annotation = CandlestickAnnotation(
        candlestick_id="ES.v.0:2026-09-09T13:30:00+00:00",
        current_regime=MarketRegime.RANGE,
        directional_outlook=DirectionalOutlook.BEARISH,
    )

    # Both supervised targets remain attached to the exact candlestick that
    # was visible when the interpretation was made.
    assert (
        annotation.candlestick_id
        == "ES.v.0:2026-09-09T13:30:00+00:00"
    )
    assert annotation.current_regime is MarketRegime.RANGE
    assert annotation.directional_outlook is DirectionalOutlook.BEARISH


def test_candlestick_annotation_rejects_empty_identifier() -> None:
    # Reject an annotation that could not later be matched to model inputs.
    with pytest.raises(
        ValueError,
        match="Candlestick identifier cannot be empty",
    ):
        CandlestickAnnotation(
            candlestick_id=" ",
            current_regime=MarketRegime.BULL,
            directional_outlook=DirectionalOutlook.BULLISH,
        )
