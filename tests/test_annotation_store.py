import sqlite3

import pytest

from pricesanity.annotation.schema import (
    CandlestickAnnotation,
    DirectionalOutlook,
    MarketRegime,
)
from pricesanity.annotation.store import (
    load_annotation,
    save_annotation,
)


def test_annotation_store_updates_existing_candlestick(tmp_path) -> None:
    # Use an isolated database so the test never reads or writes real human
    # annotations.
    database_path = tmp_path / "annotations.sqlite3"
    candlestick_id = "ES.v.0:2026-09-09T13:30:00+00:00"

    # Save an initial judgment before correcting the same candlestick.
    save_annotation(
        database_path,
        CandlestickAnnotation(
            candlestick_id=candlestick_id,
            current_regime=MarketRegime.RANGE,
            directional_outlook=DirectionalOutlook.BEARISH,
        ),
    )
    corrected_annotation = CandlestickAnnotation(
        candlestick_id=candlestick_id,
        current_regime=MarketRegime.BEAR,
        directional_outlook=DirectionalOutlook.BEARISH,
    )
    save_annotation(database_path, corrected_annotation)

    # Loading returns the corrected judgment used for training and replay.
    assert load_annotation(database_path, candlestick_id) == (
        corrected_annotation
    )

    # Updating a candlestick retains one current row instead of accumulating
    # revision history.
    with sqlite3.connect(database_path) as connection:
        annotation_count = connection.execute(
            "SELECT COUNT(*) FROM annotations"
        ).fetchone()
    assert annotation_count == (1,)


def test_annotation_store_returns_none_for_unannotated_candle(
    tmp_path,
) -> None:
    # A missing database is equivalent to a valid store with no matching
    # annotation because neither contains a human judgment for this candle.
    assert (
        load_annotation(
            tmp_path / "missing.sqlite3",
            "ES.v.0:2026-09-09T13:35:00+00:00",
        )
        is None
    )


def test_annotation_store_rejects_empty_identifier(tmp_path) -> None:
    # Refuse a lookup that cannot identify any candlestick in the dataset.
    with pytest.raises(
        ValueError,
        match="Candlestick identifier cannot be empty",
    ):
        load_annotation(tmp_path / "annotations.sqlite3", " ")
