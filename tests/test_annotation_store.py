import sqlite3

import pytest

from pricesanity.annotation.schema import (
    CandlestickAnnotation,
    MarketRegime,
)
from pricesanity.annotation.store import (
    load_annotation,
    save_annotation,
)


def test_annotation_store_updates_existing_candlestick(tmp_path) -> None:
    """Verify annotation store updates existing candlestick."""

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
            anticipated_regime=MarketRegime.BEAR,
        ),
    )
    corrected_annotation = CandlestickAnnotation(
        candlestick_id=candlestick_id,
        current_regime=MarketRegime.BEAR,
        anticipated_regime=MarketRegime.RANGE,
    )
    save_annotation(database_path, corrected_annotation)

    # Loading returns the corrected judgment used for training and replay.
    assert load_annotation(database_path, candlestick_id) == (corrected_annotation)

    # Updating a candlestick retains one current row instead of accumulating
    # revision history.
    with sqlite3.connect(database_path) as connection:
        annotation_count = connection.execute("SELECT COUNT(*) FROM annotations").fetchone()

    assert annotation_count == (1,)


def test_annotation_store_returns_none_for_unannotated_candle(
    tmp_path,
) -> None:
    """Verify annotation store returns none for unannotated candle."""

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
    """Verify annotation store rejects empty identifier."""

    # Refuse a lookup that cannot identify any candlestick in the dataset.
    # An empty identifier cannot identify a candle and must fail before any annotation is
    # written.
    with pytest.raises(
        ValueError,
        match="Candlestick identifier cannot be empty",
    ):
        load_annotation(tmp_path / "annotations.sqlite3", " ")


def test_store_reuses_connection_and_closes_explicitly(tmp_path, monkeypatch) -> None:
    """Verify store reuses connection and closes explicitly."""

    from pricesanity.annotation.store import AnnotationStore

    original_connect = sqlite3.connect
    connections = []

    def connect(*args, **kwargs):
        """Count opened SQLite connections while retaining real transaction behavior."""

        connection = original_connect(*args, **kwargs)
        connections.append(connection)

        return connection

    monkeypatch.setattr(sqlite3, "connect", connect)
    store = AnnotationStore(tmp_path / "annotations.db")
    annotation = CandlestickAnnotation("candle", MarketRegime.BULL, MarketRegime.BEAR)

    # Repeat writes through the same store to prove routine annotation does not reopen SQLite.
    for _ in range(3):
        store.save(annotation)

        assert store.load("candle") == annotation

    assert len(connections) == 1
    store.close()
    store.close()

    # Closing the store must release the retained connection rather than merely hiding the
    # wrapper.
    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        connections[0].execute("SELECT 1")


def test_failed_transaction_preserves_previous_annotation(tmp_path) -> None:
    """Verify failed transaction preserves previous annotation."""

    from contextlib import closing
    from pricesanity.annotation.store import AnnotationStore

    path = tmp_path / "annotations.db"

    # Keep one annotation store alive while a separate SQLite writer deliberately holds
    # contention.
    with closing(AnnotationStore(path)) as store:
        original = CandlestickAnnotation("candle", MarketRegime.BULL, MarketRegime.BEAR)
        store.save(original)

        # A second connection supplies real database contention instead of a mocked save
        # exception.
        with closing(sqlite3.connect(path)) as blocker:
            blocker.execute("BEGIN EXCLUSIVE")

            # A blocked save must fail clearly so it cannot be mistaken for a committed human
            # judgment.
            with pytest.raises(sqlite3.OperationalError, match="locked"):
                store.save(CandlestickAnnotation("candle", MarketRegime.RANGE, MarketRegime.RANGE))

            blocker.rollback()

        assert store.load("candle") == original
        store.save(CandlestickAnnotation("next", MarketRegime.BEAR, MarketRegime.BULL))

        assert store.load("next") is not None


def test_annotation_store_loads_all_annotations(tmp_path) -> None:
    """The training boundary can retrieve every complete judgement."""

    from contextlib import closing

    from pricesanity.annotation.store import AnnotationStore

    database_path = tmp_path / "annotations.sqlite3"
    annotations = [
        CandlestickAnnotation(
            "candle-1",
            MarketRegime.BULL,
            MarketRegime.RANGE,
        ),
        CandlestickAnnotation(
            "candle-2",
            MarketRegime.BEAR,
            MarketRegime.BULL,
        ),
    ]

    # Save through the production interface before requesting the training collection.
    with closing(AnnotationStore(database_path)) as store:
        for annotation in annotations:
            store.save(annotation)

        loaded_annotations = store.load_all()

    assert set(loaded_annotations) == set(annotations)


def test_annotation_store_loads_only_requested_saved_annotations(tmp_path) -> None:
    """A chart redraw can load one session without inventing unsaved labels."""

    from contextlib import closing

    from pricesanity.annotation.store import AnnotationStore

    database_path = tmp_path / "annotations.sqlite3"
    first_annotation = CandlestickAnnotation(
        "candle-1",
        MarketRegime.BULL,
        MarketRegime.RANGE,
    )
    later_annotation = CandlestickAnnotation(
        "candle-3",
        MarketRegime.BEAR,
        MarketRegime.BULL,
    )

    with closing(AnnotationStore(database_path)) as store:
        store.save(first_annotation)
        store.save(later_annotation)

        # The absent middle candle stays absent so the chart can leave a visible gap.
        loaded_annotations = store.load_many(
            ["candle-1", "candle-2", "candle-3"],
        )

        assert loaded_annotations == {
            "candle-1": first_annotation,
            "candle-3": later_annotation,
        }
        assert store.load_many([]) == {}
