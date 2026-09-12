"""Persist candlestick annotations without altering market data."""

from contextlib import closing
import sqlite3
from pathlib import Path

from pricesanity.annotation.schema import CandlestickAnnotation, MarketRegime


class AnnotationStore:
    """One explicitly owned SQLite connection, used on its creating thread.

    Close the store when its window or batch operation ends. Each save commits
    both labels together; a failed transaction rolls back before propagating.
    """

    def __init__(self, database_path: str | Path) -> None:
        database_path = Path(database_path)
        database_path.parent.mkdir(parents=True, exist_ok=True)
        # A busy database should offer a prompt retry instead of freezing the GUI.
        self._connection = sqlite3.connect(database_path, timeout=0.25)
        try:
            with self._connection:
                self._connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS annotations (
                        candlestick_id TEXT PRIMARY KEY,
                        current_regime TEXT NOT NULL,
                        anticipated_regime TEXT NOT NULL,
                        updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                    )
                    """
                )
        except Exception:
            self.close()
            raise

    def save(self, annotation: CandlestickAnnotation) -> None:
        """Insert or replace one complete judgment in a transaction."""
        with self._connection:
            self._connection.execute(
                """
                INSERT INTO annotations (
                    candlestick_id, current_regime, anticipated_regime
                ) VALUES (?, ?, ?)
                ON CONFLICT(candlestick_id) DO UPDATE SET
                    current_regime = excluded.current_regime,
                    anticipated_regime = excluded.anticipated_regime,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (
                    annotation.candlestick_id,
                    annotation.current_regime.value,
                    annotation.anticipated_regime.value,
                ),
            )

    def load(self, candlestick_id: str) -> CandlestickAnnotation | None:
        """Load one judgment through the indexed candle identifier."""
        return _load_annotation(self._connection, candlestick_id)

    def close(self) -> None:
        """Release the connection; calling this more than once is harmless."""
        self._connection.close()


def _load_annotation(
    connection: sqlite3.Connection, candlestick_id: str
) -> CandlestickAnnotation | None:
    if not candlestick_id.strip():
        raise ValueError("Candlestick identifier cannot be empty.")
    row = connection.execute(
        """
        SELECT current_regime, anticipated_regime
        FROM annotations WHERE candlestick_id = ?
        """,
        (candlestick_id,),
    ).fetchone()
    if row is None:
        return None
    return CandlestickAnnotation(
        candlestick_id=candlestick_id,
        current_regime=MarketRegime(row[0]),
        anticipated_regime=MarketRegime(row[1]),
    )


def initialize_annotation_store(database_path: str | Path) -> None:
    """Create the schema and immediately release the connection."""
    with closing(AnnotationStore(database_path)):
        pass


def save_annotation(
    database_path: str | Path, annotation: CandlestickAnnotation
) -> None:
    """Save a judgment for callers that do not own a long-lived store."""
    with closing(AnnotationStore(database_path)) as store:
        store.save(annotation)


def load_annotation(
    database_path: str | Path, candlestick_id: str
) -> CandlestickAnnotation | None:
    """Read a judgment without creating a database when none exists."""
    if not candlestick_id.strip():
        raise ValueError("Candlestick identifier cannot be empty.")
    database_path = Path(database_path)
    if not database_path.exists():
        return None
    # SQLite's transaction context manager does not close its connection.
    with closing(sqlite3.connect(database_path)) as connection:
        return _load_annotation(connection, candlestick_id)
