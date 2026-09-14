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
        """Initialize the owned annotation connection and schema.

        Args:
            database_path: SQLite file containing the human annotations.
        """

        # Resolve the database location once so repeated annotation transactions share one
        # connection.
        database_path = Path(database_path)
        database_path.parent.mkdir(parents=True, exist_ok=True)

        # A busy database should offer a prompt retry instead of freezing the GUI.
        self._connection = sqlite3.connect(database_path, timeout=0.25)

        # Close the newly opened connection if schema initialization cannot finish.
        try:
            # Create the table within a transaction so a failed initialization cannot partially
            # commit.
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

        # A constructor that fails cannot leave its connection for the caller to close.
        except Exception:
            self.close()
            raise

    def save(self, annotation: CandlestickAnnotation) -> None:
        """Insert or replace one complete judgment in a transaction.

        Args:
            annotation: Complete current and anticipated regime judgment to persist.
        """

        # Commit the entire annotation together, or roll it back if SQLite rejects the write.
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
        """Load one judgment through the indexed candle identifier.

        Args:
            candlestick_id: Interval-qualified identifier of the requested candlestick.

        Returns:
            Saved judgment, or None when the identifier has not been annotated.
        """

        # Reuse the open connection so navigating candles does not repeatedly initialize SQLite.
        return _load_annotation(self._connection, candlestick_id)

    def load_annotated_ids(self) -> set[str]:
        """Read saved candle identifiers for one navigation search.

        Returns:
            Identifiers with persisted judgments, including other date ranges.
        """

        # Read once per seek so long annotated stretches do not require one SQL query per
        # candle. A fresh snapshot also includes judgments saved by another open window.
        return {
            row[0]
            for row in self._connection.execute(
                "SELECT candlestick_id FROM annotations"
            )
        }

    def close(self) -> None:
        """Release the connection; calling this more than once is harmless."""

        # Release the connection when its owning store is finished; no worker thread owns it
        # separately.
        self._connection.close()


def _load_annotation(
    connection: sqlite3.Connection, candlestick_id: str
) -> CandlestickAnnotation | None:
    """Read one complete judgment using the supplied connection.

    Args:
        connection: Existing SQLite connection owned by the caller.
        candlestick_id: Interval-qualified identifier of the requested candlestick.

    Returns:
        Saved judgment, or None when the identifier has not been annotated.

    Raises:
        ValueError: If the candlestick identifier is empty or a stored label is invalid.
    """

    # Reject an empty identifier before looking up a judgment that cannot name a candle.
    if not candlestick_id.strip():
        raise ValueError("Candlestick identifier cannot be empty.")

    row = connection.execute(
        """
        SELECT current_regime, anticipated_regime
        FROM annotations WHERE candlestick_id = ?
        """,
        (candlestick_id,),
    ).fetchone()

    # An unannotated candle has no judgment to reconstruct.
    if row is None:
        return None

    return CandlestickAnnotation(
        candlestick_id=candlestick_id,
        current_regime=MarketRegime(row[0]),
        anticipated_regime=MarketRegime(row[1]),
    )


def initialize_annotation_store(database_path: str | Path) -> None:
    """Create the schema and immediately release the connection.

    Args:
        database_path: SQLite file containing the human annotations.
    """

    # This one-shot helper owns its connection and must release it after initialization.
    with closing(AnnotationStore(database_path)):
        pass


def save_annotation(database_path: str | Path, annotation: CandlestickAnnotation) -> None:
    """Save a judgment for callers that do not own a long-lived store.

    Args:
        database_path: SQLite file containing the human annotations.
        annotation: Complete current and anticipated regime judgment to persist.
    """

    # Keep the reusable store API available to callers that only need a single write.
    with closing(AnnotationStore(database_path)) as store:
        store.save(annotation)


def load_annotation(
    database_path: str | Path,
    candlestick_id: str,
) -> CandlestickAnnotation | None:
    """Read a judgment without creating a database when none exists.

    Args:
        database_path: SQLite file containing the human annotations.
        candlestick_id: Interval-qualified identifier of the requested candlestick.

    Returns:
        Saved judgment, or None when the database or annotation does not exist.
    """

    # An empty identifier cannot refer to a valid saved annotation.
    if not candlestick_id.strip():
        raise ValueError("Candlestick identifier cannot be empty.")

    # Resolve the database location once so repeated annotation transactions share one
    # connection.
    database_path = Path(database_path)

    # Reading an absent store must not create a new empty database as a side effect.
    if not database_path.exists():
        return None

    # SQLite's transaction context manager does not close its connection.
    with closing(sqlite3.connect(database_path)) as connection:
        return _load_annotation(connection, candlestick_id)
