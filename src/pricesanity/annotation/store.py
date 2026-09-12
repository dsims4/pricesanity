"""Persist candlestick annotations without altering market data."""

import sqlite3
from pathlib import Path

from pricesanity.annotation.schema import (
    CandlestickAnnotation,
    DirectionalOutlook,
    MarketRegime,
)


def initialize_annotation_store(database_path: str | Path) -> None:
    """Create the annotation database and its annotation table.

    Args:
        database_path: Path to the SQLite annotation database.
    """
    # Keep annotations beneath their own directory so creating the store never
    # requires changing or placing files beside immutable market data.
    database_path = Path(database_path)
    database_path.parent.mkdir(parents=True, exist_ok=True)

    # SQLite keeps the current annotation for every candlestick in one local
    # file without placing labels inside immutable market data.
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS annotations (
                candlestick_id TEXT PRIMARY KEY,
                current_regime TEXT NOT NULL,
                directional_outlook TEXT NOT NULL,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )


def save_annotation(
    database_path: str | Path,
    annotation: CandlestickAnnotation,
) -> None:
    """Save the current annotation for one candlestick.

    Args:
        database_path: Path to the SQLite annotation database.
        annotation: Causal interpretation to save.
    """
    # Ensure the store exists so the first save is as safe as every later edit.
    initialize_annotation_store(database_path)

    # Insert a new candlestick or replace its two labels so the table contains
    # exactly one current annotation for each stable identifier.
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            """
            INSERT INTO annotations (
                candlestick_id,
                current_regime,
                directional_outlook
            )
            VALUES (?, ?, ?)
            ON CONFLICT(candlestick_id) DO UPDATE SET
                current_regime = excluded.current_regime,
                directional_outlook = excluded.directional_outlook,
                updated_at = CURRENT_TIMESTAMP
            """,
            (
                annotation.candlestick_id,
                annotation.current_regime.value,
                annotation.directional_outlook.value,
            ),
        )


def load_annotation(
    database_path: str | Path,
    candlestick_id: str,
) -> CandlestickAnnotation | None:
    """Load the annotation for one candlestick.

    Args:
        database_path: Path to the SQLite annotation database.
        candlestick_id: Stable identifier of the requested candlestick.

    Returns:
        The saved annotation, or none if the candle is not annotated.

    Raises:
        ValueError: If the candlestick identifier is empty.
    """
    # Reject an unusable lookup for the same reason the annotation schema
    # rejects an empty identifier when saving.
    if not candlestick_id.strip():
        raise ValueError("Candlestick identifier cannot be empty.")

    # A missing database means no annotations exist yet, which is a normal
    # state when a newly prepared dataset is opened for the first time.
    database_path = Path(database_path)
    if not database_path.exists():
        return None

    # The candlestick identifier is the table's primary key, so at most one
    # current annotation can match this lookup.
    with sqlite3.connect(database_path) as connection:
        stored_annotation = connection.execute(
            """
            SELECT current_regime, directional_outlook
            FROM annotations
            WHERE candlestick_id = ?
            """,
            (candlestick_id,),
        ).fetchone()

    # No matching row means this particular candle remains unannotated.
    if stored_annotation is None:
        return None

    # Convert stored strings back into typed model targets before returning
    # them to the GUI or dataset builder.
    current_regime, directional_outlook = stored_annotation
    return CandlestickAnnotation(
        candlestick_id=candlestick_id,
        current_regime=MarketRegime(current_regime),
        directional_outlook=DirectionalOutlook(directional_outlook),
    )
