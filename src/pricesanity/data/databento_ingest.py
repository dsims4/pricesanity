"""Load OHLC data exported from Databento without modifying the source file."""

from pathlib import Path

import pandas as pd


OHLC_COLUMNS = ("open", "high", "low", "close")


def load_ohlc_csv(
    path: str | Path,
    timestamp_column: str = "ts_event",
    source_timezone: str = "UTC",
) -> pd.DataFrame:
    """Load and validate a chronological OHLC CSV."""
    csv_path = Path(path)
    required_columns = (timestamp_column, *OHLC_COLUMNS)

    header = pd.read_csv(csv_path, nrows=0)
    missing_columns = set(required_columns) - set(header.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"CSV is missing required columns: {missing}")

    frame = pd.read_csv(csv_path, usecols=list(required_columns))
    timestamps = pd.to_datetime(
        frame[timestamp_column],
        errors="coerce",
    )
    if timestamps.isna().any():
        raise ValueError("CSV contains invalid timestamps.")

    if timestamps.dt.tz is None:
        timestamps = timestamps.dt.tz_localize(source_timezone)

    frame[timestamp_column] = timestamps.dt.tz_convert("UTC")

    for column in OHLC_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    if frame[list(OHLC_COLUMNS)].isna().any().any():
        raise ValueError("CSV contains invalid OHLC values.")

    if frame[timestamp_column].duplicated().any():
        raise ValueError("CSV contains duplicate timestamps.")

    if not frame[timestamp_column].is_monotonic_increasing:
        raise ValueError("CSV timestamps must be in chronological order.")

    return frame
