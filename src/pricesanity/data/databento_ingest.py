"""Load OHLC data exported from Databento without modifying the source file."""

from pathlib import Path

import pandas as pd


# Exclude volume and vendor metadata because later price-action processing uses
# only the four values that define OHLC candlestick geometry.
REQUIRED_OHLC_COLUMNS = ("open", "high", "low", "close")


def load_ohlc_csv(
    csv_path: str | Path,
    timestamp_column: str = "ts_event",
    source_timezone: str = "UTC",
) -> pd.DataFrame:
    """Load and validate OHLC candlestick data from a CSV file.

    Args:
        csv_path: Path to the CSV file.
        timestamp_column: Column containing candlestick timestamps.
        source_timezone: Timezone used when timestamps lack one.

    Returns:
        Candlestick data with UTC timestamps and only the required columns.

    Raises:
        ValueError: If required columns, timestamps, prices, or row order are
            invalid.
    """
    # Treat string and Path inputs alike so validation and loading use one path
    # representation.
    csv_path = Path(csv_path)

    # The timestamp places each candle in sequence, while all four prices are
    # needed to preserve its complete OHLC shape.
    required_csv_columns = (timestamp_column, *REQUIRED_OHLC_COLUMNS)

    # Read only the header first so a large source file is not loaded when its
    # columns cannot support the rest of the pipeline.
    csv_header = pd.read_csv(csv_path, nrows=0)

    # Find every absent field at once so the source schema can be corrected
    # before any candlestick data is processed.
    missing_required_columns = set(required_csv_columns) - set(
        csv_header.columns
    )

    # Without every required field, the file cannot identify and reconstruct
    # complete OHLC candlesticks.
    if missing_required_columns:
        # Include every missing field in one stable message so repeated runs are
        # not needed to discover separate schema problems.
        missing_column_names = ", ".join(
            sorted(missing_required_columns)
        )
        raise ValueError(
            f"CSV is missing required columns: {missing_column_names}"
        )

    # Load only timestamp and OHLC data because volume and other vendor fields
    # are intentionally outside this model's price-action representation.
    candlestick_data = pd.read_csv(
        csv_path,
        usecols=list(required_csv_columns),
    )

    # Convert every timestamp together and mark parsing failures as NaT so one
    # check can reject all unusable time values.
    parsed_timestamps = pd.to_datetime(
        candlestick_data[timestamp_column],
        errors="coerce",
    )

    # A candle without a valid timestamp cannot be ordered, assigned to a
    # session, or protected from future-data leakage.
    if parsed_timestamps.isna().any():
        raise ValueError("CSV contains invalid timestamps.")

    # If the file omits timezone information, apply the configured source
    # timezone before treating each timestamp as an absolute moment.
    if parsed_timestamps.dt.tz is None:
        parsed_timestamps = parsed_timestamps.dt.tz_localize(source_timezone)

    # Store all timestamps in UTC so ordering, resampling, and session matching
    # compare one shared time reference.
    candlestick_data[timestamp_column] = parsed_timestamps.dt.tz_convert(
        "UTC"
    )

    # Convert every OHLC field to numbers and mark failures as NaN so malformed
    # prices cannot silently enter normalization.
    for column in REQUIRED_OHLC_COLUMNS:
        candlestick_data[column] = pd.to_numeric(
            candlestick_data[column],
            errors="coerce",
        )

    # Every candle needs four valid prices to preserve its body and range.
    if candlestick_data[list(REQUIRED_OHLC_COLUMNS)].isna().any().any():
        raise ValueError("CSV contains invalid OHLC values.")

    # More than one candle at the same timestamp would make the source interval
    # and later resampling groups ambiguous.
    if candlestick_data[timestamp_column].duplicated().any():
        raise ValueError("CSV contains duplicate timestamps.")

    # Chronological input is required so normalization and later model features
    # cannot accidentally use future prices.
    if not candlestick_data[timestamp_column].is_monotonic_increasing:
        raise ValueError("CSV timestamps must be in chronological order.")

    # Return the isolated, validated fields without changing the source file.
    return candlestick_data
