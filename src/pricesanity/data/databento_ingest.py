"""Load OHLC data exported from Databento without modifying the source file."""

from pathlib import Path

import pandas as pd


# List the four vendor price columns required by later candle processing.
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
    # Convert the location to a Path so string and Path arguments use the same
    # file operations.
    csv_path = Path(csv_path)

    # Combine the timestamp and OHLC names for header validation and selective
    # CSV loading.
    required_csv_columns = (timestamp_column, *REQUIRED_OHLC_COLUMNS)

    # Read zero data rows so an invalid header can be rejected before loading
    # the full dataset.
    csv_header = pd.read_csv(csv_path, nrows=0)

    # Compare required names with the header to collect every missing column.
    missing_required_columns = set(required_csv_columns) - set(
        csv_header.columns
    )

    # Reject the CSV before loading its rows when required columns are missing.
    if missing_required_columns:
        # Sort and join the missing names so the error is stable and readable.
        missing_column_names = ", ".join(
            sorted(missing_required_columns)
        )
        raise ValueError(
            f"CSV is missing required columns: {missing_column_names}"
        )

    # Load only columns used later, leaving volume and other vendor data out.
    candlestick_data = pd.read_csv(
        csv_path,
        usecols=list(required_csv_columns),
    )

    # Parse timestamps and replace failures with NaT for one validation check.
    parsed_timestamps = pd.to_datetime(
        candlestick_data[timestamp_column],
        errors="coerce",
    )

    # Reject the CSV when at least one parsed timestamp is NaT.
    if parsed_timestamps.isna().any():
        raise ValueError("CSV contains invalid timestamps.")

    # Attach the source timezone so naive timestamps represent absolute moments
    # before UTC conversion.
    if parsed_timestamps.dt.tz is None:
        parsed_timestamps = parsed_timestamps.dt.tz_localize(source_timezone)

    # Store all timestamps in UTC so later ordering and session conversions use
    # one shared reference timezone.
    candlestick_data[timestamp_column] = parsed_timestamps.dt.tz_convert(
        "UTC"
    )

    # Convert OHLC values to numbers and replace failures with NaN for one
    # validation check.
    for column in REQUIRED_OHLC_COLUMNS:
        candlestick_data[column] = pd.to_numeric(
            candlestick_data[column],
            errors="coerce",
        )

    # Reject the CSV when any converted OHLC value is NaN.
    if candlestick_data[list(REQUIRED_OHLC_COLUMNS)].isna().any().any():
        raise ValueError("CSV contains invalid OHLC values.")

    # Reject duplicate timestamps so each source interval has one candlestick.
    if candlestick_data[timestamp_column].duplicated().any():
        raise ValueError("CSV contains duplicate timestamps.")

    # Reject unordered rows so later causal operations cannot use future data.
    if not candlestick_data[timestamp_column].is_monotonic_increasing:
        raise ValueError("CSV timestamps must be in chronological order.")

    # Return the validated timestamp and OHLC columns as candlestick data.
    return candlestick_data
