"""Load Databento exports without modifying their source files."""

import json
from collections.abc import Mapping
from pathlib import Path

import pandas as pd


# Exclude volume and vendor metadata because later price-action processing uses
# only the four values that define OHLC candlestick geometry.
REQUIRED_OHLC_COLUMNS = ("open", "high", "low", "close")

# Scheduled status extraction needs these three vendor fields to distinguish
# ordinary RTH boundaries from unrelated market-state records.
REQUIRED_STATUS_COLUMNS = ("reason", "trading_event", "is_trading")

# Daily condition metadata connects each trading date with Databento's quality
# assessment so degraded or unavailable sessions can be rejected later.
REQUIRED_CONDITION_COLUMNS = ("date", "condition")


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


def load_status_csv(
    csv_path: str | Path,
    timestamp_column: str = "ts_event",
) -> pd.DataFrame:
    """Load the Databento fields needed to extract session transitions.

    Args:
        csv_path: Path to the status CSV file.
        timestamp_column: Column containing status timestamps.

    Returns:
        Status timestamps and the fields needed for transition extraction.

    Raises:
        ValueError: If the CSV is missing a required field.
    """
    csv_path = Path(csv_path)

    # A transition needs its timestamp, cause, event, and resulting trading
    # state before later logic can decide whether it marks an RTH boundary.
    required_csv_columns = (timestamp_column, *REQUIRED_STATUS_COLUMNS)

    # Read only the header first so an unsuitable status export is rejected
    # before its full contents consume memory.
    csv_header = pd.read_csv(csv_path, nrows=0)

    # Report every absent field together so the correct Databento export can be
    # supplied without discovering schema problems one at a time.
    missing_required_columns = set(required_csv_columns) - set(
        csv_header.columns
    )

    # Missing context could cause an unrelated or unscheduled state change to
    # be mistaken for a valid session boundary.
    if missing_required_columns:
        # Sort the names to keep the error stable across repeated runs.
        missing_column_names = ", ".join(
            sorted(missing_required_columns)
        )
        raise ValueError(
            f"Status CSV is missing required columns: {missing_column_names}"
        )

    # Load only the fields used by transition extraction because symbol and
    # delivery metadata do not help determine session opening or closing times.
    status_data = pd.read_csv(
        csv_path,
        usecols=list(required_csv_columns),
    )

    # Preserve Databento's raw values here because the status transformation is
    # responsible for parsing timestamps and normalizing vendor state labels.
    return status_data


def load_dataset_conditions_json(json_path: str | Path) -> pd.DataFrame:
    """Load Databento's daily dataset conditions from a JSON file.

    Args:
        json_path: Path to the condition JSON file.

    Returns:
        Trading dates and their Databento data conditions.

    Raises:
        ValueError: If the JSON structure or required fields are invalid.
    """
    json_path = Path(json_path)

    # Decode the file with the standard library because the batch condition
    # artifact is a small collection of metadata records, not tabular prices.
    with json_path.open(encoding="utf-8") as condition_file:
        condition_records = json.load(condition_file)

    # Databento returns one mapping per date, so any other top-level structure
    # cannot establish an ordered collection of daily quality decisions.
    if not isinstance(condition_records, list):
        raise ValueError("Condition JSON must contain a list of records.")

    # An empty date range is valid and still needs stable columns so downstream
    # schedule validation can handle it predictably.
    if not condition_records:
        return pd.DataFrame(columns=list(REQUIRED_CONDITION_COLUMNS))

    # Validate each item before creating a table because a scalar or nested list
    # would otherwise produce unclear columns or missing quality information.
    if not all(isinstance(record, Mapping) for record in condition_records):
        raise ValueError("Each condition record must be a JSON object.")

    # Convert the records together so their common fields can be validated and
    # selected with the same tabular operations used by the rest of the pipeline.
    condition_data = pd.DataFrame(condition_records)

    # Both fields are needed to attach one quality decision to a trading date.
    missing_required_columns = set(REQUIRED_CONDITION_COLUMNS) - set(
        condition_data.columns
    )

    # Without a date or condition, later filtering could accidentally accept a
    # session whose data quality Databento did not confirm.
    if missing_required_columns:
        # Sort the names to keep the error stable across repeated runs.
        missing_column_names = ", ".join(
            sorted(missing_required_columns)
        )
        raise ValueError(
            f"Condition JSON is missing required fields: "
            f"{missing_column_names}"
        )

    # Keep only pipeline inputs because last-modified and other delivery
    # metadata do not affect session boundaries or the quality decision.
    return condition_data.loc[:, list(REQUIRED_CONDITION_COLUMNS)].copy()
