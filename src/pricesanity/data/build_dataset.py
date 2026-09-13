"""Build a model-ready candlestick dataset from Databento export files."""

import argparse
from collections.abc import Sequence
from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from pricesanity.config import load_config
from pricesanity.data.databento_ingest import (
    load_dataset_conditions_json,
    load_status_csv,
)
from pricesanity.data.pipeline import prepare_csv_session_tables


def build_dataset_from_exports(
    candlestick_csv_path: str | Path,
    status_csv_path: str | Path,
    condition_json_path: str | Path,
    output_path: str | Path,
    *,
    config_path: str | Path,
    interim_output_path: str | Path,
    overwrite: bool = False,
    csv_chunk_rows: int = 100_000,
) -> pd.DataFrame:
    """Prepare and save model-ready data from Databento export files.

    Args:
        candlestick_csv_path: Path to the OHLC CSV file.
        status_csv_path: Path to the status CSV file.
        condition_json_path: Path to the daily condition JSON file.
        output_path: Path for the processed Parquet file.
        config_path: Path to the project configuration file.
        interim_output_path: Path for validated 5-minute OHLC data.
        overwrite: Whether an existing output file may be replaced.
        csv_chunk_rows: Maximum source rows per pandas read.

    Returns:
        The model-ready candlestick data saved to the output file.

    Raises:
        FileExistsError: If the output exists and overwrite is false.
        ValueError: If the output path is invalid or matches an input path.
    """

    # Convert every path once so validation, loading, and saving all refer to
    # the same filesystem representations.
    candlestick_csv_path = Path(candlestick_csv_path)
    status_csv_path = Path(status_csv_path)
    condition_json_path = Path(condition_json_path)
    output_path = Path(output_path)
    config_path = Path(config_path)
    interim_output_path = Path(interim_output_path)

    # Parquet preserves timestamp and numeric types, preventing the processed
    # dataset from losing schema information during a CSV round trip.
    if output_path.suffix.lower() != ".parquet":
        raise ValueError("The processed output path must end in .parquet.")

    # The chart-ready OHLC table also uses Parquet so timestamps and prices
    # retain their types without additional parsing in the annotation GUI.
    if interim_output_path.suffix.lower() != ".parquet":
        raise ValueError("The interim output path must end in .parquet.")

    # Resolve paths before comparison so alternate spellings of the same file
    # cannot allow processed data to replace a raw input export.
    resolved_output_path = output_path.resolve()
    resolved_input_paths = {
        candlestick_csv_path.resolve(),
        status_csv_path.resolve(),
        condition_json_path.resolve(),
        config_path.resolve(),
    }

    # Resolve the OHLC destination so it can be checked against every source
    # and the separate normalized output.
    resolved_interim_output_path = interim_output_path.resolve()

    # Raw data and configuration remain immutable because reproducing or
    # auditing a dataset requires their original contents.
    if resolved_output_path in resolved_input_paths:
        raise ValueError("The output path cannot match an input path.")

    # Keep the two derived representations in distinct files and prevent the
    # interim artifact from replacing any raw input or configuration file.
    if resolved_interim_output_path in resolved_input_paths:
        raise ValueError("The interim output path cannot match an input path.")

    # Raw geometry and normalized features must not overwrite the same destination.
    if resolved_interim_output_path == resolved_output_path:
        raise ValueError("The interim and processed outputs must differ.")

    # Require deliberate permission before replacing a processed artifact so a
    # previous reproducible dataset is not erased by an accidental rerun.
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Output already exists: {output_path}. Use --overwrite to replace it."
        )

    # Apply the same overwrite protection to chart-ready OHLC data so one run
    # cannot silently replace only half of an aligned artifact pair.
    if interim_output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Output already exists: {interim_output_path}. "
            "Use --overwrite to replace it."
        )

    # Load the typed settings first because the timestamp field, source
    # timezone, intervals, and session boundaries control every later stage.
    config = load_config(config_path)

    # Preserve the raw scheduled-state evidence needed to discover each date's
    # official session close, including early closes.
    status_data = load_status_csv(
        status_csv_path,
        timestamp_column=config.data.timestamp_column,
    )

    # Load daily quality decisions separately because an accurate schedule does
    # not prove that its underlying candlestick data is complete or trustworthy.
    data_conditions = load_dataset_conditions_json(condition_json_path)

    # Apply the established filtering, resampling, schedule validation, and
    # causal price normalization in one shared pipeline.
    prepared_sessions = prepare_csv_session_tables(
        candlestick_csv_path,
        status_data,
        data_conditions,
        config=config,
        csv_chunk_rows=csv_chunk_rows,
    )

    # Stage on each destination's filesystem so its final replacement is atomic.
    # Both writes must finish before either existing artifact is touched.
    artifacts = (
        (prepared_sessions.normalized, output_path),
        (prepared_sessions.ohlc, interim_output_path),
    )

    # Keep all staging directories alive until both outputs have been written and validated for
    # publication.
    with ExitStack() as staging:
        staged_paths = []

        # Stage each representation beside its destination so each final replacement remains
        # atomic.
        for table, destination in artifacts:
            destination.parent.mkdir(parents=True, exist_ok=True)
            directory = staging.enter_context(
                TemporaryDirectory(
                prefix=".pricesanity-", dir=destination.parent
            )
            )
            temporary_path = Path(directory) / destination.name
            table.to_parquet(temporary_path, index=False)
            staged_paths.append((temporary_path, destination))

        # Recheck before publication in case an output appeared during staging.
        if not overwrite:
            # Check both destinations again because a file may have appeared while the tables
            # were being written.
            for _, destination in staged_paths:
                # Honor overwrite protection even if another operation created a file during
                # staging.
                if destination.exists():
                    raise FileExistsError(
                        f"Output already exists: {destination}. "
                        "Use --overwrite to replace it."
                    )

        # These replacements are individually atomic, not a transaction across
        # both files. A failure here can still leave a partially published pair.
        for temporary_path, destination in staged_paths:
            temporary_path.replace(destination)

    # Return the same table for notebooks, tests, or later Python orchestration
    # without requiring the newly written Parquet file to be read again.
    return prepared_sessions.normalized


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for dataset preparation.

    Returns:
        Parser describing every required input and output path.
    """

    # Keep command-line definitions in a separate function so tests and future
    # interfaces can inspect them without executing dataset preparation.
    argument_parser = argparse.ArgumentParser(
        description=(
            "Prepare model-ready intraday candlesticks from Databento exports."
        )
    )

    # The configuration makes one run reproducible without hard-coding project
    # settings into this command.
    argument_parser.add_argument(
        "--config",
        required=True,
        type=Path,
        help="Path to the YAML project configuration.",
    )

    # Separate file arguments preserve the distinct price, schedule, and data-
    # quality responsibilities of the three Databento exports.
    argument_parser.add_argument(
        "--candlesticks",
        required=True,
        type=Path,
        help="Path to the Databento OHLC CSV.",
    )
    argument_parser.add_argument(
        "--status",
        required=True,
        type=Path,
        help="Path to the Databento status CSV.",
    )
    argument_parser.add_argument(
        "--conditions",
        required=True,
        type=Path,
        help="Path to the Databento condition JSON.",
    )

    # A required destination prevents processed data from being written to an
    # implicit location that the user may confuse with raw data.
    argument_parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Path for the processed Parquet file.",
    )

    # The interim artifact contains the exact OHLC candles viewed by the
    # annotator while the processed output remains model-only.
    argument_parser.add_argument(
        "--interim-output",
        required=True,
        type=Path,
        help="Path for validated 5-minute OHLC candlesticks.",
    )

    # Make replacement opt-in because rebuilding an artifact can change the
    # training set even when its destination name stays the same.
    argument_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace the output file if it already exists.",
    )

    argument_parser.add_argument(
        "--csv-chunk-rows",
        type=int,
        default=100_000,
        help="One-minute CSV rows per read before aggregation (default: 100000).",
    )

    return argument_parser


def main(arguments: Sequence[str] | None = None) -> int:
    """Run dataset preparation from command-line arguments.

    Args:
        arguments: Optional argument sequence used instead of the terminal.

    Returns:
        Zero after the processed dataset is saved successfully.
    """

    # Parse terminal input through one declared interface so missing or unknown
    # options receive consistent usage errors.
    parsed_arguments = build_argument_parser().parse_args(arguments)

    # Delegate all work to the reusable function so the CLI and Python callers
    # cannot develop different preparation behavior.
    prepared_candlestick_data = build_dataset_from_exports(
        parsed_arguments.candlesticks,
        parsed_arguments.status,
        parsed_arguments.conditions,
        parsed_arguments.output,
        config_path=parsed_arguments.config,
        interim_output_path=parsed_arguments.interim_output,
        overwrite=parsed_arguments.overwrite,
        csv_chunk_rows=parsed_arguments.csv_chunk_rows,
    )

    # Report the exact artifact and row count so a terminal run gives immediate
    # confirmation without printing the potentially large dataset.
    print(
        f"Saved {len(prepared_candlestick_data)} candlesticks to "
        f"{parsed_arguments.output}."
    )

    print(
        f"Saved {len(prepared_candlestick_data)} OHLC candlesticks to "
        f"{parsed_arguments.interim_output}."
    )

    return 0


# Only direct execution should begin dataset preparation.
if __name__ == "__main__":
    # Support direct module execution while the installed console command uses
    # the same main function through the project metadata.
    raise SystemExit(main())
