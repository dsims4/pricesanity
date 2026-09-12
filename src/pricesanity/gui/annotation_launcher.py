"""Load prepared sessions and launch the annotation window."""

import argparse
from collections.abc import Sequence
from datetime import date, timedelta
from pathlib import Path
import re

import numpy as np
import pandas as pd
from pyarrow import ArrowInvalid
from PySide6.QtWidgets import QApplication

from pricesanity.config import AppConfig, load_config
from pricesanity.data.identifiers import build_candlestick_id
from pricesanity.gui.annotation_app import AnnotationWindow


CORPUS_DATE_PATTERN = re.compile(r"(\d{4}-\d{2}-\d{2})")


def parse_corpus_date_bounds(corpus_path: str | Path) -> tuple[date, date]:
    """Read inclusive corpus boundaries from its dated filename.

    Args:
        corpus_path: Path containing inclusive start and exclusive end dates.

    Returns:
        Inclusive first and last calendar dates represented by the filename.

    Raises:
        ValueError: If two valid increasing dates cannot be identified.
    """
    # Use the final two ISO dates so instrument names may contain other digits
    # without changing how the requested corpus boundaries are interpreted.
    date_values = CORPUS_DATE_PATTERN.findall(Path(corpus_path).name)
    if len(date_values) < 2:
        raise ValueError(
            "Candlestick filenames must contain start and end dates."
        )

    # Databento and Price Sanity filenames preserve an exclusive ending date,
    # so the last included calendar date is the preceding day.
    starting_date = date.fromisoformat(date_values[-2])
    exclusive_ending_date = date.fromisoformat(date_values[-1])
    if starting_date >= exclusive_ending_date:
        raise ValueError("Corpus start date must be before its end date.")
    ending_date = exclusive_ending_date - timedelta(days=1)
    return starting_date, ending_date


def load_annotation_sessions(
    candlestick_path: str | Path,
    normalized_path: str | Path,
    *,
    config: AppConfig,
) -> pd.DataFrame:
    """Load prepared OHLC sessions and create stable candle identifiers.

    Args:
        candlestick_path: Path to validated interim OHLC candlesticks.
        normalized_path: Path to aligned normalized candlestick features.
        config: Project data and timezone settings.

    Returns:
        Chronological sessions with dates and stable candlestick identifiers.

    Raises:
        ValueError: If the file cannot provide valid candlesticks.
    """
    # Read the real-price interim artifact because annotations should be made
    # from recognizable OHLC candles rather than normalized model features.
    candlestick_data = pd.read_parquet(Path(candlestick_path))

    # Read the aligned model artifact separately because its causal opening gap
    # should be displayed without replacing the chart's real OHLC prices.
    try:
        normalized_data = pd.read_parquet(
            Path(normalized_path),
            columns=[config.data.timestamp_column, "open_gap"],
        )
    except ArrowInvalid as error:
        raise ValueError(
            "Normalized artifact must be valid Parquet with timestamp "
            "and open_gap columns."
        ) from error

    # Fail before date selection when the artifact is not the output expected
    # from the trusted data-preparation pipeline.
    required_columns = {
        config.data.timestamp_column,
        "open",
        "high",
        "low",
        "close",
    }
    missing_columns = required_columns.difference(candlestick_data.columns)
    if missing_columns:
        raise ValueError(
            "Interim candlestick data is missing required columns: "
            f"{', '.join(sorted(missing_columns))}."
        )

    # Normalize timestamps to UTC so identifiers remain stable across daylight
    # saving changes and computers with different local timezone settings.
    timestamps = pd.to_datetime(
        candlestick_data[config.data.timestamp_column],
        utc=True,
        errors="raise",
    )
    candlestick_data[config.data.timestamp_column] = timestamps

    # Apply the same UTC representation to normalized timestamps before using
    # them as alignment keys.
    normalized_timestamps = pd.to_datetime(
        normalized_data[config.data.timestamp_column],
        utc=True,
        errors="raise",
    )
    normalized_data[config.data.timestamp_column] = normalized_timestamps

    # The GUI cannot display a missing or infinite opening gap as a trustworthy
    # annotation reference.
    normalized_data["open_gap"] = pd.to_numeric(
        normalized_data["open_gap"],
        errors="coerce",
    )
    has_finite_opening_gaps = np.isfinite(
        normalized_data["open_gap"].to_numpy(dtype=float)
    ).all()
    if not has_finite_opening_gaps:
        raise ValueError("Opening gaps must be finite numbers.")

    # Duplicate feature timestamps would make one OHLC candle match more than
    # one opening-gap value.
    if normalized_data[config.data.timestamp_column].duplicated().any():
        raise ValueError("Normalized candlestick timestamps must be unique.")

    # Require complete two-way alignment so neither displayed candles nor model
    # features silently disappear during the merge.
    candlestick_timestamps = set(
        candlestick_data[config.data.timestamp_column]
    )
    normalized_timestamp_set = set(
        normalized_data[config.data.timestamp_column]
    )
    if candlestick_timestamps != normalized_timestamp_set:
        raise ValueError(
            "OHLC and normalized candlestick timestamps must match exactly."
        )

    # Attach only the opening gap through a validated one-to-one timestamp join.
    candlestick_data = candlestick_data.merge(
        normalized_data,
        on=config.data.timestamp_column,
        how="left",
        validate="one_to_one",
    )

    # Sort before assigning identifiers because the chart and arrow navigation
    # must follow the same chronology used by the Transformer.
    candlestick_data = candlestick_data.sort_values(
        config.data.timestamp_column
    ).reset_index(drop=True)

    # Convert UTC timestamps to the exchange timezone only for deciding which
    # local trading date each prepared candle belongs to.
    local_session_dates = candlestick_data[
        config.data.timestamp_column
    ].dt.tz_convert(config.data.session_timezone).dt.date
    if local_session_dates.empty:
        raise ValueError("Interim candlestick data cannot be empty.")
    candlestick_data["session_date"] = local_session_dates

    # Combine instrument identity with each UTC timestamp so annotations remain
    # aligned when files, sessions, or tables are later combined.
    selected_timestamps = candlestick_data[
        config.data.timestamp_column
    ]
    candlestick_ids = selected_timestamps.map(
        lambda timestamp: build_candlestick_id(
            config.data.instrument, timestamp, config.data.target_interval
        )
    )
    candlestick_data.insert(
        0,
        "candlestick_id",
        candlestick_ids,
    )

    # Return every prepared date because the GUI applies and changes its range
    # without rereading this file or reopening the application.
    return candlestick_data


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for the annotation GUI.

    Returns:
        Parser describing the candlestick, configuration, and database paths.
    """
    # Keep parsing separate from launching so tests can inspect this interface
    # without opening a desktop window.
    argument_parser = argparse.ArgumentParser(
        description="Annotate prepared candlestick sessions."
    )

    # The interim path contains validated real-price OHLC data for the chart.
    argument_parser.add_argument(
        "--candlesticks",
        required=True,
        type=Path,
        help="Path to validated interim OHLC Parquet data.",
    )

    # The normalized artifact supplies the causal opening gap aligned with each
    # real-price candlestick shown in the chart.
    argument_parser.add_argument(
        "--normalized",
        required=True,
        type=Path,
        help="Path to aligned normalized Parquet data.",
    )

    # Configuration supplies the instrument and exchange timezone used to
    # create session boundaries and stable candlestick identifiers.
    argument_parser.add_argument(
        "--config",
        required=True,
        type=Path,
        help="Path to the YAML project configuration.",
    )

    # Human annotations default to the ignored annotation directory so they
    # cannot accidentally enter a repository push with market data.
    argument_parser.add_argument(
        "--database",
        type=Path,
        default=Path("data/annotations/pricesanity.sqlite3"),
        help=(
            "Ignored SQLite output path "
            "(default: data/annotations/pricesanity.sqlite3)."
        ),
    )

    return argument_parser


def main(arguments: Sequence[str] | None = None) -> int:
    """Load the requested session and run its annotation window.

    Args:
        arguments: Optional argument sequence used instead of the terminal.

    Returns:
        Qt application exit status after the window closes.
    """
    # Parse every path before creating GUI state.
    argument_parser = build_argument_parser()
    parsed_arguments = argument_parser.parse_args(arguments)

    # Use the same typed configuration that prepared the interim data so the
    # launcher cannot silently assume a different instrument or timezone.
    config = load_config(parsed_arguments.config)

    # Load and identify all prepared sessions before Qt opens a desktop window.
    try:
        corpus_start_date, corpus_end_date = parse_corpus_date_bounds(
            parsed_arguments.candlesticks
        )
        normalized_start_date, normalized_end_date = parse_corpus_date_bounds(
            parsed_arguments.normalized
        )
        if (
            corpus_start_date != normalized_start_date
            or corpus_end_date != normalized_end_date
        ):
            raise ValueError(
                "OHLC and normalized filenames must use the same date range."
            )

        candlestick_data = load_annotation_sessions(
            parsed_arguments.candlesticks,
            parsed_arguments.normalized,
            config=config,
        )
    except (FileNotFoundError, ValueError) as error:
        argument_parser.error(str(error))

    # Reuse a Qt application when launched from an interactive Python process;
    # otherwise create the one application allowed for this process.
    application = QApplication.instance()
    if application is None:
        application = QApplication([])

    # Keep the window alive until Qt's event loop ends and give it enough room
    # to show a complete regular session without crowding the label controls.
    window = AnnotationWindow(
        candlestick_data,
        parsed_arguments.database,
        timestamp_column=config.data.timestamp_column,
        corpus_start_date=corpus_start_date,
        corpus_end_date=corpus_end_date,
    )
    window.setWindowTitle("Price Sanity Annotation")
    window.resize(1400, 900)
    window.show()

    return application.exec()


if __name__ == "__main__":
    # Support direct module execution while the installed terminal command uses
    # this same function through the project metadata.
    raise SystemExit(main())
