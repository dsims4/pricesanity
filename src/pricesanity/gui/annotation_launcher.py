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
from pricesanity.data.annotation_evidence import validate_annotation_evidence
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

    # Both date boundaries are required to interpret the corpus filename safely.
    if len(date_values) < 2:
        raise ValueError(
            "Candlestick filenames must contain start and end dates."
        )

    # Databento and Price Sanity filenames preserve an exclusive ending date,
    # so the last included calendar date is the preceding day.
    starting_date = date.fromisoformat(date_values[-2])
    exclusive_ending_date = date.fromisoformat(date_values[-1])

    # A reversed or empty range cannot describe any included trading dates.
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
            columns=[config.data.timestamp_column, "instrument", "open_gap"],
        )

    # Report an unreadable Parquet export as an input error with the relevant path.
    except ArrowInvalid as error:
        raise ValueError(
            "Normalized artifact must be valid Parquet with timestamp, "
            "instrument, and open_gap columns."
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

    # The annotation view needs complete candle geometry and identifiers before opening a window.
    if missing_columns:
        raise ValueError(
            "Interim candlestick data is missing required columns: "
            f"{', '.join(sorted(missing_columns))}."
        )

    # The chart must not turn malformed text, infinities, or impossible geometry
    # into a visual candle that appears safe to annotate.
    price_columns = ["open", "high", "low", "close"]
    for price_column in price_columns:
        candlestick_data[price_column] = pd.to_numeric(
            candlestick_data[price_column], errors="coerce"
        )
    if not np.isfinite(candlestick_data[price_columns].to_numpy(dtype=float)).all():
        raise ValueError("Interim OHLC prices must be finite numbers.")
    invalid_high = candlestick_data["high"] < candlestick_data[
        ["open", "close"]
    ].max(axis="columns")
    invalid_low = candlestick_data["low"] > candlestick_data[
        ["open", "close"]
    ].min(axis="columns")
    if invalid_high.any() or invalid_low.any():
        raise ValueError("Interim candlestick data contains invalid OHLC geometry.")

    # Normalize timestamps to UTC so identifiers remain stable across daylight
    # saving changes and computers with different local timezone settings.
    timestamps = pd.to_datetime(
        candlestick_data[config.data.timestamp_column],
        utc=True,
        errors="raise",
    )
    candlestick_data[config.data.timestamp_column] = timestamps

    # Prepared artifacts must already be unique and chronological. Sorting here
    # would hide a damaged export and weaken the row-for-row alignment check below.
    if timestamps.duplicated().any() or not timestamps.is_monotonic_increasing:
        raise ValueError(
            "Interim candlestick timestamps must be unique and chronological."
        )

    # Apply the same UTC representation to normalized timestamps before using
    # them as alignment keys.
    normalized_timestamps = pd.to_datetime(
        normalized_data[config.data.timestamp_column],
        utc=True,
        errors="raise",
    )
    normalized_data[config.data.timestamp_column] = normalized_timestamps

    # One configured instrument must describe every normalized row so a valid
    # price table cannot receive identifiers belonging to another market.
    normalized_instruments = normalized_data["instrument"].astype(str)
    if normalized_instruments.empty or not normalized_instruments.eq(
        config.data.instrument
    ).all():
        raise ValueError(
            "Normalized candlestick instrument must match the configured instrument."
        )

    # The GUI cannot display a missing or infinite opening gap as a trustworthy
    # annotation reference.
    normalized_data["open_gap"] = pd.to_numeric(
        normalized_data["open_gap"],
        errors="coerce",
    )
    has_finite_opening_gaps = np.isfinite(
        normalized_data["open_gap"].to_numpy(dtype=float)
    ).all()

    # Nonfinite opening gaps cannot be displayed as valid normalized input.
    if not has_finite_opening_gaps:
        raise ValueError("Opening gaps must be finite numbers.")

    # Exact row order is part of the prepared artifact contract. Set equality
    # alone could conceal a reordered normalized table.
    if not timestamps.equals(normalized_timestamps):
        raise ValueError(
            "OHLC and normalized candlestick timestamps must match exactly row for row."
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
    ).reset_index(
        drop=True
    )

    # Convert UTC timestamps to the exchange timezone only for deciding which
    # local trading date each prepared candle belongs to.
    local_session_dates = (
        candlestick_data[
        config.data.timestamp_column
    ].dt.tz_convert(config.data.session_timezone)
        .dt.date
    )

    # Do not open an annotation window without an eligible session to display.
    if local_session_dates.empty:
        raise ValueError("Interim candlestick data cannot be empty.")

    candlestick_data["session_date"] = local_session_dates

    # Confirm the supplied configuration still names the interval used to
    # prepare these sessions before that interval becomes part of every stable ID.
    target_interval = pd.Timedelta(config.data.target_interval)
    if pd.isna(target_interval) or target_interval <= pd.Timedelta(0):
        raise ValueError("Configured target interval must be positive.")
    within_session_differences = candlestick_data.groupby(
        "session_date", sort=False
    )[config.data.timestamp_column].diff().dropna()
    if not within_session_differences.eq(target_interval).all():
        raise ValueError(
            "Prepared candlestick spacing must match the configured target interval."
        )

    # Validated export metadata preserves reference-only days that are absent
    # from this display. Check full session grids and every opening-gap ratio
    # before any candle identifier can become an annotation target.
    validate_annotation_evidence(candlestick_data, normalized_data, config=config)

    # Validation is complete. Keeping the corpus metadata on this display table
    # would make pandas copy every session's evidence during routine row access.
    candlestick_data.attrs.clear()

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

        # The two files must describe the same requested corpus rather than merely overlapping
        # timestamps.
        if corpus_start_date != normalized_start_date or corpus_end_date != normalized_end_date:
            raise ValueError(
                "OHLC and normalized filenames must use the same date range."
            )

        candlestick_data = load_annotation_sessions(
            parsed_arguments.candlesticks,
            parsed_arguments.normalized,
            config=config,
        )

    # Input problems should produce a concise CLI error before a window is shown.
    except (FileNotFoundError, ValueError) as error:
        argument_parser.error(str(error))

    # Reuse a Qt application when launched from an interactive Python process;
    # otherwise create the one application allowed for this process.
    application = QApplication.instance()

    # Reuse the existing Qt application when another window or test already owns it.
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


# Only direct execution should launch the GUI; imports remain reusable by tests and callers.
if __name__ == "__main__":
    # Support direct module execution while the installed terminal command uses
    # this same function through the project metadata.
    raise SystemExit(main())
