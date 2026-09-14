"""Load one saved model run and open its read-only results window."""

import argparse
from collections.abc import Sequence
from pathlib import Path

from PySide6.QtWidgets import QApplication

from pricesanity.config import load_config
from pricesanity.gui.test_results import TestResultsWindow, load_test_run


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the terminal interface for saved test-result inspection."""

    argument_parser = argparse.ArgumentParser(
        description="Inspect saved Price Sanity test predictions."
    )
    argument_parser.add_argument(
        "--run",
        required=True,
        type=Path,
        help="Run directory containing model, prediction, and metadata artifacts.",
    )
    argument_parser.add_argument(
        "--candlesticks",
        type=Path,
        help="OHLC Parquet path; defaults to the path recorded during training.",
    )
    argument_parser.add_argument(
        "--config",
        required=True,
        type=Path,
        help="Path to the YAML project configuration.",
    )
    return argument_parser


def main(arguments: Sequence[str] | None = None) -> int:
    """Validate saved artifacts before opening the read-only Qt window."""

    argument_parser = build_argument_parser()
    parsed_arguments = argument_parser.parse_args(arguments)
    try:
        config = load_config(parsed_arguments.config)
        test_run = load_test_run(
            parsed_arguments.run,
            parsed_arguments.candlesticks,
            config=config,
        )
    except (FileNotFoundError, ValueError) as error:
        argument_parser.error(str(error))

    application = QApplication.instance() or QApplication([])
    window = TestResultsWindow(
        test_run,
        timestamp_column=config.data.timestamp_column,
    )
    window.setWindowTitle("Price Sanity Test Results")
    window.resize(1400, 900)
    window.show()
    return application.exec()


if __name__ == "__main__":
    raise SystemExit(main())
