"""Open the benchmark comparison view from verified saved artifacts."""

import argparse
from collections.abc import Sequence
from pathlib import Path

from PySide6.QtWidgets import QApplication

from pricesanity.gui.model_comparison import ModelComparisonWindow, load_comparison_runs


def main(arguments: Sequence[str] | None = None) -> int:
    """Load completed runs once, then keep GUI navigation inference-free."""

    parser = argparse.ArgumentParser(description="Compare saved Price Sanity benchmark runs.")
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=Path("data/models/benchmark"),
    )
    parsed = parser.parse_args(arguments)

    # Reuse the host Qt application for embedded launches; the window owns artifact summaries.
    application = QApplication.instance() or QApplication([])
    try:
        # Startup needs summaries only, not every saved prediction table or fitted model.
        runs = load_comparison_runs(parsed.artifact_root)
    except ValueError as error:
        parser.error(str(error))

    window = ModelComparisonWindow(runs)
    window.setWindowTitle("Price Sanity Model Comparison")
    window.showMaximized()

    # Keep the window alive for navigation without invoking any training workflow.
    return application.exec()


if __name__ == "__main__":
    raise SystemExit(main())
