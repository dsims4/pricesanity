"""Read and display saved causal predictions without modifying annotations."""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PySide6.QtCore import Qt
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QCheckBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from pricesanity.config import AppConfig
from pricesanity.benchmark.artifacts import load_benchmark_run
from pricesanity.data.identifiers import build_candlestick_id
from pricesanity.gui.chart_widget import CandlestickChart
from pricesanity.training.artifacts import load_run_artifacts


@dataclass(frozen=True)
class TestRunData:
    """Verified OHLC candles, predictions, and boundaries for one test run."""

    metadata: dict[str, object]
    candlesticks: pd.DataFrame


def find_regime_change_markers(
    predicted_current_regimes: Sequence[str],
) -> tuple[tuple[int, str], ...]:
    """Locate the first candle of every new predicted current regime.

    Candle zero establishes the separately displayed starting regime. It is not
    called a change because no earlier prediction exists inside that session.
    """

    valid_regimes = {"bull", "bear", "range"}
    if not predicted_current_regimes:
        raise ValueError("A test session must contain current-regime predictions.")
    if not set(predicted_current_regimes).issubset(valid_regimes):
        raise ValueError("Test session contains an unknown predicted regime.")

    return tuple(
        (candle_position, predicted_current_regimes[candle_position])
        for candle_position in range(1, len(predicted_current_regimes))
        if predicted_current_regimes[candle_position]
        != predicted_current_regimes[candle_position - 1]
    )


def load_test_run(
    run_directory: str | Path,
    candlestick_path: str | Path | None,
    *,
    config: AppConfig,
) -> TestRunData:
    """Read a run without inference or database writes; require exact whole-session alignment.

    Args:
        run_directory: Directory containing model, prediction, and metadata artifacts.
        candlestick_path: OHLC Parquet path, or None to use recorded run metadata.
        config: Instrument, timestamp, timezone, and interval settings.

    Returns:
        Exactly aligned real test candles and integrity-checked saved predictions.

    Raises:
        ValueError: If artifacts are malformed, incompatible, or misaligned.
    """

    run_directory = Path(run_directory)
    if (run_directory / "benchmark_metadata.json").is_file():
        benchmark_metadata, predictions, _ = load_benchmark_run(run_directory)
        dataset = benchmark_metadata.get("dataset", {})
        identity = benchmark_metadata.get("identity", {})
        if not isinstance(dataset, dict) or not isinstance(identity, dict):
            raise ValueError("Benchmark run is missing dataset or model identity.")
        for field, expected_value in (
            ("instrument", config.data.instrument),
            ("target_interval", config.data.target_interval),
            ("session_timezone", config.data.session_timezone),
        ):
            if dataset.get(field) != expected_value:
                raise ValueError(f"Benchmark {field} does not match project configuration.")
        metadata = {
            "artifact_kind": "benchmark",
            "run_index": 1,
            "run_count": 1,
            "instrument": dataset["instrument"],
            "target_interval": dataset["target_interval"],
            "session_timezone": dataset["session_timezone"],
            "candlestick_path": dataset.get("candlestick_path"),
            "model_name": identity.get("model_name"),
            "run_name": identity.get("run_name"),
            "track": identity.get("track"),
        }
    else:
        metadata, predictions = load_run_artifacts(run_directory, config=config)

    selected_candlestick_path = (
        Path(candlestick_path)
        if candlestick_path is not None
        else Path(str(metadata.get("candlestick_path") or ""))
    )
    if not selected_candlestick_path.is_file():
        raise ValueError("A valid OHLC candlestick artifact is required.")
    candlesticks = pd.read_parquet(selected_candlestick_path)
    required_ohlc_columns = {
        config.data.timestamp_column,
        "open",
        "high",
        "low",
        "close",
    }
    missing_ohlc_columns = required_ohlc_columns.difference(candlesticks.columns)
    if missing_ohlc_columns:
        raise ValueError(
            "OHLC data is missing columns: "
            + ", ".join(sorted(missing_ohlc_columns))
        )

    # Keep identity columns and preparation evidence until validation finishes. Rebuilding
    # every ID from the requested instrument alone would disguise a wrong-market artifact.
    evidence = candlesticks.attrs.get("pricesanity_session_validation", {})
    if not isinstance(evidence, dict):
        raise ValueError("OHLC preparation evidence must be a named mapping.")

    if "instrument" in candlesticks:
        if not candlesticks["instrument"].eq(config.data.instrument).all():
            raise ValueError("OHLC instrument does not match the run.")
    elif evidence.get("instrument") != config.data.instrument:
        raise ValueError("OHLC requires instrument identity or validated preparation evidence.")

    expected_interval = pd.Timedelta(config.data.target_interval)
    if expected_interval <= pd.Timedelta(0):
        raise ValueError("The configured candle interval must be positive.")
    if evidence and (
        evidence.get("instrument") != config.data.instrument
        or pd.Timedelta(evidence.get("interval")) != expected_interval
        or evidence.get("timezone") != config.data.session_timezone
    ):
        raise ValueError("OHLC preparation evidence does not match the run settings.")

    candlesticks = candlesticks.copy()
    timestamps = pd.to_datetime(
        candlesticks[config.data.timestamp_column], utc=True, errors="raise"
    ).astype("datetime64[ns, UTC]")
    if (
        timestamps.isna().any()
        or timestamps.duplicated().any()
        or not timestamps.is_monotonic_increasing
    ):
        raise ValueError("OHLC candlesticks must be unique and chronological.")

    candlesticks[config.data.timestamp_column] = timestamps
    session_dates = timestamps.dt.tz_convert(config.data.session_timezone).dt.date

    # Select whole test days, never just timestamps found in predictions. An inner join
    # could make one-minute OHLC appear compatible by discarding four of every five rows.
    test_rows = session_dates.isin(predictions["session_date"])
    candlesticks = candlesticks.loc[test_rows].reset_index(drop=True)
    candlesticks.attrs.clear()
    test_timestamps = candlesticks[config.data.timestamp_column]
    prediction_timestamps = predictions["timestamp"].astype("datetime64[ns, UTC]")
    if not test_timestamps.equals(prediction_timestamps):
        raise ValueError("OHLC and prediction timestamps must match exactly in count and order.")

    # Mirror the annotation boundary's price checks without pulling Qt annotation or database
    # behavior into this read-only loader. Invalid candle geometry cannot be a useful audit.
    price_columns = ["open", "high", "low", "close"]
    for column in price_columns:
        candlesticks[column] = pd.to_numeric(candlesticks[column], errors="raise")

    if not np.isfinite(candlesticks[price_columns].to_numpy(dtype=float)).all():
        raise ValueError("OHLC prices must be finite numbers.")
    if (
        candlesticks["high"].lt(candlesticks[["open", "close", "low"]].max(axis=1)).any()
        or candlesticks["low"].gt(candlesticks[["open", "close", "high"]].min(axis=1)).any()
    ):
        raise ValueError("OHLC data contains invalid geometry.")

    for _, session_timestamps in test_timestamps.groupby(predictions["session_date"]):
        if not session_timestamps.diff().iloc[1:].eq(expected_interval).all():
            raise ValueError("OHLC session spacing does not match the configured interval.")

    expected_ids = pd.Series([
        build_candlestick_id(config.data.instrument, timestamp, config.data.target_interval)
        for timestamp in test_timestamps
    ])
    if not predictions["candlestick_id"].equals(expected_ids):
        raise ValueError("Prediction identifiers do not match their OHLC timestamps.")
    if "candlestick_id" in candlesticks and not candlesticks["candlestick_id"].equals(
        expected_ids
    ):
        raise ValueError("OHLC identifiers do not match the configured instrument and interval.")

    # Row-for-row validation has already succeeded; copy only the named price columns so an
    # unexpected source column can never replace model predictions or human comparison labels.
    aligned_candlesticks = predictions.copy()
    aligned_candlesticks[config.data.timestamp_column] = test_timestamps
    for column in price_columns:
        aligned_candlesticks[column] = candlesticks[column]

    return TestRunData(
        metadata=metadata,
        candlesticks=aligned_candlesticks.reset_index(drop=True),
    )


class TestResultsWindow(QMainWindow):
    """Navigate saved model predictions without opening an annotation store."""

    def __init__(
        self,
        test_run: TestRunData,
        *,
        timestamp_column: str = "ts_event",
    ) -> None:
        """Create a read-only retrospective test-results window."""

        super().__init__()
        self.test_run = test_run
        self.timestamp_column = timestamp_column
        self.session_timezone = str(test_run.metadata["session_timezone"])
        self.all_candlestick_data = test_run.candlesticks.copy()
        self.session_dates = list(
            self.all_candlestick_data["session_date"].drop_duplicates()
        )
        if not self.session_dates:
            raise ValueError("Test results must contain at least one session.")

        # Build positions once so revisiting sessions never filters the complete results table.
        self._session_indices = self.all_candlestick_data.groupby("session_date").indices

        self.active_session_position = 0
        self.active_candlestick_position = 0
        self.candlestick_data = pd.DataFrame()

        central_widget = QWidget(self)
        window_layout = QVBoxLayout(central_widget)

        navigation_layout = QHBoxLayout()
        window_layout.addLayout(navigation_layout)
        self.previous_session_button = QPushButton("Previous session")
        self.previous_session_button.clicked.connect(
            lambda: self.move_session(-1)
        )
        navigation_layout.addWidget(self.previous_session_button)
        self.next_session_button = QPushButton("Next session")
        self.next_session_button.clicked.connect(lambda: self.move_session(1))
        navigation_layout.addWidget(self.next_session_button)
        navigation_layout.addStretch()
        self.stop_button = QPushButton("Stop")
        self.stop_button.clicked.connect(self.close)
        navigation_layout.addWidget(self.stop_button)

        self.session_information = QLabel("")
        window_layout.addWidget(self.session_information)
        self.starting_regime_label = QLabel("")
        window_layout.addWidget(self.starting_regime_label)

        self.chart_frame = QFrame(self)
        self.chart_frame.setFrameShape(QFrame.Shape.Box)
        chart_layout = QVBoxLayout(self.chart_frame)
        chart_layout.setContentsMargins(0, 0, 0, 0)
        self.chart = CandlestickChart(
            self.chart_frame,
            timestamp_column=timestamp_column,
            session_timezone=self.session_timezone,
        )
        chart_layout.addWidget(self.chart)
        window_layout.addWidget(self.chart_frame, stretch=1)

        model_information_layout = QHBoxLayout()
        window_layout.addLayout(model_information_layout)
        self.candle_information = QLabel("")
        model_information_layout.addWidget(self.candle_information)
        model_information_layout.addStretch()
        self.model_current_label = QLabel("")
        model_information_layout.addWidget(self.model_current_label)
        self.model_anticipated_label = QLabel("")
        model_information_layout.addWidget(self.model_anticipated_label)

        human_information_layout = QHBoxLayout()
        window_layout.addLayout(human_information_layout)
        self.show_human_annotations = QCheckBox("Show human annotations")
        self.show_human_annotations.toggled.connect(self._update_active_labels)
        human_information_layout.addWidget(self.show_human_annotations)
        self.human_current_label = QLabel("Human current: Hidden")
        human_information_layout.addWidget(self.human_current_label)
        self.human_anticipated_label = QLabel("Human anticipated: Hidden")
        human_information_layout.addWidget(self.human_anticipated_label)
        human_information_layout.addStretch()

        self.setCentralWidget(central_widget)
        self._load_active_session()

        # Navigation reads saved rows only. No annotation store or write action
        # exists in this window, keeping evaluation separate from labeling.
        self.previous_candle_shortcut = QShortcut(QKeySequence("Left"), self.chart)
        self.previous_candle_shortcut.setContext(Qt.ShortcutContext.WidgetShortcut)
        self.previous_candle_shortcut.activated.connect(
            lambda: self.move_candle(-1)
        )
        self.next_candle_shortcut = QShortcut(QKeySequence("Right"), self.chart)
        self.next_candle_shortcut.setContext(Qt.ShortcutContext.WidgetShortcut)
        self.next_candle_shortcut.activated.connect(lambda: self.move_candle(1))

    def _load_active_session(self) -> None:
        """Draw one complete test session and its causal change markers."""

        session_date = self.session_dates[self.active_session_position]
        self.candlestick_data = self.all_candlestick_data.iloc[
            self._session_indices[session_date]
        ].reset_index(drop=True)
        self.active_candlestick_position = 0
        self.chart.draw_session(self.candlestick_data, 0)

        predicted_regimes = self.candlestick_data[
            "predicted_current_regime"
        ].tolist()
        regime_change_markers = find_regime_change_markers(predicted_regimes)

        # The completed chart is visible for retrospective inspection, but
        # these stored predictions were generated with causal attention only.
        self.chart.set_regime_change_markers(
            regime_change_markers, starting_regime=predicted_regimes[0]
        )
        if self.test_run.metadata.get("artifact_kind") == "benchmark":
            model_name = str(self.test_run.metadata.get("model_name", "Unknown model"))
            track = str(self.test_run.metadata.get("track", "unknown track"))
            self.session_information.setText(
                f"{model_name} | {track} | "
                f"Session {self.active_session_position + 1} / {len(self.session_dates)} | "
                f"{session_date}"
            )
        else:
            run_index = int(self.test_run.metadata["run_index"])
            run_count = int(self.test_run.metadata["run_count"])
            self.session_information.setText(
                f"Walk-forward run {run_index} / {run_count} | "
                f"Test session {self.active_session_position + 1} / "
                f"{len(self.session_dates)} | {session_date}"
            )
        self.starting_regime_label.setText(
            "Starting model regime: " + predicted_regimes[0].title()
        )
        self.previous_session_button.setEnabled(self.active_session_position > 0)
        self.next_session_button.setEnabled(
            self.active_session_position < len(self.session_dates) - 1
        )
        self._update_active_labels()
        self.chart.setFocus(Qt.FocusReason.OtherFocusReason)

    def _update_active_labels(self) -> None:
        """Display model evidence and optionally reveal human comparison labels."""

        active_candle = self.candlestick_data.iloc[
            self.active_candlestick_position
        ]
        timestamp = pd.Timestamp(active_candle[self.timestamp_column]).tz_convert(
            self.session_timezone
        )
        self.candle_information.setText(
            f"{timestamp.strftime('%Y-%m-%d %H:%M %Z')} | "
            f"Candle {self.active_candlestick_position + 1} / {len(self.candlestick_data)}"
        )
        self.model_current_label.setText(
            "Model current: " + str(active_candle["predicted_current_regime"]).title()
        )
        self.model_anticipated_label.setText(
            "Model anticipated: "
            + str(active_candle["predicted_anticipated_regime"]).title()
        )

        if self.show_human_annotations.isChecked():
            self.human_current_label.setText(
                "Human current: " + str(active_candle["human_current_regime"]).title()
            )
            self.human_anticipated_label.setText(
                "Human anticipated: "
                + str(active_candle["human_anticipated_regime"]).title()
            )
        else:
            self.human_current_label.setText("Human current: Hidden")
            self.human_anticipated_label.setText("Human anticipated: Hidden")

    def move_candle(self, direction: int) -> None:
        """Move one candle, crossing test-session boundaries when necessary."""

        requested_position = self.active_candlestick_position + direction
        if 0 <= requested_position < len(self.candlestick_data):
            self.active_candlestick_position = requested_position
            self.chart.set_active_candlestick(requested_position)
            self._update_active_labels()
            return

        requested_session = self.active_session_position + direction
        if not 0 <= requested_session < len(self.session_dates):
            return
        self.active_session_position = requested_session
        self._load_active_session()
        if direction < 0:
            self.active_candlestick_position = len(self.candlestick_data) - 1
            self.chart.set_active_candlestick(self.active_candlestick_position)
            self._update_active_labels()

    def move_session(self, direction: int) -> None:
        """Open the neighboring test session at its first candle."""

        requested_session = self.active_session_position + direction
        if not 0 <= requested_session < len(self.session_dates):
            return
        self.active_session_position = requested_session
        self._load_active_session()
