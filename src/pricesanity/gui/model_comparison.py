"""Compare verified saved benchmark metrics without training or inference."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json

import numpy as np
import pandas as pd

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QGridLayout,
    QHeaderView,
    QLabel,
    QMainWindow,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from pricesanity.benchmark.artifacts import load_benchmark_summary
from pricesanity.benchmark.artifacts import load_benchmark_run
from pricesanity.benchmark.aggregation import aggregate_seed_results, add_baseline_deltas
from pricesanity.benchmark.registry import list_model_families

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle

from pricesanity.benchmark.artifacts import file_sha256


@dataclass(frozen=True)
class ComparisonRun:
    """Verified metrics needed by the leaderboard and two-model detail view."""

    display_name: str
    directory: Path
    metadata: dict[str, Any]
    metrics: dict[str, Any]


def load_aligned_comparison_predictions(
    first: ComparisonRun,
    second: ComparisonRun,
) -> tuple[Any, Any]:
    """Lazily load two runs and require their exact ordered evaluation population."""

    _, first_predictions, _ = load_benchmark_run(first.directory)
    _, second_predictions, _ = load_benchmark_run(second.directory)
    identity_columns = [
        "candlestick_id", "timestamp", "session_date", "session_index",
        "candle_position", "human_current_regime", "human_anticipated_regime",
    ]
    if not first_predictions[identity_columns].equals(
        second_predictions[identity_columns]
    ):
        raise ValueError(
            "Selected models do not share the exact ordered evaluation candles."
        )
    return first_predictions, second_predictions


def format_uncertainty_values(row: Any, *, head: str) -> str:
    """Label probability estimates and uncalibrated scores honestly for one candle."""

    if head not in {"current", "anticipated"}:
        raise ValueError("Prediction head must be current or anticipated.")
    values = ", ".join(
        f"{regime.title()} {row[f'{head}_{value_kind}_{regime}']:.3f}"
        for regime in ("bull", "bear", "range")
        for value_kind in (
            ["score"]
            if row["uncertainty_kind"] == "uncalibrated_decision_score"
            else ["probability"]
        )
    )
    label = (
        "Uncalibrated score"
        if row["uncertainty_kind"] == "uncalibrated_decision_score"
        else "Probability estimate"
    )
    return f"{label}: {values}"


def load_comparison_runs(artifact_root: str | Path) -> tuple[ComparisonRun, ...]:
    """Read only complete checksummed benchmark runs beneath the artifact root."""

    runs = []
    for metadata_path in sorted(Path(artifact_root).rglob("benchmark_metadata.json")):
        metadata, metrics = load_benchmark_summary(metadata_path.parent)
        identity = metadata["identity"]
        runs.append(
            ComparisonRun(
                display_name=(
                    f"{identity['track']} / {identity['model_name']} / "
                    f"{identity['run_name']} / seed {identity['seed']}"
                ),
                directory=metadata_path.parent,
                metadata=metadata,
                metrics=metrics,
            )
        )
    return tuple(runs)


class ModelComparisonWindow(QMainWindow):
    """Explore complete immutable benchmark artifacts without running a model."""

    def __init__(self, runs: tuple[ComparisonRun, ...]) -> None:
        """Build an honest empty state or a selectable comparison."""

        super().__init__()
        self.runs = runs
        self._ohlc_cache: dict[Path, pd.DataFrame] = {}
        central_widget = QWidget(self)
        layout = QVBoxLayout(central_widget)
        self.status_label = QLabel()
        layout.addWidget(self.status_label)
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs, stretch=1)

        leaderboard_page = QWidget()
        leaderboard_layout = QVBoxLayout(leaderboard_page)
        self.leaderboard = QTableWidget(0, 15)
        self.leaderboard.setHorizontalHeaderLabels([
            "Track", "Model", "Configuration", "Seeds", "Current macro-F1",
            "Anticipated macro-F1", "Mean-head macro-F1", "Δ majority",
            "Δ persistence*", "Parameters", "Model bytes", "Training seconds",
            "Inference rows/sec", "Device", "Runtime comparable",
        ])
        self.leaderboard.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.ResizeToContents
        )
        self.leaderboard.setSortingEnabled(True)
        leaderboard_layout.addWidget(self.leaderboard)
        leaderboard_layout.addWidget(QLabel(
            "* Persistence repeats the prior human regime and is a non-deployable reference."
        ))
        self.tabs.addTab(leaderboard_page, "Leaderboard")

        comparison_page = QWidget()
        comparison_layout = QVBoxLayout(comparison_page)
        selectors = QGridLayout()
        comparison_layout.addLayout(selectors)
        selectors.addWidget(QLabel("Model A"), 0, 0)
        selectors.addWidget(QLabel("Model B"), 0, 1)
        self.model_a = QComboBox()
        self.model_b = QComboBox()
        selectors.addWidget(self.model_a, 1, 0)
        selectors.addWidget(self.model_b, 1, 1)
        selectors.addWidget(QLabel("Session"), 0, 2)
        self.session_selector = QComboBox()
        selectors.addWidget(self.session_selector, 1, 2)
        self.comparison_chart = _ABSessionCanvas()
        comparison_layout.addWidget(self.comparison_chart, stretch=1)
        self.model_a_details = QLabel()
        self.model_b_details = QLabel()
        self.model_a_details.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.model_b_details.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        selectors.addWidget(self.model_a_details, 2, 0)
        selectors.addWidget(self.model_b_details, 2, 1)
        self.tabs.addTab(comparison_page, "A/B Session Comparison")

        self.metrics_canvas = _ConfusionCanvas()
        self.tabs.addTab(self.metrics_canvas, "Metrics / Confusion")

        self.learning_curves = QTableWidget(0, 4)
        self.learning_curves.setHorizontalHeaderLabels(
            ["Track", "Model", "Training sessions", "Mean-head macro-F1"]
        )
        self.learning_curves.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        self.tabs.addTab(self.learning_curves, "Learning Curves")

        self.efficiency = QTableWidget(0, 7)
        self.efficiency.setHorizontalHeaderLabels([
            "Track", "Model", "Device", "Hardware", "Training seconds",
            "Inference rows/sec", "Model bytes",
        ])
        self.efficiency.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.ResizeToContents
        )
        self.tabs.addTab(self.efficiency, "Efficiency")
        self.setCentralWidget(central_widget)

        if not runs:
            self.status_label.setText(
                "No completed benchmark artifacts. The annotation GUI and existing "
                "Transformer results remain available."
            )
            self.model_a.setEnabled(False)
            self.model_b.setEnabled(False)
            self.session_selector.setEnabled(False)
            return

        self.status_label.setText(f"{len(runs)} verified completed artifact run(s).")
        for run in runs:
            self.model_a.addItem(run.display_name)
            self.model_b.addItem(run.display_name)
        if len(runs) > 1:
            self.model_b.setCurrentIndex(1)
        self.model_a.currentIndexChanged.connect(self._update_details)
        self.model_b.currentIndexChanged.connect(self._update_details)
        self.session_selector.currentIndexChanged.connect(self._draw_comparison)
        self._populate_leaderboard()
        self._populate_learning_curves()
        self._populate_efficiency()
        self._update_details()

    def _populate_leaderboard(self) -> None:
        """Render comparable saved fields without constructing an opaque score."""

        raw = _runs_frame(self.runs, final_only=True)
        if raw.empty:
            self.status_label.setText(
                "Artifacts exist, but no final-holdout runs are complete yet."
            )
            return
        stochastic = [family.name for family in list_model_families() if family.stochastic]
        try:
            aggregated = add_baseline_deltas(aggregate_seed_results(
                raw,
                stochastic_models=stochastic,
                expected_stochastic_seed_count=3,
            ))
        except ValueError as error:
            # A partial seed collection is valid resumable work, but it is not a final result.
            # Hiding that distinction would reward whichever random seed finished first.
            self.status_label.setText(f"Incomplete final seed set: {error}")
            self.leaderboard.setRowCount(0)
            return
        self.status_label.setText(
            f"{len(aggregated)} complete configuration(s); all values are saved artifacts."
        )
        self.leaderboard.setRowCount(len(aggregated))
        for row_index, row in aggregated.iterrows():
            configuration = row.get("model_configuration") or {}
            hardware = row.get("hardware_fingerprint") or {}
            values = (
                row["track"], row["model_name"], _configuration_text(configuration),
                row["seed_count"],
                f"{row['current_macro_f1_mean']:.4f} ± {row['current_macro_f1_std']:.4f}",
                f"{row['anticipated_macro_f1_mean']:.4f} ± {row['anticipated_macro_f1_std']:.4f}",
                f"{row['mean_head_macro_f1']:.4f}",
                _optional_number(row.get("delta_vs_majority")),
                _optional_number(row.get("delta_vs_persistence")),
                row.get("parameter_count") or "N/A",
                row.get("serialized_model_bytes") or "N/A",
                _optional_number(row.get("training_seconds_mean"), digits=3),
                _optional_number(row.get("inference_samples_per_second_mean"), digits=1),
                row.get("device") or "N/A",
                "Yes" if row.get("hardware_compatible") else "No",
            )
            for column_index, value in enumerate(values):
                self.leaderboard.setItem(
                    row_index, column_index, QTableWidgetItem(str(value))
                )

    def _update_details(self) -> None:
        """Show per-class and transition evidence for the two selected runs."""

        if not self.runs:
            return
        self.model_a_details.setText(
            _format_run_details(self.runs[self.model_a.currentIndex()])
        )
        self.model_b_details.setText(
            _format_run_details(self.runs[self.model_b.currentIndex()])
        )
        self.metrics_canvas.show_runs(
            self.runs[self.model_a.currentIndex()],
            self.runs[self.model_b.currentIndex()],
        )
        self._load_comparison_sessions()

    def _load_comparison_sessions(self) -> None:
        """List sessions only after strict row-for-row A/B compatibility succeeds."""

        try:
            first, _ = load_aligned_comparison_predictions(
                self.runs[self.model_a.currentIndex()],
                self.runs[self.model_b.currentIndex()],
            )
        except ValueError as error:
            self.session_selector.clear()
            self.comparison_chart.show_message(str(error))
            return
        previous = self.session_selector.currentText()
        self.session_selector.blockSignals(True)
        self.session_selector.clear()
        for session_date in first["session_date"].drop_duplicates():
            self.session_selector.addItem(str(session_date))
        if previous:
            selected = self.session_selector.findText(previous)
            if selected >= 0:
                self.session_selector.setCurrentIndex(selected)
        self.session_selector.blockSignals(False)
        self._draw_comparison()

    def _draw_comparison(self) -> None:
        if not self.runs or not self.session_selector.currentText():
            return
        first_run = self.runs[self.model_a.currentIndex()]
        second_run = self.runs[self.model_b.currentIndex()]
        try:
            first, second = load_aligned_comparison_predictions(first_run, second_run)
            session_date = self.session_selector.currentText()
            mask = first["session_date"].astype(str).eq(session_date)
            session_first = first.loc[mask].reset_index(drop=True)
            self.comparison_chart.show_session(
                session_first,
                second.loc[mask].reset_index(drop=True),
                first_name=first_run.display_name,
                second_name=second_run.display_name,
                ohlc=self._load_ohlc_session(first_run, second_run, session_first),
            )
        except ValueError as error:
            self.comparison_chart.show_message(str(error))

    def _load_ohlc_session(
        self,
        first_run: ComparisonRun,
        second_run: ComparisonRun,
        predictions: pd.DataFrame,
    ) -> pd.DataFrame | None:
        """Load a checksummed OHLC source once; prediction browsing never invokes inference."""

        first_dataset = first_run.metadata.get("dataset", {})
        second_dataset = second_run.metadata.get("dataset", {})
        if not isinstance(first_dataset, dict) or not isinstance(second_dataset, dict):
            return None
        path_value = first_dataset.get("candlestick_path")
        expected_hash = first_dataset.get("candlestick_sha256")
        if (
            not path_value
            or expected_hash != second_dataset.get("candlestick_sha256")
            or path_value != second_dataset.get("candlestick_path")
        ):
            return None
        path = Path(str(path_value))
        if not path.is_file() or file_sha256(path) != expected_hash:
            return None
        if path not in self._ohlc_cache:
            self._ohlc_cache[path] = pd.read_parquet(path)
        candles = self._ohlc_cache[path]
        timestamp_column = "ts_event" if "ts_event" in candles.columns else "timestamp"
        if not {timestamp_column, "open", "high", "low", "close"}.issubset(candles.columns):
            return None
        timestamps = pd.to_datetime(candles[timestamp_column], utc=True, errors="coerce")
        selected = candles.loc[timestamps.isin(predictions["timestamp"])].copy()
        selected[timestamp_column] = pd.to_datetime(
            selected[timestamp_column], utc=True, errors="raise"
        )
        selected = selected.sort_values(timestamp_column).reset_index(drop=True)
        if not selected[timestamp_column].equals(
            predictions["timestamp"].astype("datetime64[ns, UTC]")
        ):
            return None
        return selected.rename(columns={timestamp_column: "timestamp"})

    def _populate_learning_curves(self) -> None:
        rows = []
        for run in self.runs:
            run_name = str(run.metadata["identity"]["run_name"])
            if not run_name.startswith("train_"):
                continue
            rows.append((
                run.metadata["identity"]["track"],
                run.metadata["identity"]["model_name"],
                run_name.removeprefix("train_"),
                f"{run.metrics['mean_head_macro_f1']:.4f}",
            ))
        self.learning_curves.setRowCount(len(rows))
        _fill_table(self.learning_curves, rows)

    def _populate_efficiency(self) -> None:
        rows = []
        for run in self.runs:
            efficiency = run.metrics.get("efficiency") or {}
            hardware = efficiency.get("hardware_fingerprint") or {}
            rows.append((
                run.metadata["identity"]["track"],
                run.metadata["identity"]["model_name"],
                efficiency.get("device", "N/A"),
                hardware.get("accelerator_name") or hardware.get("processor") or hardware.get("machine", "N/A"),
                _optional_number(efficiency.get("training_seconds"), digits=3),
                _optional_number(efficiency.get("inference_samples_per_second"), digits=1),
                efficiency.get("serialized_model_bytes", "N/A"),
            ))
        self.efficiency.setRowCount(len(rows))
        _fill_table(self.efficiency, rows)


class _ABSessionCanvas(FigureCanvasQTAgg):
    """Render aligned prices, six regime rows, and persisted uncertainty evidence."""

    def __init__(self) -> None:
        self.figure = Figure(figsize=(12, 7), tight_layout=True)
        super().__init__(self.figure)

    def show_message(self, message: str) -> None:
        self.figure.clear()
        axes = self.figure.add_subplot(1, 1, 1)
        axes.text(0.5, 0.5, message, ha="center", va="center", wrap=True)
        axes.set_axis_off()
        self.draw_idle()

    def show_session(
        self,
        first: pd.DataFrame,
        second: pd.DataFrame,
        *,
        first_name: str,
        second_name: str,
        ohlc: pd.DataFrame | None,
    ) -> None:
        self.figure.clear()
        price_axes, regime_axes, uncertainty_axes = self.figure.subplots(
            3, 1, sharex=True, gridspec_kw={"height_ratios": [3, 1.7, 1]}
        )
        positions = np.arange(len(first))
        if ohlc is None:
            price_axes.text(
                0.5, 0.5,
                "OHLC source was not recorded with this frozen snapshot.\n"
                "Regime and uncertainty comparison remains exact.",
                transform=price_axes.transAxes, ha="center", va="center",
            )
            price_axes.set_ylabel("Price unavailable")
        else:
            _draw_candles(price_axes, ohlc)
            price_axes.set_ylabel("Price")
            price_axes.yaxis.tick_right()
            price_axes.yaxis.set_label_position("right")
        regime_columns = (
            (first, "human_current_regime", "Human current"),
            (first, "human_anticipated_regime", "Human anticipated"),
            (first, "predicted_current_regime", "A current"),
            (first, "predicted_anticipated_regime", "A anticipated"),
            (second, "predicted_current_regime", "B current"),
            (second, "predicted_anticipated_regime", "B anticipated"),
        )
        mapping = {"bull": 0, "bear": 1, "range": 2}
        matrix = np.asarray([
            values[column].map(mapping).to_numpy(dtype=int)
            for values, column, _ in regime_columns
        ])
        regime_axes.imshow(
            matrix,
            aspect="auto",
            interpolation="nearest",
            cmap=ListedColormap(["tab:green", "tab:red", "tab:orange"]),
            vmin=0,
            vmax=2,
            extent=(-0.5, len(first) - 0.5, len(regime_columns) - 0.5, -0.5),
        )
        regime_axes.set_yticks(
            np.arange(len(regime_columns)), [row[2] for row in regime_columns]
        )
        regime_axes.set_ylabel("Saved labels")
        for values, prefix, color in (
            (first, "A", "tab:blue"), (second, "B", "tab:purple")
        ):
            for head, line_style in (("current", "-"), ("anticipated", "--")):
                uncertainty = _uncertainty_series(values, head=head)
                uncertainty_axes.plot(
                    positions,
                    uncertainty,
                    line_style,
                    color=color,
                    label=f"{prefix} {head}",
                )
        uncertainty_axes.set_ylabel("Probability / score")
        uncertainty_axes.set_xlabel("Candle position")
        uncertainty_axes.legend(loc="upper left", ncol=4, fontsize=7)
        session_date = str(first["session_date"].iloc[0])
        price_axes.set_title(
            f"{session_date} — A: {first_name}\nB: {second_name}", fontsize=9
        )
        price_axes.grid(axis="y", alpha=0.2)
        self.draw_idle()


class _ConfusionCanvas(FigureCanvasQTAgg):
    """Show four readable matrices instead of nested-list text."""

    def __init__(self) -> None:
        self.figure = Figure(figsize=(9, 6), tight_layout=True)
        super().__init__(self.figure)

    def show_runs(self, first: ComparisonRun, second: ComparisonRun) -> None:
        self.figure.clear()
        axes = self.figure.subplots(2, 2)
        for row, (run, model_label) in enumerate(((first, "Model A"), (second, "Model B"))):
            for column, head in enumerate(("current", "anticipated")):
                matrix = np.asarray(run.metrics[head]["confusion_matrix"], dtype=int)
                target = axes[row, column]
                target.imshow(matrix, cmap="Blues")
                for human in range(3):
                    for predicted in range(3):
                        target.text(predicted, human, str(matrix[human, predicted]),
                                    ha="center", va="center")
                target.set_xticks(range(3), ["Bull", "Bear", "Range"])
                target.set_yticks(range(3), ["Bull", "Bear", "Range"])
                target.set_xlabel("Predicted")
                target.set_ylabel("Human")
                target.set_title(f"{model_label}: {head.title()}")
        self.draw_idle()


def _runs_frame(runs: tuple[ComparisonRun, ...], *, final_only: bool) -> pd.DataFrame:
    rows = []
    for run in runs:
        identity = run.metadata["identity"]
        if final_only and not str(identity["run_name"]).startswith("final_seed_"):
            continue
        efficiency = run.metrics.get("efficiency") or {}
        rows.append({
            "track": identity["track"],
            "model_name": identity["model_name"],
            "model_configuration_sha256": identity["model_configuration_sha256"],
            "representation_sha256": identity["representation_sha256"],
            "test_session_ids_sha256": identity["test_session_ids_sha256"],
            "seed": identity["seed"],
            "current_macro_f1": run.metrics["current"]["macro_f1"],
            "anticipated_macro_f1": run.metrics["anticipated"]["macro_f1"],
            "model_configuration": run.metadata.get("model_configuration"),
            "training_seconds": efficiency.get("training_seconds"),
            "inference_seconds": efficiency.get("inference_seconds"),
            "inference_samples_per_second": efficiency.get("inference_samples_per_second"),
            "serialized_model_bytes": efficiency.get("serialized_model_bytes"),
            "parameter_count": efficiency.get("parameter_count"),
            "device": efficiency.get("device"),
            "hardware_fingerprint": efficiency.get("hardware_fingerprint"),
        })
    return pd.DataFrame(rows)


def _configuration_text(configuration: Any) -> str:
    if not isinstance(configuration, dict):
        return "N/A"
    parameters = configuration.get("parameters", configuration)
    return json.dumps(parameters, sort_keys=True, separators=(", ", ": "))


def _optional_number(value: Any, *, digits: int = 4) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "N/A"
    return f"{numeric:.{digits}f}" if np.isfinite(numeric) else "N/A"


def _fill_table(table: QTableWidget, rows: list[tuple[Any, ...]]) -> None:
    for row_index, values in enumerate(rows):
        for column_index, value in enumerate(values):
            table.setItem(row_index, column_index, QTableWidgetItem(str(value)))


def _draw_candles(axes: Any, candles: pd.DataFrame) -> None:
    for position, candle in enumerate(candles.itertuples(index=False)):
        open_price = float(candle.open)
        high_price = float(candle.high)
        low_price = float(candle.low)
        close_price = float(candle.close)
        color = "white" if close_price >= open_price else "black"
        axes.vlines(position, low_price, high_price, color="black", linewidth=0.8)
        bottom = min(open_price, close_price)
        height = max(abs(close_price - open_price), 1e-9)
        axes.add_patch(Rectangle(
            (position - 0.32, bottom), 0.64, height,
            facecolor=color, edgecolor="black", linewidth=0.8,
        ))
    axes.set_xlim(-1, len(candles))


def _uncertainty_series(predictions: pd.DataFrame, *, head: str) -> np.ndarray:
    kind = str(predictions["uncertainty_kind"].iloc[0])
    value_kind = "score" if kind == "uncalibrated_decision_score" else "probability"
    columns = [f"{head}_{value_kind}_{regime}" for regime in ("bull", "bear", "range")]
    return predictions[columns].max(axis=1).to_numpy(dtype=float)


def _format_run_details(run: ComparisonRun) -> str:
    """Keep detail formatting independent from artifact loading and validation."""

    metrics = run.metrics
    transition = metrics["transitions"]
    lines = [
        run.display_name,
        f"Current accuracy: {metrics['current']['accuracy']:.4f}",
        f"Current macro-F1: {metrics['current']['macro_f1']:.4f}",
        f"Anticipated accuracy: {metrics['anticipated']['accuracy']:.4f}",
        f"Anticipated macro-F1: {metrics['anticipated']['macro_f1']:.4f}",
        f"Exact transitions: {transition['exact']['matched_transition_count']}",
        f"Transitions within 1 candle: {transition['within_one_candle']['matched_transition_count']}",
        f"Transitions within 2 candles: {transition['within_two_candles']['matched_transition_count']}",
    ]
    for head in ("current", "anticipated"):
        lines.append(f"{head.title()} per class:")
        for regime, values in metrics[head]["per_class"].items():
            lines.append(
                f"  {regime.title()}: P {values['precision']:.3f} | "
                f"R {values['recall']:.3f} | F1 {values['f1']:.3f}"
            )
        lines.append(f"  Confusion: {metrics[head]['confusion_matrix']}")
    return "\n".join(lines)


def _mean_std(values: list[float]) -> str:
    """Format seed aggregation without presenting one lucky seed as the model result."""

    import numpy as np

    numeric = np.asarray(values, dtype=float)
    return f"{numeric.mean():.4f} ± {numeric.std(ddof=0):.4f}"
