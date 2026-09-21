"""Behavior and accessibility checks for the shared presentation system."""

import os
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pandas as pd
import pytest
from PySide6.QtCore import QRect
from PySide6.QtWidgets import QAbstractItemView, QPushButton

from pricesanity.gui.annotation_app import AnnotationWindow
from pricesanity.gui.model_comparison import ModelComparisonWindow, _ABSessionCanvas
from pricesanity.gui.theme import STYLE, apply_theme, regime_icon


def test_shared_theme_uses_native_fonts_icons_and_readonly_tables(qt_application):
    """Shared visuals retain native typography, semantic icons, and read-only results."""

    assert "font-family" not in STYLE
    assert "font-size: 14px" in STYLE
    assert 'QFrame[role="chart"]' in STYLE
    assert "QCalendarWidget" in STYLE

    icon = regime_icon("bull", device_pixel_ratio=2)
    assert not icon.isNull()

    window = ModelComparisonWindow(())
    try:
        assert (
            window.leaderboard.editTriggers()
            == QAbstractItemView.EditTrigger.NoEditTriggers
        )
        assert window.property("pricesanityTheme") == "shared-v1"
        assert window.minimumWidth() <= 900
        assert window.minimumHeight() <= 640
    finally:
        window.close()


def test_annotation_layout_prioritizes_chart_at_compact_and_large_sizes(
    qt_application,
    candlestick_data,
    tmp_path,
):
    """The shared theme keeps controls usable while added height belongs to the chart."""

    window = AnnotationWindow(candlestick_data, tmp_path / "scaled.db")

    class SmallScreen:
        def availableGeometry(self):
            return QRect(0, 0, 1280, 720)

    try:
        # Qt reports logical pixels after display scaling. Simulate that reduced workspace so
        # the test protects the compact branch without requiring a particular monitor.
        with patch.object(
            AnnotationWindow,
            "screen",
            return_value=SmallScreen(),
        ):
            apply_theme(window)
        window.resize(1280, 700)
        window.show()
        qt_application.processEvents()
        compact_height = window.chart_frame.height()

        assert window.height() <= 700
        assert window.minimumSize().width() == 760
        assert window.minimumSize().height() == 480
        assert all(
            button.minimumHeight() >= 30
            for button in window.findChildren(QPushButton)
        )
        assert window.property("pricesanityTheme") == "shared-v1"
        assert window.current_card is not window.anticipated_card

        window.resize(1400, 900)
        qt_application.processEvents()

        assert compact_height >= 240
        assert window.chart_frame.height() > compact_height
        assert window.chart_frame.height() > window.current_card.height() * 2
        assert all(
            button.minimumHeight() >= 44
            for button in window.findChildren(QPushButton)
        )
    finally:
        window.close()


@pytest.mark.parametrize("size", [(720, 260), (920, 300), (1240, 450)])
def test_scores_get_separate_labeled_axes(qt_application, size):
    """Probabilities and uncalibrated scores must not share a misleading visual scale."""

    frame = pd.DataFrame(
        {
            "session_date": ["2026-01-05"] * 3,
            "uncertainty_kind": ["probability_estimate"] * 3,
        }
    )
    for kind in ("human", "predicted"):
        for head in ("current", "anticipated"):
            frame[f"{kind}_{head}_regime"] = ["bull", "range", "bear"]
    for head in ("current", "anticipated"):
        for regime in ("bull", "bear", "range"):
            frame[f"{head}_probability_{regime}"] = [0.3, 0.3, 0.4]

    # Put probability estimates and uncalibrated margins in one comparison. Sharing an axis
    # would falsely imply that both numerical scales have the same confidence semantics.
    scores = frame.copy()
    scores["uncertainty_kind"] = "uncalibrated_decision_score"
    for head in ("current", "anticipated"):
        for regime in ("bull", "bear", "range"):
            scores[f"{head}_score_{regime}"] = [-1.0, 2.0, 0.5]

    canvas = _ABSessionCanvas()
    try:
        canvas.show_session(
            frame,
            scores,
            first_name="A",
            second_name="B",
            ohlc=None,
        )
        canvas.resize(*size)
        canvas.show()
        qt_application.processEvents()
        canvas.draw()
        renderer = canvas.get_renderer()
        for axis in canvas.figure.axes:
            assert axis.bbox.height >= 12
            for label in axis.get_yticklabels() + axis.get_xticklabels():
                if label.get_visible():
                    bounds = label.get_window_extent(renderer)
                    assert bounds.x0 >= 0
                    assert bounds.y0 >= 0
        assert len(canvas.figure.axes) == 4
        assert "probability" in canvas.figure.axes[2].get_ylabel()
        assert "uncalibrated score" in canvas.figure.axes[3].get_ylabel()
        assert canvas.figure.axes[2].get_ylim() == (0.0, 1.0)
    finally:
        canvas.close()


def test_populated_leaderboard_keeps_cells_in_their_rows(
    qt_application,
    monkeypatch,
    tmp_path,
):
    """Summary-only leaderboard loading must preserve every model's row alignment."""

    import pricesanity.gui.model_comparison as comparison
    from pricesanity.benchmark.metrics import evaluate_benchmark_predictions
    from pricesanity.gui.model_comparison import ComparisonRun
    from test_benchmark_artifacts import _identity

    metrics = evaluate_benchmark_predictions(
        human_current=np.array([0, 1]),
        predicted_current=np.array([0, 0]),
        human_anticipated=np.array([0, 2]),
        predicted_anticipated=np.array([0, 0]),
        session_indices=np.array([0, 0]),
    ).to_dict()
    runs = []
    for index, name in enumerate(
        ("logistic_regression", "gaussian_naive_bayes")
    ):
        identity = {
            **_identity().to_dict(),
            "model_name": name,
            "run_name": "final_seed_42",
            "model_configuration_sha256": str(index),
        }
        runs.append(
            ComparisonRun(
                name,
                tmp_path / name,
                {"identity": identity},
                metrics,
            )
        )

    # Leaderboard construction should use summary artifacts only. Loading full predictions
    # here would make opening the window scale with every saved candle in every run.
    def forbidden(*args, **kwargs):
        raise AssertionError("Leaderboard read predictions eagerly")

    monkeypatch.setattr(
        comparison,
        "load_aligned_comparison_predictions",
        forbidden,
    )
    window = ModelComparisonWindow(tuple(runs))
    try:
        assert window.leaderboard.rowCount() == 2
        for row in range(2):
            assert all(
                window.leaderboard.item(row, column) is not None
                for column in range(15)
            )
        assert {
            window.leaderboard.item(row, 1).text() for row in range(2)
        } == {"logistic_regression", "gaussian_naive_bayes"}
    finally:
        window.close()


@pytest.mark.parametrize("size", [(760, 480), (960, 540), (1280, 720),
                                 (1440, 900), (1920, 1080)])
def test_supported_window_sizes(qt_application, candlestick_data, tmp_path, size):
    """Actual layout geometry must fit, not merely advertise a small minimum size."""

    windows = [
        AnnotationWindow(candlestick_data, tmp_path / "layout.db"),
        ModelComparisonWindow(()),
    ]
    try:
        for window in windows:
            window.resize(1920, 1080)
            window.show()
            qt_application.processEvents()
            window.resize(*size)
            qt_application.processEvents()
            qt_application.processEvents()
            assert (window.width(), window.height()) == size
            for button in window.findChildren(QPushButton):
                if button.isVisible():
                    bounds = button.mapTo(window, button.rect().bottomRight())
                    assert bounds.x() < window.width()
                    assert bounds.y() < window.height()
            if hasattr(window, "chart"):
                chart = window.chart
                chart.draw()
                renderer = chart.get_renderer()
                for text in [chart.time_axes.xaxis.label, *chart.time_axes.get_xticklabels()]:
                    bounds = text.get_window_extent(renderer)
                    assert bounds.y0 >= 0
                    assert bounds.x0 >= 0
                    assert bounds.x1 <= chart.figure.bbox.width
                assert chart.axes.bbox.height >= 65
            else:
                window.tabs.setCurrentIndex(1)
                qt_application.processEvents()
                assert window.comparison_chart.height() >= 220
    finally:
        for window in windows:
            window.close()


@pytest.mark.parametrize("entry", ["annotation", "test_results", "model_comparison"])
def test_launchers_maximize_without_fullscreen(monkeypatch, entry):
    """Each real CLI path requests an ordinary maximized window after loading its input."""

    from importlib import import_module
    from types import SimpleNamespace
    from unittest.mock import Mock

    launcher = import_module(f"pricesanity.gui.{entry}_launcher")
    window = Mock()
    application = Mock()
    application.exec.return_value = 0
    monkeypatch.setattr(launcher, "QApplication", SimpleNamespace(instance=lambda: application))
    if entry == "model_comparison":
        monkeypatch.setattr(launcher, "load_comparison_runs", lambda path: ())
        monkeypatch.setattr(launcher, "ModelComparisonWindow", lambda *args: window)
        arguments = []
    else:
        config = SimpleNamespace(data=SimpleNamespace(timestamp_column="ts_event"))
        monkeypatch.setattr(launcher, "load_config", lambda path: config)
        arguments = ["--config", "config.yaml"]
        if entry == "annotation":
            monkeypatch.setattr(launcher, "parse_corpus_date_bounds", lambda path: (None, None))
            monkeypatch.setattr(launcher, "load_annotation_sessions", lambda *a, **k: None)
            monkeypatch.setattr(launcher, "AnnotationWindow", lambda *a, **k: window)
            arguments += ["--candlesticks", "ohlc.parquet", "--normalized", "norm.parquet"]
        else:
            monkeypatch.setattr(launcher, "load_test_run", lambda *a, **k: None)
            monkeypatch.setattr(launcher, "TestResultsWindow", lambda *a, **k: window)
            arguments += ["--run", "saved-run"]
    assert launcher.main(arguments) == 0
    window.showMaximized.assert_called_once_with()
    window.showFullScreen.assert_not_called()


def test_restored_geometry_uses_half_available_screen(qt_application):
    """Available logical geometry, rather than physical monitor resolution, sets the target."""

    window = ModelComparisonWindow(())
    try:
        screen = type("Screen", (), {"availableGeometry": lambda self: QRect(0, 0, 1920, 1080)})()
        with patch.object(ModelComparisonWindow, "screen", return_value=screen):
            apply_theme(window)
        assert (window.width(), window.height()) == (960, 540)
    finally:
        window.close()


def test_chart_and_navigation_follow_the_window_tier(
    qt_application, candlestick_data, tmp_path,
):
    """Rendered bounds and text requirements protect the actual compact-window regression."""

    window = AnnotationWindow(candlestick_data, tmp_path / "compact.db")
    try:
        window.resize(760, 480)
        window.show()
        qt_application.processEvents()
        chart = window.chart
        chart.draw()
        renderer = chart.get_renderer()
        compact_font = chart.axes.get_yticklabels()[0].get_fontsize()
        assert compact_font == 8
        assert chart.axes.bbox.height > chart.figure.bbox.height * 0.60
        assert chart.axes.bbox.height > chart.timeline_axes.bbox.height * 4
        for label in [chart.axes.yaxis.label, *chart.axes.get_yticklabels(),
                      *chart.time_axes.get_xticklabels()]:
            bounds = label.get_window_extent(renderer)
            assert bounds.x0 >= 3
            assert bounds.y0 >= 3
            assert bounds.x1 <= chart.figure.bbox.width - 3
            assert bounds.y1 <= chart.figure.bbox.height - 3
        for button in window.findChildren(QPushButton):
            if button.property("responsiveNav"):
                assert button.width() >= button.sizeHint().width()
        assert window.previous_candle_button.text() == "Prev candle"
        assert chart.axes.get_ylabel() == "Price (USD)"
        assert not chart.time_axes.xaxis.label.get_visible()

        window.resize(1440, 900)
        qt_application.processEvents()
        chart.draw()
        assert chart.axes.get_yticklabels()[0].get_fontsize() > compact_font
        assert chart.time_axes.xaxis.label.get_visible()
        assert chart.axes.get_ylabel() == "Price (U.S. Dollars)"
        assert window.previous_candle_button.text() == "Previous candle"
    finally:
        window.close()
