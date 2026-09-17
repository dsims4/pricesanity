"""Behavior and accessibility checks for the shared presentation system."""

import os
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pandas as pd
from PySide6.QtCore import QRect
from PySide6.QtWidgets import QAbstractItemView, QPushButton

from pricesanity.annotation.schema import MarketRegime
from pricesanity.gui.annotation_app import AnnotationWindow
from pricesanity.gui.model_comparison import ModelComparisonWindow, _ABSessionCanvas
from pricesanity.gui.theme import STYLE, apply_theme, regime_icon
from test_annotation_app import candlestick_data, qt_application


def test_regime_buttons_save_pair_and_update_progress(
    qt_application,
    candlestick_data,
    tmp_path,
):
    window = AnnotationWindow(candlestick_data, tmp_path / "annotations.db")
    try:
        assert window.property("pricesanityTheme") == "shared-v1"
        assert all(
            button.minimumHeight() >= 40
            for button in window.findChildren(QPushButton)
        )
        assert window.current_card is not window.anticipated_card

        # The first click is intentionally incomplete: annotations are durable only after
        # both heads have values, so progress must not advance yet.
        window.regime_buttons["current", MarketRegime.BULL].click()
        assert window.regime_buttons["current", MarketRegime.BULL].isChecked()
        assert "choose anticipated" in window.commit_state.text()
        assert window.session_progress.value() == 0

        window.regime_buttons["anticipated", MarketRegime.RANGE].click()
        assert window.session_progress.value() == 1
        assert window.active_candlestick_position == 1

        # Revisiting a saved candle should restore the complete pair rather than presenting
        # an apparently blank form that could overwrite one side accidentally.
        window.move_to_previous_candlestick()
        assert "Pair saved" in window.commit_state.text()
        assert "0 / 1 sessions complete" in window.progress_label.text()
    finally:
        window.close()


def test_icons_and_readonly_tables(qt_application):
    icon = regime_icon("bull", device_pixel_ratio=2)
    assert not icon.isNull()

    window = ModelComparisonWindow(())
    try:
        assert (
            window.leaderboard.editTriggers()
            == QAbstractItemView.EditTrigger.NoEditTriggers
        )
        assert window.property("pricesanityTheme") == "shared-v1"
    finally:
        window.close()


def test_theme_inherits_qt_platform_font(qt_application):
    """Avoid asking Qt to resolve a generic family that may not be installed."""

    assert "font-family" not in STYLE
    assert "font-size: 13px" in STYLE


def test_annotation_fits_scaled_1080p(
    qt_application,
    candlestick_data,
    tmp_path,
):
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

        assert window.height() <= 700
        assert all(
            button.minimumHeight() >= 44
            for button in window.findChildren(QPushButton)
        )
    finally:
        window.close()


def test_scores_get_separate_labeled_axes(qt_application):
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
