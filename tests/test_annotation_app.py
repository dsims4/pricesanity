"""Tests for the candlestick annotation window."""

import os

import pandas as pd
import pytest

# Use Qt without opening a visible window during automated tests.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QDate, Qt
from PySide6.QtGui import QColor
from PySide6.QtTest import QTest
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QSizePolicy,
    QSpinBox,
    QToolButton,
)

from pricesanity.annotation.schema import MarketRegime
from pricesanity.annotation.store import load_annotation
from pricesanity.gui.annotation_app import AnnotationWindow


@pytest.fixture(scope="module")
def qt_application() -> QApplication:
    """Reuse the Qt application required by the widget tests."""

    return QApplication.instance() or QApplication([])


@pytest.fixture
def candlestick_data() -> pd.DataFrame:
    """Provide a short session for testing navigation boundaries."""

    return pd.DataFrame(
        {
            "candlestick_id": ["candle-1", "candle-2", "candle-3"],
            "session_date": ["2026-09-09"] * 3,
            "ts_event": pd.date_range(
                "2026-09-09 13:30:00",
                periods=3,
                freq="5min",
                tz="UTC",
            ),
            "open_gap": [0.001, -0.002, 0.0],
            "open": [100.0, 101.0, 100.5],
            "high": [101.5, 102.0, 101.0],
            "low": [99.5, 100.0, 99.0],
            "close": [101.0, 100.5, 99.5],
        }
    )


def test_annotation_window_moves_between_candlesticks(
    qt_application: QApplication,
    candlestick_data: pd.DataFrame,
    tmp_path,
) -> None:
    """Navigation changes the active candle without leaving the session."""

    window = AnnotationWindow(candlestick_data, tmp_path / "annotations.db")

    assert window.opening_gap_value.text() == "+0.100%"

    window.move_to_next_candlestick()

    assert window.active_candlestick_position == 1
    assert window.opening_gap_value.text() == "-0.200%"

    window.move_to_previous_candlestick()

    assert window.active_candlestick_position == 0
    assert window.opening_gap_value.text() == "+0.100%"

    # Moving backward from the first candle must leave it selected.
    window.move_to_previous_candlestick()

    assert window.active_candlestick_position == 0

    # Moving forward repeatedly must stop at the final candle.
    for _ in range(len(candlestick_data) + 1):
        window.move_to_next_candlestick()

    assert window.active_candlestick_position == len(candlestick_data) - 1

    window.close()


def test_regime_hotkeys_save_then_restore_both_choices(
    qt_application: QApplication,
    candlestick_data: pd.DataFrame,
    tmp_path,
) -> None:
    """Two number keys save a candle and navigation restores its choices."""

    database_path = tmp_path / "annotations.db"
    window = AnnotationWindow(candlestick_data, database_path)

    # Untouched candles begin without labels or implied defaults.
    assert window.current_regime_value.text() == "Not selected"
    assert window.anticipated_regime_value.text() == "Not selected"
    assert window.selection_prompt.text().startswith("Step 1 of 2")
    assert "press 1 for Bull" in window.selection_prompt.text()

    # The first key fills only the current-regime scalar.
    window.regime_shortcuts["1"].activated.emit()

    assert window.current_regime_value.text() == "1 - Bull"
    assert window.anticipated_regime_value.text() == "Not selected"
    assert window.selection_prompt.text().startswith("Step 2 of 2")
    assert "saves both choices" in window.selection_prompt.text()
    assert load_annotation(database_path, "candle-1") is None

    # The second key completes, saves, and advances the annotation.
    window.regime_shortcuts["2"].activated.emit()

    assert window.active_candlestick_position == 1
    assert window.current_regime_value.text() == "Not selected"
    assert window.anticipated_regime_value.text() == "Not selected"

    saved_annotation = load_annotation(database_path, "candle-1")

    assert saved_annotation is not None
    assert saved_annotation.current_regime is MarketRegime.BULL
    assert saved_annotation.anticipated_regime is MarketRegime.BEAR

    # Returning to the completed candle restores both saved choices.
    window.move_to_previous_candlestick()

    assert window.current_regime_value.text() == "1 - Bull"
    assert window.anticipated_regime_value.text() == "2 - Bear"

    window.close()


def test_annotation_window_shortcuts_control_navigation(
    qt_application: QApplication,
    candlestick_data: pd.DataFrame,
    tmp_path,
) -> None:
    """Arrow shortcuts call the matching candle-navigation actions."""

    window = AnnotationWindow(candlestick_data, tmp_path / "annotations.db")

    # Confirm the displayed shortcut names before exercising their signals.
    assert window.previous_candlestick_shortcut.key().toString() == "Left"
    assert window.next_candlestick_shortcut.key().toString() == "Right"
    assert window.previous_candlestick_shortcut.parent() is window.chart
    assert window.previous_candlestick_shortcut.context() == Qt.ShortcutContext.WidgetShortcut

    # Emit the Qt signals directly so the test does not depend on window focus.
    window.next_candlestick_shortcut.activated.emit()

    assert window.active_candlestick_position == 1

    window.previous_candlestick_shortcut.activated.emit()

    assert window.active_candlestick_position == 0

    window.close()


def test_chart_and_date_inputs_receive_click_focus(
    qt_application: QApplication,
    candlestick_data: pd.DataFrame,
    tmp_path,
) -> None:
    """Mouse focus separates chart shortcuts from typeable date fields."""

    window = AnnotationWindow(candlestick_data, tmp_path / "annotations.db")
    window.show()

    # Both calendar controls remain editable as well as popup-clickable.
    assert not window.start_date_input.isReadOnly()
    assert not window.end_date_input.isReadOnly()

    # Clicking the chart activates its widget-scoped annotation shortcuts.
    QTest.mouseClick(window.chart, Qt.MouseButton.LeftButton)
    qt_application.processEvents()

    assert window.chart.hasFocus()

    # Clicking a date field transfers focus away from chart shortcuts.
    QTest.mouseClick(window.start_date_input, Qt.MouseButton.LeftButton)
    qt_application.processEvents()

    assert window.start_date_input.hasFocus()
    assert not window.chart.hasFocus()
    assert window.chart_frame.frameShape() == QFrame.Shape.Box

    # Each lower value begins directly below its label, and the two compact
    # columns stay together at the left edge instead of spanning the window.
    assert window.current_regime_label.x() == window.current_regime_value.x()
    assert window.anticipated_regime_label.x() == window.anticipated_regime_value.x()
    assert window.anticipated_regime_value.x() - window.current_regime_value.x() < 180

    window.close()


def test_date_inputs_are_compact_and_enforce_corpus_boundaries(
    qt_application: QApplication,
    candlestick_data: pd.DataFrame,
    tmp_path,
) -> None:
    """Compact date dropdowns reject dates without validated sessions."""

    corpus_start = QDate(2026, 9, 8)
    corpus_end = QDate(2026, 9, 10)
    window = AnnotationWindow(
        candlestick_data,
        tmp_path / "annotations.db",
        corpus_start_date=corpus_start.toPython(),
        corpus_end_date=corpus_end.toPython(),
    )

    # Both date controls must retain the same bounded month-and-year navigation behavior.
    for date_input in (window.start_date_input, window.end_date_input):
        calendar = date_input.calendarWidget()

        assert date_input.sizePolicy().horizontalPolicy() == QSizePolicy.Policy.Fixed
        assert date_input.width() == 150
        assert calendar.isNavigationBarVisible()
        assert date_input.minimumDate() == corpus_start
        assert date_input.maximumDate() == corpus_end
        assert calendar.minimumDate() == corpus_start
        assert calendar.maximumDate() == corpus_end
        assert calendar.dateTextFormat(QDate(2026, 9, 7)).foreground().color() == QColor("red")
        assert calendar.dateTextFormat(QDate(2026, 9, 8)).foreground().color() == QColor("red")
        assert calendar.dateTextFormat(QDate(2026, 9, 9)).foreground().color() != QColor("red")
        assert calendar.dateTextFormat(QDate(2026, 9, 10)).foreground().color() == QColor("red")
        assert calendar.dateTextFormat(QDate(2026, 9, 11)).foreground().color() == QColor("red")

        month_button = calendar.findChild(
            QToolButton,
            "qt_calendar_monthbutton",
        )
        year_button = calendar.findChild(
            QToolButton,
            "qt_calendar_yearbutton",
        )
        year_editor = calendar.findChild(QSpinBox, "qt_calendar_yearedit")

        assert month_button is not None and month_button.menu() is not None
        assert year_button is not None and year_button.menu() is not None
        assert year_editor is not None and year_editor.isHidden()
        assert calendar.minimumWidth() == 300
        assert month_button.width() == 105
        assert year_button.width() == 75

        # Dates outside the corpus are first clamped by Qt, then unavailable
        # dates inside it are returned to the last validated session.
        date_input.setDate(QDate(2026, 9, 7))

        assert date_input.date() == QDate(2026, 9, 9)
        date_input.setDate(QDate(2026, 9, 11))

        assert date_input.date() == QDate(2026, 9, 9)

    window.close()


def test_stop_button_closes_annotation_window(
    qt_application: QApplication,
    candlestick_data: pd.DataFrame,
    tmp_path,
) -> None:
    """The Stop button closes the annotation run and its database store."""

    window = AnnotationWindow(candlestick_data, tmp_path / "annotations.db")
    window.resize(1000, 700)
    window.show()
    qt_application.processEvents()

    assert window.stop_button.x() > window.width() // 2

    QTest.mouseClick(window.stop_button, Qt.MouseButton.LeftButton)
    qt_application.processEvents()

    assert not window.isVisible()


def test_date_range_stays_open_while_navigation_crosses_sessions(
    qt_application: QApplication,
    tmp_path,
) -> None:
    """The window changes dates and crosses session boundaries in one run."""

    candlestick_data = pd.DataFrame(
        {
            "candlestick_id": ["day-1-a", "day-1-b", "day-2-a", "day-2-b"],
            "session_date": [
                "2026-09-09",
                "2026-09-09",
                "2026-09-10",
                "2026-09-10",
            ],
            "ts_event": pd.to_datetime(
                [
                    "2026-09-09T13:30:00Z",
                    "2026-09-09T13:35:00Z",
                    "2026-09-10T13:30:00Z",
                    "2026-09-10T13:35:00Z",
                ]
            ),
            "open_gap": [0.001, 0.002, -0.003, 0.0],
            "open": [100.0, 101.0, 102.0, 103.0],
            "high": [101.0, 102.0, 103.0, 104.0],
            "low": [99.0, 100.0, 101.0, 102.0],
            "close": [100.5, 101.5, 102.5, 103.5],
        }
    )
    window = AnnotationWindow(
        candlestick_data,
        tmp_path / "annotations.db",
    )

    # Moving beyond one session's final candle opens the next selected date.
    window.move_to_next_candlestick()
    window.move_to_next_candlestick()

    assert window.active_session_position == 1
    assert window.active_candlestick_position == 0
    assert "2026-09-10" in window.session_position_label.text()

    # Moving backward from an opening candle returns to the prior session end.
    window.move_to_previous_candlestick()

    assert window.active_session_position == 0
    assert window.active_candlestick_position == 1

    # Applying a smaller range reuses this window and begins on its first date.
    second_date = QDate(2026, 9, 10)
    window.start_date_input.setDate(second_date)
    window.end_date_input.setDate(second_date)
    window.apply_date_range()

    assert window.selected_session_dates[0].isoformat() == "2026-09-10"
    assert len(window.selected_session_dates) == 1
    assert window.active_candlestick_position == 0

    window.close()


def test_autosave_advances_from_one_session_to_the_next(
    qt_application: QApplication,
    tmp_path,
) -> None:
    """Completing a session's final candle opens the next selected session."""

    candlestick_data = pd.DataFrame(
        {
            "candlestick_id": ["day-1", "day-2"],
            "session_date": ["2026-09-09", "2026-09-10"],
            "ts_event": pd.to_datetime(["2026-09-09T13:30:00Z", "2026-09-10T13:30:00Z"]),
            "open_gap": [0.001, -0.002],
            "open": [100.0, 101.0],
            "high": [101.0, 102.0],
            "low": [99.0, 100.0],
            "close": [100.5, 101.5],
        }
    )
    window = AnnotationWindow(
        candlestick_data,
        tmp_path / "annotations.db",
    )

    window.regime_shortcuts["3"].activated.emit()
    window.regime_shortcuts["1"].activated.emit()

    assert window.active_session_position == 1
    assert window.active_candlestick_position == 0
    assert window.current_regime_value.text() == "Not selected"
    assert window.anticipated_regime_value.text() == "Not selected"

    window.close()


def test_failed_save_stays_on_candle_and_can_retry(
    qt_application, candlestick_data, tmp_path, monkeypatch
) -> None:
    """Verify failed save stays on candle and can retry."""

    import sqlite3

    window = AnnotationWindow(candlestick_data, tmp_path / "annotations.db")
    original_save = window.annotation_store.save

    def fail_save(annotation):
        """Inject a save failure so navigation cannot advance past an unsaved judgment."""

        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(window.annotation_store, "save", fail_save)
    window.select_regime(MarketRegime.BULL)
    window.select_regime(MarketRegime.BEAR)

    assert window.active_candlestick_position == 0
    assert "Not saved" in window.statusBar().currentMessage()
    assert window.annotation_store.load("candle-1") is None
    monkeypatch.setattr(window.annotation_store, "save", original_save)
    window.select_regime(MarketRegime.BEAR)

    assert window.active_candlestick_position == 1
    assert window.annotation_store.load("candle-1").anticipated_regime is MarketRegime.BEAR
    assert window.statusBar().currentMessage() == ""
    window.close()

    # Closing the window must release its retained SQLite connection.
    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        window.annotation_store.load("candle-1")


def test_candle_navigation_does_not_rebuild_session(
    qt_application, candlestick_data, tmp_path, monkeypatch
) -> None:
    """Verify candle navigation does not rebuild session."""

    window = AnnotationWindow(candlestick_data, tmp_path / "annotations.db")

    def unexpected_rebuild(*args):
        """Fail if changing date bounds rebuilds the already initialized calendar controls."""

        pytest.fail("Navigation rebuilt unchanged session geometry")

    monkeypatch.setattr(window.chart, "draw_session", unexpected_rebuild)
    window.move_to_next_candlestick()
    window.move_to_previous_candlestick()
    window.close()


def test_failed_load_does_not_offer_blank_labels_for_overwrite(
    qt_application, candlestick_data, tmp_path, monkeypatch
) -> None:
    """Verify failed load does not offer blank labels for overwrite."""

    import sqlite3

    window = AnnotationWindow(candlestick_data, tmp_path / "annotations.db")
    original_load = window.annotation_store.load

    def fail_load(candlestick_id):
        """Inject a read failure to verify that an unknown saved judgment cannot be overwritten."""

        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(window.annotation_store, "load", fail_load)
    window.move_to_next_candlestick()

    assert window.current_regime_value.text() == "Unavailable"
    assert "Could not load" in window.statusBar().currentMessage()
    window.select_regime(MarketRegime.BULL)
    window.select_regime(MarketRegime.BEAR)

    assert window.active_candlestick_position == 1
    assert original_load("candle-2") is None
    monkeypatch.setattr(window.annotation_store, "load", original_load)
    window.apply_date_range()

    assert window.current_regime_value.text() == "Not selected"
    assert window.statusBar().currentMessage() == ""
    window.close()
