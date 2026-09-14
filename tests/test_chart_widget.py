import os
from typing import cast

import pandas as pd
import pytest

# Use Qt's nonvisual platform so chart tests never open a desktop window.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from pricesanity.gui.chart_widget import CandlestickChart


@pytest.fixture(scope="module")
def qt_application() -> QApplication:
    """Reuse the Qt application required by the widget tests."""

    # Reuse an existing application because Qt permits only one application
    # object within a process.
    application = QApplication.instance()

    # Reuse pytest's existing Qt application because Qt permits only one application per
    # process.
    if application is None:
        application = QApplication([])

    return cast(QApplication, application)


def test_draw_session_displays_all_candles_and_active_arrow(
    qt_application: QApplication,
) -> None:
    """Verify draw session displays all candles and active arrow."""

    # Include candles on both sides of the active position to prove the complete
    # retrospective session remains visible.
    candlestick_data = pd.DataFrame(
        {
            "ts_event": pd.date_range(
                "2026-09-09 13:30:00",
                periods=3,
                freq="5min",
                tz="UTC",
            ),
            "open": [100.0, 101.0, 100.5],
            "high": [102.0, 103.0, 102.0],
            "low": [99.0, 100.0, 98.0],
            "close": [101.0, 100.5, 99.0],
        }
    )
    chart = CandlestickChart()

    # Select the middle candle without removing the later candle from view.
    chart.draw_session(candlestick_data, active_candlestick_position=1)

    # Every candle has a body, and one annotation points to the active candle.
    assert len(chart.axes.patches) == 3
    assert len(chart.axes.texts) == 1
    assert chart.axes.texts[0].get_text() == "Active"
    assert chart.axes.texts[0].xy == (1, 103.0)
    assert chart.axes.get_xlabel() == "Time of day (New York)"
    assert chart.axes.get_ylabel() == "Price"
    assert chart.axes.yaxis.get_ticks_position() == "right"
    assert chart.axes.get_xticklabels()[0].get_text() == "09:30"
    assert chart.focusPolicy() == Qt.FocusPolicy.StrongFocus

    chart.close()


def test_draw_session_rejects_invalid_active_position(
    qt_application: QApplication,
) -> None:
    """Verify draw session rejects invalid active position."""

    # One candle permits only position zero as the active annotation target.
    candlestick_data = pd.DataFrame(
        {
            "ts_event": pd.to_datetime(["2026-09-09T13:30:00Z"]),
            "open": [100.0],
            "high": [101.0],
            "low": [99.0],
            "close": [100.5],
        }
    )
    chart = CandlestickChart()

    # The active marker cannot move to a position beyond the displayed session.
    with pytest.raises(ValueError, match="outside the session"):
        chart.draw_session(candlestick_data, active_candlestick_position=1)

    chart.close()


def test_navigation_reuses_candles_and_axis_layout(qt_application) -> None:
    """Verify navigation reuses candles and axis layout."""

    data = pd.DataFrame(
        {
            "ts_event": pd.date_range("2026-09-09T13:30:00Z", periods=81, freq="5min"),
            "open": [100.0] * 81,
            "high": [102.0] * 81,
            "low": [99.0] * 81,
            "close": [101.0] * 81,
        }
    )
    chart = CandlestickChart()
    chart.draw_session(data, 0)
    bodies = tuple(chart.axes.patches)
    wicks = tuple(chart.axes.collections)
    marker = chart.axes.texts[0]
    limits = (chart.axes.get_xlim(), chart.axes.get_ylim())

    # Exercise forward, long-distance, and backward moves without rebuilding the candle
    # collection.
    for position in (1, 80, 0):
        chart.set_active_candlestick(position)

        assert marker.xy == (position, 102.0)
        assert tuple(chart.axes.patches) == bodies
        assert tuple(chart.axes.collections) == wicks
        assert chart.axes.texts[0] is marker
        assert (chart.axes.get_xlim(), chart.axes.get_ylim()) == limits

    assert len(wicks) == 1
    chart.draw_session(data.iloc[:3], 2)

    assert len(chart.axes.patches) == 3
    assert chart.axes.texts[0] is not marker

    # An invalid move must leave the existing chart available rather than drawing an out-of-
    # range marker.
    with pytest.raises(ValueError, match="outside"):
        chart.set_active_candlestick(3)

    chart.close()


def test_regime_strips_include_start_and_replace_without_vertical_lines(qt_application) -> None:
    """Regime spans cover exact candle widths without obscuring price geometry."""

    candles = pd.DataFrame({
        "ts_event": pd.date_range("2026-09-09T13:30:00Z", periods=4, freq="5min"),
        "open": [100.0] * 4, "high": [102.0] * 4,
        "low": [99.0] * 4, "close": [101.0] * 4,
    })
    chart = CandlestickChart(session_timezone="America/Chicago")
    chart.draw_session(candles, 0)
    chart.set_regime_change_markers([(2, "bear")], starting_regime="bull")

    # The first strip starts at candle zero's left edge and ends at the next regime's edge.
    strips = list(chart.axes.patches)[4:]
    assert [(strip.get_x(), strip.get_width()) for strip in strips] == [(-0.5, 2), (1.5, 2)]
    assert chart.regime_change_markers == ((2, "bear"),)
    assert len(chart.axes.lines) == 0
    assert [label.get_text() for label in chart.axes.texts] == ["Active", "Bu", "Be", "Bear"]
    assert chart.axes.texts[-1].xy == (2, 102.0)
    assert chart.axes.get_xlabel() == "Time of day (Chicago)"
    assert chart.axes.get_xticklabels()[0].get_text() == "08:30"

    # Refreshing a constant day replaces prior spans and retains its starting regime.
    chart.set_regime_change_markers([], starting_regime="range")
    assert len(chart.axes.patches) == 5
    assert chart.axes.patches[-1].get_width() == 4
    assert [label.get_text() for label in chart.axes.texts] == ["Active", "Range"]
    assert chart.regime_change_markers == ()
    chart.close()
