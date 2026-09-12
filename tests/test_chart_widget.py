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
    # Reuse an existing application because Qt permits only one application
    # object within a process.
    application = QApplication.instance()
    if application is None:
        application = QApplication([])
    return cast(QApplication, application)


def test_draw_session_displays_all_candles_and_active_arrow(
    qt_application: QApplication,
) -> None:
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

    with pytest.raises(ValueError, match="outside the session"):
        chart.draw_session(candlestick_data, active_candlestick_position=1)

    chart.close()


def test_navigation_reuses_candles_and_axis_layout(qt_application) -> None:
    data = pd.DataFrame({
        "ts_event": pd.date_range("2026-09-09T13:30:00Z", periods=81, freq="5min"),
        "open": [100.] * 81, "high": [102.] * 81,
        "low": [99.] * 81, "close": [101.] * 81,
    })
    chart = CandlestickChart()
    chart.draw_session(data, 0)
    bodies = tuple(chart.axes.patches)
    wicks = tuple(chart.axes.collections)
    marker = chart.axes.texts[0]
    limits = (chart.axes.get_xlim(), chart.axes.get_ylim())
    for position in (1, 80, 0):
        chart.set_active_candlestick(position)
        assert marker.xy == (position, 102.)
        assert tuple(chart.axes.patches) == bodies
        assert tuple(chart.axes.collections) == wicks
        assert chart.axes.texts[0] is marker
        assert (chart.axes.get_xlim(), chart.axes.get_ylim()) == limits
    assert len(wicks) == 1
    chart.draw_session(data.iloc[:3], 2)
    assert len(chart.axes.patches) == 3
    assert chart.axes.texts[0] is not marker
    with pytest.raises(ValueError, match="outside"):
        chart.set_active_candlestick(3)
    chart.close()
