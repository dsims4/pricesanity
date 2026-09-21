import os
from types import SimpleNamespace
from typing import cast

import pandas as pd
import pytest
from matplotlib.patches import FancyBboxPatch, Rectangle

# Use Qt's nonvisual platform so chart tests never open a desktop window.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from pricesanity.gui.chart_widget import CandlestickChart
from pricesanity.gui.theme import PLOT_TEXT_COLOR


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
    assert len(chart.axes.texts) == 2
    assert chart.axes.texts[0].get_text() == ""
    assert chart.axes.texts[0].xy == (1, 103.0)
    assert not chart.axes.texts[1].get_visible()
    assert chart.time_axes.get_xlabel() == "Time of day (New York)"
    assert chart.axes.get_ylabel() == "Price (U.S. Dollars)"
    assert chart.axes.yaxis.get_ticks_position() == "right"
    assert chart.time_axes.get_xticklabels()[0].get_text() == "09:30"
    assert chart.axes.get_facecolor() == (1.0, 1.0, 1.0, 1.0)
    assert chart.timeline_axes.get_facecolor() != chart.axes.get_facecolor()
    assert chart.axes.yaxis.get_major_formatter()(100.0) == "100.00"
    assert chart.axes.yaxis.label.get_color() == PLOT_TEXT_COLOR
    assert chart.axes.get_yticklabels()[0].get_color() == PLOT_TEXT_COLOR
    assert isinstance(chart.axes.patch, FancyBboxPatch)
    assert chart.axes.patch.get_edgecolor() == (0.0, 0.0, 0.0, 1.0)
    assert not any(spine.get_visible() for spine in chart.axes.spines.values())
    assert not any(line.get_visible() for line in chart.axes.xaxis.get_ticklines())
    assert chart.time_axes.spines["bottom"].get_edgecolor() != (0.0, 0.0, 0.0, 1.0)
    assert any(
        line.get_visible() and line.get_markersize() > 0
        for line in chart.time_axes.xaxis.get_ticklines()
    )
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


def test_crosshair_snaps_to_candle_time_and_quarter_point_price(qt_application) -> None:
    """Pointer movement shows Trade Tank-style time and price locator boxes."""

    candles = pd.DataFrame({
        "ts_event": pd.date_range("2026-09-09T13:30:00Z", periods=3, freq="5min"),
        "open": [100.0] * 3,
        "high": [102.0] * 3,
        "low": [99.0] * 3,
        "close": [101.0] * 3,
    })
    chart = CandlestickChart()
    chart.draw_session(candles, 0)

    chart._update_crosshair(
        SimpleNamespace(inaxes=chart.axes, xdata=1.4, ydata=100.62)
    )

    assert chart._crosshair_vertical.get_xdata()[0] == 1.4
    assert chart._crosshair_horizontal.get_ydata()[0] == 100.5
    assert chart._crosshair_time_label.get_text() == "09:35"
    assert chart._crosshair_price_label.get_text() == "100.50"
    assert chart._crosshair_time_label.get_visible()
    assert chart._crosshair_price_label.get_visible()

    left_edge, right_edge = chart.axes.get_xlim()
    chart._update_crosshair(
        SimpleNamespace(inaxes=chart.axes, xdata=left_edge, ydata=100.62)
    )
    assert chart._crosshair_vertical.get_xdata()[0] == left_edge
    assert chart._crosshair_time_label.get_text() == "09:30"

    chart._update_crosshair(
        SimpleNamespace(inaxes=chart.axes, xdata=right_edge, ydata=100.62)
    )
    assert chart._crosshair_vertical.get_xdata()[0] == right_edge
    assert chart._crosshair_time_label.get_text() == "09:40"

    chart._hide_crosshair()
    assert not chart._crosshair_vertical.get_visible()
    assert not chart._crosshair_horizontal.get_visible()
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
    time_labels = [label.get_text() for label in chart.time_axes.get_xticklabels()]
    assert 5 <= len(time_labels) <= 8
    assert time_labels[0] == "09:30"
    assert time_labels[-1] == "16:00"
    assert 5 <= len(chart.axes.get_yticks()) <= 9
    price_ticks = chart.axes.get_yticks()
    assert price_ticks[0] > chart.axes.get_ylim()[0]
    assert price_ticks[-1] < chart.axes.get_ylim()[1]
    assert abs((price_ticks[0] * 4) - round(price_ticks[0] * 4)) < 1e-9
    assert abs((price_ticks[-1] * 4) - round(price_ticks[-1] * 4)) < 1e-9
    assert all((right - left) >= 0.25 for left, right in zip(price_ticks, price_ticks[1:]))
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
    strips = [
        patch for patch in chart.timeline_axes.patches
        if type(patch) is Rectangle
    ]
    left_edge, right_edge = chart.timeline_axes.get_xlim()
    assert strips[0].get_x() == left_edge
    assert strips[-1].get_x() + strips[-1].get_width() == right_edge
    assert all(
        spine.get_edgecolor() == (0.0, 0.0, 0.0, 1.0)
        for spine in chart.timeline_axes.spines.values()
    )
    assert chart.regime_change_markers == ((2, "bear"),)
    assert not any(line.get_visible() for line in chart.axes.lines)
    assert [
        label.get_text() for label in chart.axes.texts if label.get_visible()
    ] == [""]
    assert chart.time_axes.get_xlabel() == "Time of day (Chicago)"
    assert chart.time_axes.get_xticklabels()[0].get_text() == "08:30"
    assert [label.get_text() for label in chart.timeline_axes.get_yticklabels()] == [
        "MODEL · CURRENT"
    ]

    # Refreshing a constant day replaces prior spans and retains its starting regime.
    chart.set_regime_change_markers([], starting_regime="range")
    strips = [
        patch for patch in chart.timeline_axes.patches
        if type(patch) is Rectangle
    ]
    assert len(strips) == 1
    assert strips[-1].get_width() == 5
    assert any(
        isinstance(patch, FancyBboxPatch)
        for patch in chart.timeline_axes.patches
    )
    assert [
        label.get_text()
        for label in chart.timeline_axes.texts
        if label.get_visible()
    ] == []
    assert chart.regime_change_markers == ()
    chart.close()


def test_regime_labels_leave_unannotated_candles_blank(qt_application) -> None:
    """Human annotation strips preserve gaps instead of implying a regime through them."""

    candles = pd.DataFrame({
        "ts_event": pd.date_range("2026-09-09T13:30:00Z", periods=4, freq="5min"),
        "open": [100.0] * 4, "high": [102.0] * 4,
        "low": [99.0] * 4, "close": [101.0] * 4,
    })
    chart = CandlestickChart()
    chart.draw_session(candles, 0)
    chart.set_regime_labels(
        ["bull", "bull", None, "bear"],
        legend_title="Human current regime",
    )

    strips = [
        patch for patch in chart.timeline_axes.patches
        if type(patch) is Rectangle
    ]
    assert [(strip.get_x(), strip.get_width()) for strip in strips] == [
        (-1.0, 2.5),
        (1.5, 1),
        (2.5, 1.5),
    ]
    assert strips[1].get_hatch() == "///"
    assert chart.regime_labels == ("bull", "bull", None, "bear")
    assert chart.regime_change_markers == ()
    assert [label.get_text() for label in chart.timeline_axes.get_yticklabels()] == [
        "HUMAN CURRENT REGIME"
    ]
    legend = chart.time_axes.get_legend()
    assert {text.get_text() for text in legend.get_texts()} == {
        "Bull", "Bear", "Range", "Missing"
    }
    assert all(isinstance(handle, FancyBboxPatch) for handle in legend.legend_handles)
    assert all(
        handle.get_edgecolor() == (0.0, 0.0, 0.0, 1.0)
        for handle in legend.legend_handles
    )

    chart.close()
