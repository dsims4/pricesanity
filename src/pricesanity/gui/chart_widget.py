"""Interactive candlestick chart for retrospective annotation."""

from typing import cast

import pandas as pd
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.collections import LineCollection
from matplotlib.patches import Rectangle
from matplotlib.text import Annotation
from PySide6.QtCore import Qt
from PySide6.QtGui import QMouseEvent
from PySide6.QtWidgets import QWidget


class CandlestickChart(FigureCanvasQTAgg):
    """Matplotlib chart that can be placed inside a PySide6 window."""

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        timestamp_column: str = "ts_event",
    ) -> None:
        """Create an empty candlestick chart.

        Args:
            parent: PySide6 widget responsible for this chart.
            timestamp_column: Column containing UTC candle timestamps.
        """

        # Give this Qt widget its own figure so chart updates do not depend on global pyplot
        # state.
        figure = Figure(figsize=(12, 7), tight_layout=True)

        self.axes = figure.add_subplot(1, 1, 1)

        # Retain the configured timestamp name for New York time labels.
        self.timestamp_column = timestamp_column
        self._active_marker: Annotation | None = None
        self._highs: tuple[float, ...] = ()
        self._arrow_offset = 0.0

        super().__init__(figure)

        self.setParent(parent)

        # Allow a mouse click to give the chart keyboard focus so annotation
        # shortcuts remain inactive while the user is typing into date fields.
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        """Focus the chart before handling a mouse click.

        Args:
            event: Qt mouse event delivered to the chart canvas.
        """

        # Make the chart's shortcuts active immediately after it is clicked.
        self.setFocus(Qt.FocusReason.MouseFocusReason)
        super().mousePressEvent(event)

    def draw_session(
        self,
        candlestick_data: pd.DataFrame,
        active_candlestick_position: int,
    ) -> None:
        """Draw one complete session and identify the active candlestick.

        Args:
            candlestick_data: Chronological 5-minute OHLC candlesticks.
            active_candlestick_position: Zero-based candle being annotated.

        Raises:
            ValueError: If the active candle position is outside the session.
        """

        # Reject a position that cannot identify a candle in this session.
        if not 0 <= active_candlestick_position < len(candlestick_data):
            raise ValueError(
                "Active candlestick position is outside the session."
            )

        # Remove the previous chart before drawing the current session state.
        self.axes.clear()

        # Convert timestamps to New York time only for readable axis labels;
        # their stored UTC values and annotation identifiers remain unchanged.
        new_york_timestamps = pd.to_datetime(
            candlestick_data[self.timestamp_column],
            utc=True,
            errors="raise",
        ).dt.tz_convert("America/New_York")

        # Draw every candle because retrospective annotation uses the completed
        # session to assign the regimes that actually occurred.
        wick_segments = []

        # Place candles at consecutive positions so closed-market gaps do not stretch the
        # session chart.
        for x_position, candlestick in enumerate(
            candlestick_data.itertuples(index=False)
        ):
            # Narrow pandas' broad scalar types after the data pipeline has
            # already validated every OHLC field as a finite numeric value.
            open_price = cast(float, candlestick.open)
            high_price = cast(float, candlestick.high)
            low_price = cast(float, candlestick.low)
            close_price = cast(float, candlestick.close)

            # Use white for bullish candles and black for bearish candles.
            candle_color = "white" if close_price >= open_price else "black"

            body_bottom = min(open_price, close_price)
            body_top = max(open_price, close_price)

            # Stop each wick at the body edge so hollow candles stay clear.
            wick_segments.extend(
                [
                    [(x_position, low_price), (x_position, body_bottom)],
                    [(x_position, body_top), (x_position, high_price)],
                ]
            )

            body_height = abs(close_price - open_price)

            # Draw the open-to-close body around the horizontal position.
            candlestick_body = Rectangle(
                (x_position - 0.35, body_bottom),
                width=0.7,
                height=body_height,
                facecolor=candle_color,
                edgecolor="black",
            )
            self.axes.add_patch(candlestick_body)

        # One collection draws all wicks; candle bodies remain individual patches.
        self.axes.add_collection(
            LineCollection(
                wick_segments,
                colors="black",
                linewidths=1,
            )
        )
        self._highs = tuple(float(value) for value in candlestick_data["high"])
        active_high = self._highs[active_candlestick_position]

        # Measure the complete session range so the arrow remains visible at
        # different ES price levels and across quiet or volatile sessions.
        session_high = cast(float, candlestick_data["high"].max())
        session_low = cast(float, candlestick_data["low"].min())
        arrow_offset = max((session_high - session_low) * 0.08, 0.01)

        # Point to the candle currently selected for regime annotation.
        self._arrow_offset = arrow_offset
        self._active_marker = self.axes.annotate(
            "Active",
            xy=(active_candlestick_position, active_high),
            xytext=(
                active_candlestick_position,
                active_high + arrow_offset,
            ),
            horizontalalignment="center",
            color="tab:blue",
            arrowprops={"arrowstyle": "-|>", "color": "tab:blue"},
        )

        # Leave one empty candle position around the complete session chart.
        self.axes.set_xlim(-1, len(candlestick_data))

        # Extend the vertical limits so the active-candle arrow is not clipped.
        self.axes.set_ylim(
            session_low - arrow_offset,
            session_high + (2 * arrow_offset),
        )

        # Use a small set of evenly spaced clock labels so a full regular
        # session remains readable while the chart changes size.
        tick_interval = max(len(candlestick_data) // 8, 1)
        tick_positions = list(range(0, len(candlestick_data), tick_interval))
        final_position = len(candlestick_data) - 1

        # Include the final candle label even when the regular tick spacing misses it.
        if final_position not in tick_positions:
            tick_positions.append(final_position)

        tick_labels = [
            new_york_timestamps.iloc[position].strftime("%H:%M") for position in tick_positions
        ]
        self.axes.set_xticks(tick_positions, tick_labels)

        # Put New York time below the chart and price values along its right
        # edge, matching the layout commonly used for market charts.
        self.axes.set_xlabel("Time of day (New York)")
        self.axes.set_ylabel("Price")
        self.axes.yaxis.tick_right()
        self.axes.yaxis.set_label_position("right")
        self.axes.grid(axis="y", alpha=0.2)

        # Ask the existing Qt canvas to repaint after the chart has changed.
        self.draw_idle()

    def set_active_candlestick(self, position: int) -> None:
        """Move the marker without rebuilding the loaded session's artists.

        Args:
            position: Zero-based position of the active candlestick in the loaded session.
        """

        # A marker can move only after a session is loaded and the requested position exists.
        if self._active_marker is None or not 0 <= position < len(self._highs):
            raise ValueError("Active candlestick position is outside the session.")

        high = self._highs[position]
        self._active_marker.xy = (position, high)
        self._active_marker.set_position((position, high + self._arrow_offset))
        self.draw_idle()
