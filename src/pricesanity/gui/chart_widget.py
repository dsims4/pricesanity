"""Interactive candlestick chart for annotation and retrospective evaluation."""

from collections.abc import Sequence
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
        session_timezone: str = "America/New_York",
    ) -> None:
        """Create an empty candlestick chart.

        Args:
            parent: PySide6 widget responsible for this chart.
            timestamp_column: Column containing UTC candle timestamps.
            session_timezone: Configured timezone used only for display labels.
        """

        # Give this Qt widget its own figure so chart updates do not depend on global pyplot
        # state.
        figure = Figure(figsize=(12, 7), tight_layout=True)

        self.axes = figure.add_subplot(1, 1, 1)

        # Retain the configured timestamp field and timezone for display labels.
        self.timestamp_column = timestamp_column
        self.session_timezone = session_timezone
        self._regime_artists = []
        self._active_marker: Annotation | None = None
        self._highs: tuple[float, ...] = ()
        self._arrow_offset = 0.0
        self._regime_change_markers: tuple[tuple[int, str], ...] = ()
        self._regime_labels: tuple[str | None, ...] = ()

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
        self._regime_artists = []
        self._regime_change_markers = ()
        self._regime_labels = ()

        # Convert timestamps to session-local time only for readable axis labels;
        # their stored UTC values and annotation identifiers remain unchanged.
        display_timestamps = pd.to_datetime(
            candlestick_data[self.timestamp_column],
            utc=True,
            errors="raise",
        ).dt.tz_convert(self.session_timezone)

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
            display_timestamps.iloc[position].strftime("%H:%M") for position in tick_positions
        ]
        self.axes.set_xticks(tick_positions, tick_labels)

        # Put session-local time below the chart and price values along its right
        # edge, matching the layout commonly used for market charts.
        display_timezone = self.session_timezone.rsplit("/", 1)[-1].replace("_", " ")
        self.axes.set_xlabel(f"Time of day ({display_timezone})")
        self.axes.set_ylabel("Price")
        self.axes.yaxis.tick_right()
        self.axes.yaxis.set_label_position("right")
        self.axes.grid(axis="y", alpha=0.2)

        # Ask the existing Qt canvas to repaint after the chart has changed.
        self.draw_idle()

    @property
    def regime_change_markers(self) -> tuple[tuple[int, str], ...]:
        """Return the exact candle positions and new regimes drawn on the chart."""

        return self._regime_change_markers

    @property
    def regime_labels(self) -> tuple[str | None, ...]:
        """Return the regime label displayed for each candle, including blank positions."""

        return self._regime_labels

    def set_regime_change_markers(
        self,
        regime_change_markers: Sequence[tuple[int, str]],
        *,
        starting_regime: str | None = None,
    ) -> None:
        """Underline predicted regime spans without drawing lines through the prices.

        Args:
            regime_change_markers: Zero-based positions paired with each new regime.
            starting_regime: Optional first regime, shown from candle zero without a change.

        Raises:
            ValueError: If a marker is invalid or a session has not been drawn.
        """

        if not self._highs:
            raise ValueError("A session must be drawn before adding regime markers.")

        valid_regimes = {"bull", "bear", "range"}
        validated_markers = []
        for candle_position, regime in regime_change_markers:
            # Candle zero establishes the displayed starting regime. Treating it
            # as a change would imply a prior prediction from another session.
            if not 0 < candle_position < len(self._highs):
                raise ValueError("Regime-change marker is outside the valid session range.")
            if regime not in valid_regimes:
                raise ValueError("Regime-change marker contains an unknown regime.")
            validated_markers.append((candle_position, regime))

        if starting_regime is not None and starting_regime not in valid_regimes:
            raise ValueError("Starting regime is unknown.")
        positions = [position for position, _ in validated_markers]
        if positions != sorted(set(positions)):
            raise ValueError("Regime changes must have unique chronological positions.")

        regimes: list[str | None] = [None] * len(self._highs)
        span_starts = list(validated_markers)
        if starting_regime is not None:
            span_starts.insert(0, (0, starting_regime))
        for span_index, (starting_position, regime) in enumerate(span_starts):
            ending_position = span_starts[span_index + 1][0] if (
                span_index + 1 < len(span_starts)
            ) else len(regimes)
            regimes[starting_position:ending_position] = [regime] * (
                ending_position - starting_position
            )

        self._draw_regime_labels(regimes, legend_title="Model current regime")

    def set_regime_labels(
        self,
        regimes: Sequence[str | None],
        *,
        legend_title: str = "Current regime",
    ) -> None:
        """Underline known candle labels while leaving unsaved positions blank.

        Args:
            regimes: One Bull, Bear, Range, or absent label for every displayed candle.
            legend_title: Text distinguishing human annotations from model predictions.

        Raises:
            ValueError: If labels do not match the displayed session.
        """

        if not self._highs:
            raise ValueError("A session must be drawn before adding regime labels.")
        if len(regimes) != len(self._highs):
            raise ValueError("Regime labels must match the displayed session length.")
        if any(regime not in {None, "bull", "bear", "range"} for regime in regimes):
            raise ValueError("Regime labels contain an unknown regime.")

        self._draw_regime_labels(regimes, legend_title=legend_title)

    def _draw_regime_labels(
        self,
        regimes: Sequence[str | None],
        *,
        legend_title: str,
    ) -> None:
        """Replace the existing overlay with exact contiguous labeled spans."""

        # Replace overlays rather than accumulating artists after a save or session change.
        for artist in self._regime_artists:
            artist.remove()
        self._regime_artists = []

        regime_colors = {
            "bull": "tab:green",
            "bear": "tab:red",
            "range": "tab:orange",
        }
        short_labels = {"bull": "Bu", "bear": "Be", "range": "R"}
        spans: list[tuple[int, int, str]] = []
        change_markers: list[tuple[int, str]] = []
        starting_position = 0
        while starting_position < len(regimes):
            regime = regimes[starting_position]
            if regime is None:
                starting_position += 1
                continue

            ending_position = starting_position + 1
            while ending_position < len(regimes) and regimes[ending_position] == regime:
                ending_position += 1
            spans.append((starting_position, ending_position, regime))

            # A change label requires two adjacent known regimes. A span after an
            # annotation gap begins without inventing a transition across missing judgments.
            if starting_position > 0 and regimes[starting_position - 1] is not None:
                change_markers.append((starting_position, regime))
            starting_position = ending_position

        # A slim strip sits beneath the prices and retains exact candle-width boundaries.
        for starting_position, ending_position, regime in spans:
            strip = Rectangle(
                (starting_position - 0.5, 0.015),
                width=ending_position - starting_position,
                height=0.035,
                transform=self.axes.get_xaxis_transform(),
                facecolor=regime_colors[regime],
                edgecolor="white",
                linewidth=0.5,
                clip_on=True,
            )
            self.axes.add_patch(strip)
            self._regime_artists.append(strip)

            span_length = ending_position - starting_position
            bar_label = self.axes.text(
                (starting_position + ending_position - 1) / 2,
                0.0325,
                regime.title() if span_length >= 3 else short_labels[regime],
                transform=self.axes.get_xaxis_transform(),
                ha="center",
                va="center",
                fontsize=7,
                fontweight="bold",
                color="black" if regime == "range" else "white",
                clip_on=True,
            )
            self._regime_artists.append(bar_label)

        # Anchor each change label to its first candle. Alternating the text offset separates
        # nearby changes without restoring vertical lines through the price chart.
        for marker_index, (candle_position, regime) in enumerate(change_markers):
            change_label = self.axes.annotate(
                regime.title(),
                xy=(candle_position, self._highs[candle_position]),
                xytext=(0, 5 + 12 * (marker_index % 2)),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=7,
                color=regime_colors[regime],
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 1},
            )
            self._regime_artists.append(change_label)

        # A fixed legend identifies even one-candle spans without squeezing repeated text
        # into the plot. Candle zero is a starting span, never a fabricated change marker.
        if spans:
            legend = self.axes.legend(
                handles=[Rectangle((0, 0), 1, 1, facecolor=color,
                                   label=f"{regime.title()} ({short_labels[regime]})")
                         for regime, color in regime_colors.items()],
                loc="upper left",
                ncol=3,
                frameon=False,
                fontsize=9,
                title=legend_title,
            )
            self._regime_artists.append(legend)

        self._regime_change_markers = tuple(change_markers)
        self._regime_labels = tuple(regimes)
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
