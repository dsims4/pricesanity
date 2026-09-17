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


from pricesanity.gui.theme import (
    PLOT_LABEL_SIZE,
    PLOT_LEGEND_SIZE,
    PLOT_TICK_SIZE,
    PLOT_TIMELINE_SIZE,
    PLOT_TRACK_LABEL_SIZE,
    REGIME_COLORS,
)


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

        # Give price geometry and categorical timelines separate aligned regions. This keeps
        # bars, their labels, and clock ticks from competing for the bottom of the price plot.
        figure = Figure(figsize=(12, 7))
        grid = figure.add_gridspec(2, 1, height_ratios=(9, 1), hspace=0.08)
        self.axes = figure.add_subplot(grid[0, 0])
        self.timeline_axes = figure.add_subplot(grid[1, 0], sharex=self.axes)
        figure.subplots_adjust(left=0.13, right=0.95, top=0.88, bottom=0.15)

        # Retain the configured timestamp field and timezone for display labels.
        self.timestamp_column = timestamp_column
        self.session_timezone = session_timezone
        self._regime_artists = []
        self._active_marker: Annotation | None = None
        self._highs: tuple[float, ...] = ()
        self._arrow_offset = 0.0
        self._regime_change_markers: tuple[tuple[int, str], ...] = ()
        self._regime_labels: tuple[str | None, ...] = ()
        self._regime_tracks: tuple[tuple[str, tuple[str | None, ...]], ...] = ()
        self._tick_positions: tuple[int, ...] = ()
        self._tick_labels: tuple[str, ...] = ()
        self._time_axis_label = ""

        super().__init__(figure)

        self.setParent(parent)
        self.setMinimumHeight(360)

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
        self.timeline_axes.clear()
        for legend in list(self.figure.legends):
            legend.remove()
        self._regime_artists = []
        self._regime_change_markers = ()
        self._regime_labels = ()
        self._regime_tracks = ()

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
            fontsize=PLOT_TICK_SIZE,
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
        self._tick_positions = tuple(tick_positions)
        self._tick_labels = tuple(tick_labels)
        self.axes.set_xticks(tick_positions)
        self.axes.tick_params(axis="x", labelbottom=False)

        # Put session-local time below the chart and price values along its right
        # edge, matching the layout commonly used for market charts.
        display_timezone = self.session_timezone.rsplit("/", 1)[-1].replace("_", " ")
        self._time_axis_label = f"Time of day ({display_timezone})"
        self.axes.set_ylabel("Price", fontsize=PLOT_LABEL_SIZE)
        self.axes.yaxis.tick_right()
        self.axes.yaxis.set_label_position("right")
        self.axes.tick_params(axis="y", labelsize=PLOT_TICK_SIZE)
        self.axes.grid(axis="y", alpha=0.2)
        self._configure_timeline_axes(track_count=0)

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

    @property
    def regime_tracks(self) -> tuple[tuple[str, tuple[str | None, ...]], ...]:
        """Return every candle-aligned timeline row in display order."""

        return self._regime_tracks

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

        # Expand change points into candle-aligned spans. An absent starting regime leaves the
        # opening segment unknown rather than borrowing the first later prediction backward.
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

        self._regime_change_markers = tuple(validated_markers)
        self._draw_regime_tracks((("MODEL · CURRENT", regimes),))

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

        # Positional overlays must cover this exact chart population. Accepting a shorter list
        # could make omitted candles look unannotated instead of exposing alignment damage.
        if not self._highs:
            raise ValueError("A session must be drawn before adding regime labels.")
        if len(regimes) != len(self._highs):
            raise ValueError("Regime labels must match the displayed session length.")
        if any(regime not in {None, "bull", "bear", "range"} for regime in regimes):
            raise ValueError("Regime labels contain an unknown regime.")

        self._regime_change_markers = ()
        self._draw_regime_tracks(((legend_title.upper(), regimes),))

    def set_regime_tracks(
        self,
        tracks: Sequence[tuple[str, Sequence[str | None]]],
    ) -> None:
        """Show one or more exact candle-aligned model or human label timelines."""

        if not self._highs:
            raise ValueError("A session must be drawn before adding regime tracks.")
        if not tracks:
            raise ValueError("At least one regime track is required.")

        validated = []
        for label, regimes in tracks:
            values = tuple(regimes)
            if not label.strip():
                raise ValueError("Regime track labels must be nonempty.")
            if len(values) != len(self._highs):
                raise ValueError("Regime tracks must match the displayed session length.")
            if any(value not in {None, "bull", "bear", "range"} for value in values):
                raise ValueError("Regime tracks contain an unknown regime.")
            validated.append((label, values))
        self._draw_regime_tracks(tuple(validated))

    def _draw_regime_tracks(
        self,
        tracks: Sequence[tuple[str, Sequence[str | None]]],
    ) -> None:
        """Replace timeline rows with exact categorical spans, including missing labels."""

        self.timeline_axes.clear()
        for legend in list(self.figure.legends):
            legend.remove()
        self._regime_artists = []
        self._regime_tracks = tuple(
            (label, tuple(regimes)) for label, regimes in tracks
        )
        self._regime_labels = self._regime_tracks[0][1]
        self._configure_timeline_axes(track_count=len(tracks))
        colors = {**REGIME_COLORS, None: "#d8e0e5"}
        names = {"bull": "Bull", "bear": "Bear", "range": "Range", None: "Missing"}

        for track_index, (_, regimes) in enumerate(tracks):
            row = len(tracks) - track_index - 1
            start = 0
            while start < len(regimes):
                regime = regimes[start]
                end = start + 1
                while end < len(regimes) and regimes[end] == regime:
                    end += 1
                strip = Rectangle(
                    (start - 0.5, row + 0.08),
                    width=end - start,
                    height=0.84,
                    facecolor=colors[regime],
                    edgecolor="white",
                    linewidth=0.8,
                    hatch="///" if regime is None else None,
                )
                self.timeline_axes.add_patch(strip)
                self._regime_artists.append(strip)
                if end - start >= 3:
                    label = self.timeline_axes.text(
                        (start + end - 1) / 2,
                        row + 0.5,
                        names[regime],
                        ha="center",
                        va="center",
                        fontsize=PLOT_TIMELINE_SIZE,
                        fontweight="bold",
                        color=(
                            "#425966"
                            if regime is None
                            else "black" if regime == "range" else "white"
                        ),
                        clip_on=True,
                    )
                    self._regime_artists.append(label)
                start = end

        legend = self.figure.legend(
            handles=[
                Rectangle(
                    (0, 0), 1, 1,
                    facecolor=colors[regime],
                    edgecolor="#8799a5" if regime is None else "none",
                    hatch="///" if regime is None else None,
                    label=names[regime],
                )
                for regime in ("bull", "bear", "range", None)
            ],
            loc="upper center",
            bbox_to_anchor=(0.52, 0.995),
            ncol=4,
            frameon=False,
            fontsize=PLOT_LEGEND_SIZE,
        )
        self._regime_artists.append(legend)
        self.draw_idle()

    def _configure_timeline_axes(self, *, track_count: int) -> None:
        """Restore the shared candle/time scale after clearing timeline artists."""

        self.timeline_axes.set_xlim(-1, len(self._highs))
        self.timeline_axes.set_xticks(self._tick_positions, self._tick_labels)
        self.timeline_axes.set_xlabel(
            self._time_axis_label,
            fontsize=PLOT_LABEL_SIZE,
        )
        self.timeline_axes.set_ylim(0, max(track_count, 1))
        if track_count:
            labels = [label for label, _ in reversed(self._regime_tracks)]
            # During a redraw, use the incoming labels before _regime_tracks is replaced.
            if len(labels) != track_count:
                labels = [""] * track_count
            self.timeline_axes.set_yticks(
                [position + 0.5 for position in range(track_count)], labels
            )
        else:
            self.timeline_axes.set_yticks([])
        self.timeline_axes.tick_params(axis="x", labelsize=PLOT_TICK_SIZE)
        self.timeline_axes.tick_params(axis="y", labelsize=PLOT_TRACK_LABEL_SIZE)
        self.timeline_axes.grid(False)
        for spine in self.timeline_axes.spines.values():
            spine.set_color("#9fb1bd")
            spine.set_linewidth(1.0)

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
