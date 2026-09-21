"""Interactive candlestick chart for annotation and retrospective evaluation."""

from collections.abc import Sequence
from math import ceil, floor
from typing import cast

import pandas as pd
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.collections import LineCollection
from matplotlib.legend_handler import HandlerPatch
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.text import Annotation
from matplotlib.ticker import FormatStrFormatter, MaxNLocator
from PySide6.QtCore import Qt
from PySide6.QtGui import QMouseEvent
from PySide6.QtWidgets import QWidget


from pricesanity.gui.theme import (
    PLOT_AXES_COLOR,
    PLOT_FIGURE_COLOR,
    PLOT_LABEL_SIZE,
    PLOT_LEGEND_SIZE,
    PLOT_TICK_SIZE,
    PLOT_TRACK_LABEL_SIZE,
    PLOT_TEXT_COLOR,
    REGIME_COLORS,
)


def _rounded_legend_handle(
    legend,
    orig_handle,
    xdescent,
    ydescent,
    width,
    height,
    fontsize,
):
    """Keep regime legend samples rounded instead of Matplotlib's default rectangles."""

    return FancyBboxPatch(
        (-xdescent, -ydescent),
        width,
        height,
        boxstyle="round,pad=0.04,rounding_size=2",
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
        figure = Figure(figsize=(12, 7), facecolor=PLOT_FIGURE_COLOR)
        grid = figure.add_gridspec(
            3,
            1,
            height_ratios=(18, 1, 2.4),
            hspace=0.16,
        )
        self.axes = figure.add_subplot(grid[0, 0])
        self.timeline_axes = figure.add_subplot(grid[1, 0], sharex=self.axes)
        self.time_axes = figure.add_subplot(grid[2, 0], sharex=self.axes)
        figure.subplots_adjust(left=0.025, right=0.90, top=0.985, bottom=0.115)

        # Use the room freed below the price plot to move the complete lower chart stack up.
        lower_axis_shift = 0.020
        bar_position = self.timeline_axes.get_position()
        self.timeline_axes.set_position(
            [
                bar_position.x0,
                bar_position.y0 + lower_axis_shift,
                bar_position.width,
                bar_position.height,
            ]
        )
        time_position = self.time_axes.get_position()
        self.time_axes.set_position(
            [
                time_position.x0,
                time_position.y0 + lower_axis_shift + 0.012,
                time_position.width,
                time_position.height,
            ]
        )

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
        self._display_timestamps: tuple[pd.Timestamp, ...] = ()
        self._crosshair_vertical = None
        self._crosshair_timeline_vertical = None
        self._crosshair_horizontal = None
        self._crosshair_price_label: Annotation | None = None
        self._crosshair_time_label: Annotation | None = None
        self._crosshair_background = None

        super().__init__(figure)

        self.setParent(parent)
        self.setMinimumHeight(360)

        # Allow a mouse click to give the chart keyboard focus so annotation
        # shortcuts remain inactive while the user is typing into date fields.
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMouseTracking(True)
        self.mpl_connect("motion_notify_event", self._update_crosshair)
        self.mpl_connect("figure_leave_event", self._hide_crosshair)
        self.mpl_connect("draw_event", self._cache_crosshair_background)

    def resizeEvent(self, event) -> None:
        """Refresh responsive tick density after resizing."""

        super().resizeEvent(event)
        self._style_price_plot()
        if self._display_timestamps:
            self._update_axis_ticks()

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
        self.time_axes.clear()
        self._style_axes()
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
        self._display_timestamps = tuple(display_timestamps)

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

            # Use the original monochrome direction language on the white chart surface.
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
        price_range = session_high - session_low
        arrow_offset = max(price_range * 0.04, 0.01)

        # Point to the candle currently selected for regime annotation.
        self._arrow_offset = arrow_offset
        self._active_marker = self.axes.annotate(
            "",
            xy=(active_candlestick_position, active_high),
            xytext=(
                active_candlestick_position,
                active_high + arrow_offset,
            ),
            horizontalalignment="center",
            fontsize=PLOT_TICK_SIZE,
            color="#76bde8",
            arrowprops={
                "arrowstyle": "-|>",
                "color": "#76bde8",
                "linewidth": 1.8,
                "mutation_scale": 18,
            },
        )

        # Leave one empty candle position around the complete session chart.
        self.axes.set_xlim(-1, len(candlestick_data))

        # Extend the vertical limits so the active-candle arrow is not clipped.
        self.axes.set_ylim(
            session_low - max(price_range * 0.02, 0.01),
            session_high + (1.5 * arrow_offset),
        )

        # Select a responsive number of readable clock divisions while retaining session anchors.
        self._update_axis_ticks()
        self.axes.tick_params(axis="x", bottom=False, labelbottom=False)

        # Put session-local time below the chart and price values along its right
        # edge, matching the layout commonly used for market charts.
        display_timezone = self.session_timezone.rsplit("/", 1)[-1].replace("_", " ")
        self._time_axis_label = f"Time of day ({display_timezone})"
        self.axes.set_ylabel(
            "Price (U.S. Dollars)",
            fontsize=PLOT_LABEL_SIZE,
            labelpad=20,
        )
        self.axes.yaxis.tick_right()
        self.axes.yaxis.set_label_position("right")
        self.axes.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        self.axes.tick_params(
            axis="y",
            labelsize=PLOT_TICK_SIZE,
            colors=PLOT_TEXT_COLOR,
        )
        self.axes.yaxis.label.set_color(PLOT_TEXT_COLOR)
        self.axes.grid(False)
        self._configure_timeline_axes(track_count=0)
        self._create_crosshair()

        # Ask the existing Qt canvas to repaint after the chart has changed.
        self.draw_idle()

    def _preferred_axis_intervals(self) -> int:
        """Choose a compact tick count from the current logical canvas width."""

        chart_width = self.width()
        if chart_width < 700:
            return 4
        if chart_width < 900:
            return 5
        if chart_width < 1500:
            return 6
        return 7

    def _update_axis_ticks(self) -> None:
        """Apply responsive price ticks and session-clock ticks with fixed RTH anchors."""

        if not self._display_timestamps:
            return

        interval_count = self._preferred_axis_intervals()
        opening_position = next(
            (
                position
                for position, timestamp in enumerate(self._display_timestamps)
                if timestamp.hour == 9 and timestamp.minute == 30
            ),
            0,
        )
        closing_position = next(
            (
                position
                for position, timestamp in enumerate(self._display_timestamps)
                if timestamp.hour == 16 and timestamp.minute == 0
            ),
            len(self._display_timestamps) - 1,
        )
        if closing_position <= opening_position:
            closing_position = len(self._display_timestamps) - 1

        tick_positions = tuple(dict.fromkeys(
            round(
                opening_position
                + ((closing_position - opening_position) * division / interval_count)
            )
            for division in range(interval_count + 1)
        ))
        self._tick_positions = tick_positions
        self._tick_labels = tuple(
            self._display_timestamps[position].strftime("%H:%M")
            for position in tick_positions
        )
        self.axes.set_xticks(self._tick_positions)
        visible_low, visible_high = self.axes.get_ylim()
        if visible_high > visible_low:
            visible_range = visible_high - visible_low
            inset = visible_range * 0.05
            first_tick = ceil((visible_low + inset) * 4) / 4
            last_tick = floor((visible_high - inset) * 4) / 4
            quarter_count = round((last_tick - first_tick) * 4)
            if quarter_count < 1:
                midpoint_tick = round(((visible_low + visible_high) / 2) * 4) / 4
                self.axes.set_yticks([midpoint_tick])
                quarter_count = 0
            if quarter_count:
                chosen_intervals = min(interval_count, quarter_count)
                step = (last_tick - first_tick) / chosen_intervals
                self.axes.set_yticks(
                    [
                        first_tick + (step * division)
                        for division in range(chosen_intervals + 1)
                    ]
                )
        else:
            self.axes.yaxis.set_major_locator(MaxNLocator(nbins=interval_count))
        if hasattr(self, "time_axes"):
            self.time_axes.set_xticks(self._tick_positions, self._tick_labels)

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
        self.time_axes.clear()
        self._style_axes()
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
            left_edge, right_edge = self.timeline_axes.get_xlim()
            track_strips = []
            start = 0
            while start < len(regimes):
                regime = regimes[start]
                end = start + 1
                while end < len(regimes) and regimes[end] == regime:
                    end += 1
                strip_start = left_edge if start == 0 else start - 0.5
                strip_end = right_edge if end == len(regimes) else end - 0.5
                strip = Rectangle(
                    (strip_start, row + 0.08),
                    width=strip_end - strip_start,
                    height=0.84,
                    facecolor=colors[regime],
                    edgecolor="black",
                    linewidth=0.8,
                    hatch="///" if regime is None else None,
                )
                self.timeline_axes.add_patch(strip)
                track_strips.append(strip)
                self._regime_artists.append(strip)
                start = end

            # Clip every segment to one rounded track and draw a single uninterrupted border.
            track_outline = FancyBboxPatch(
                (left_edge, row + 0.08),
                right_edge - left_edge,
                0.84,
                boxstyle="round,pad=0,rounding_size=0.12",
                facecolor="none",
                edgecolor="black",
                linewidth=1.5,
                zorder=5,
            )
            self.timeline_axes.add_patch(track_outline)
            for strip in track_strips:
                strip.set_clip_path(track_outline)
            self._regime_artists.append(track_outline)

        legend = self.time_axes.legend(
            handles=[
                FancyBboxPatch(
                    (0, 0), 1, 1,
                    boxstyle="round,pad=0.04,rounding_size=0.18",
                    facecolor=colors[regime],
                    edgecolor="black",
                    linewidth=1.0,
                    hatch="///" if regime is None else None,
                    label=names[regime],
                )
                for regime in ("bull", "bear", "range", None)
            ],
            loc="upper center",
            bbox_to_anchor=(0.5, 1.18),
            ncol=4,
            frameon=False,
            fontsize=PLOT_LEGEND_SIZE,
            labelcolor=PLOT_TEXT_COLOR,
            handler_map={
                FancyBboxPatch: HandlerPatch(
                    patch_func=_rounded_legend_handle,
                )
            },
        )
        self._regime_artists.append(legend)
        self._create_crosshair()
        self.draw_idle()

    def _configure_timeline_axes(self, *, track_count: int) -> None:
        """Restore the shared candle/time scale after clearing timeline artists."""

        self.timeline_axes.set_xlim(-1, len(self._highs))
        self.time_axes.set_xlim(-1, len(self._highs))
        self.time_axes.set_xticks(self._tick_positions, self._tick_labels)
        self.timeline_axes.tick_params(axis="x", bottom=False, labelbottom=False)
        self.time_axes.set_xlabel(
            self._time_axis_label,
            fontsize=PLOT_LABEL_SIZE,
            labelpad=4,
        )
        self.timeline_axes.set_ylim(0, max(track_count, 1))
        if track_count:
            labels = [label for label, _ in reversed(self._regime_tracks)]
            # During a redraw, use the incoming labels before _regime_tracks is replaced.
            if len(labels) != track_count:
                labels = [""] * track_count
            if any(label.strip() for label in labels):
                self.timeline_axes.set_yticks(
                    [position + 0.5 for position in range(track_count)], labels
                )
            else:
                self.timeline_axes.set_yticks([])
        else:
            self.timeline_axes.set_yticks([])
        self.time_axes.set_ylim(0, 1)
        self.time_axes.set_yticks([])
        self.time_axes.tick_params(
            axis="x",
            labelsize=PLOT_TICK_SIZE,
            colors=PLOT_TEXT_COLOR,
            pad=5,
            length=5,
        )
        self.timeline_axes.tick_params(
            axis="y", labelsize=PLOT_TRACK_LABEL_SIZE, colors=PLOT_TEXT_COLOR
        )
        self.time_axes.xaxis.label.set_color(PLOT_TEXT_COLOR)
        self.timeline_axes.grid(False)
        for spine in self.timeline_axes.spines.values():
            spine.set_color("black")
            spine.set_linewidth(1.0)
        for side, spine in self.time_axes.spines.items():
            spine.set_visible(side == "bottom")
            spine.set_color(PLOT_TEXT_COLOR)
            spine.set_linewidth(1.0)

    def _style_axes(self) -> None:
        """Restore the chart's high-contrast white plotting surface after clearing axes."""

        self._style_price_plot()
        self.axes.tick_params(colors=PLOT_TEXT_COLOR)
        self.axes.xaxis.label.set_color(PLOT_TEXT_COLOR)
        self.axes.yaxis.label.set_color(PLOT_TEXT_COLOR)
        self.axes.title.set_color(PLOT_TEXT_COLOR)
        for spine in self.axes.spines.values():
            spine.set_visible(False)

        self.timeline_axes.set_facecolor(PLOT_AXES_COLOR)
        self.timeline_axes.tick_params(colors=PLOT_TEXT_COLOR)
        self.timeline_axes.xaxis.label.set_color(PLOT_TEXT_COLOR)
        self.timeline_axes.yaxis.label.set_color(PLOT_TEXT_COLOR)
        self.timeline_axes.title.set_color(PLOT_TEXT_COLOR)
        for spine in self.timeline_axes.spines.values():
            spine.set_color("#11171b")

        self.time_axes.set_facecolor(PLOT_FIGURE_COLOR)
        self.time_axes.tick_params(colors=PLOT_TEXT_COLOR)
        self.time_axes.xaxis.label.set_color(PLOT_TEXT_COLOR)
        self.time_axes.yaxis.label.set_color(PLOT_TEXT_COLOR)
        for spine in self.time_axes.spines.values():
            spine.set_color("#11171b")

    def _style_price_plot(self) -> None:
        """Draw the white price plot with a display-pixel-symmetric rounded border."""

        axes_bounds = self.axes.get_window_extent()
        axes_width = max(float(axes_bounds.width), 1.0)
        axes_height = max(float(axes_bounds.height), 1.0)
        corner_radius = 8.0 / axes_width
        self.axes.patch = FancyBboxPatch(
            (0, 0),
            1,
            1,
            transform=self.axes.transAxes,
            boxstyle=f"round,pad=0,rounding_size={corner_radius}",
            mutation_aspect=axes_width / axes_height,
            facecolor="white",
            edgecolor="black",
            linewidth=2.5,
            zorder=-1,
        )

    def _create_crosshair(self) -> None:
        """Create the hidden Trade Tank-style crosshair and axis locator boxes."""

        for artist in (
            self._crosshair_vertical,
            self._crosshair_timeline_vertical,
            self._crosshair_horizontal,
            self._crosshair_price_label,
            self._crosshair_time_label,
        ):
            if artist is not None:
                try:
                    artist.remove()
                except (NotImplementedError, ValueError):
                    pass

        line_style = {
            "color": "black",
            "alpha": 0.55,
            "linewidth": 1,
            "linestyle": (0, (4, 4)),
            "visible": False,
            "zorder": 20,
        }
        self._crosshair_vertical = self.axes.axvline(0, **line_style)
        timeline_line_style = {**line_style, "color": PLOT_TEXT_COLOR}
        self._crosshair_timeline_vertical = self.timeline_axes.axvline(
            0, **timeline_line_style
        )
        self._crosshair_horizontal = self.axes.axhline(0, **line_style)
        locator_box = {"boxstyle": "square,pad=0.25", "fc": "black", "ec": "black"}
        self._crosshair_price_label = self.axes.annotate(
            "",
            xy=(1.0, 0),
            xycoords=("axes fraction", "data"),
            xytext=(4, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            color="white",
            fontsize=PLOT_TICK_SIZE,
            bbox=locator_box,
            visible=False,
            annotation_clip=False,
            zorder=21,
        )
        self._crosshair_time_label = self.time_axes.annotate(
            "",
            xy=(0, 0),
            xycoords=("data", "axes fraction"),
            xytext=(0, -4),
            textcoords="offset points",
            ha="center",
            va="top",
            color="white",
            fontsize=PLOT_TICK_SIZE,
            bbox=locator_box,
            visible=False,
            annotation_clip=False,
            zorder=21,
        )

    def _update_crosshair(self, event) -> None:
        """Snap pointer movement to one candle and one visible quarter-point price."""

        if (
            event.inaxes is not self.axes
            or event.xdata is None
            or not self._display_timestamps
        ):
            self._hide_crosshair()
            return

        chart_left, chart_right = self.axes.get_xlim()
        crosshair_position = min(
            chart_right,
            max(chart_left, float(event.xdata)),
        )
        if event.ydata is None:
            return
        minimum_price, maximum_price = self.axes.get_ylim()
        minimum_tick = ceil(minimum_price * 4) / 4
        maximum_tick = floor(maximum_price * 4) / 4
        price = min(
            maximum_tick,
            max(minimum_tick, floor(float(event.ydata) * 4 + 0.5) / 4),
        )

        for line in (self._crosshair_vertical, self._crosshair_timeline_vertical):
            line.set_xdata([crosshair_position, crosshair_position])
            line.set_visible(True)
        self._crosshair_horizontal.set_ydata([price, price])
        self._crosshair_horizontal.set_visible(True)
        self._crosshair_price_label.xy = (1.0, price)
        self._crosshair_price_label.set_text(f"{price:.2f}")
        self._crosshair_price_label.set_visible(True)
        candlestick_position = min(
            len(self._display_timestamps) - 1,
            max(0, floor(crosshair_position + 0.5)),
        )
        crosshair_timestamp = self._display_timestamps[candlestick_position]
        self._crosshair_time_label.xy = (crosshair_position, 0)
        self._crosshair_time_label.set_text(crosshair_timestamp.strftime("%H:%M"))
        self._crosshair_time_label.set_visible(True)
        self._blit_crosshair()

    def _hide_crosshair(self, event=None) -> None:
        """Hide all crosshair artists when the pointer leaves the chart area."""

        artists = (
            self._crosshair_vertical,
            self._crosshair_timeline_vertical,
            self._crosshair_horizontal,
            self._crosshair_price_label,
            self._crosshair_time_label,
        )
        changed = False
        for artist in artists:
            if artist is not None and artist.get_visible():
                artist.set_visible(False)
                changed = True
        if changed:
            if self._crosshair_background is not None:
                self.restore_region(self._crosshair_background)
                self.blit(self.figure.bbox)
            else:
                self.draw_idle()

    def _cache_crosshair_background(self, event=None) -> None:
        """Cache the static figure so pointer movement can redraw at display speed."""

        self._crosshair_background = self.copy_from_bbox(self.figure.bbox)

    def _blit_crosshair(self) -> None:
        """Redraw only crosshair artists instead of repainting every candlestick."""

        if self._crosshair_background is None:
            self.draw_idle()
            return
        self.restore_region(self._crosshair_background)
        for artist in (
            self._crosshair_vertical,
            self._crosshair_timeline_vertical,
            self._crosshair_horizontal,
            self._crosshair_price_label,
            self._crosshair_time_label,
        ):
            artist.axes.draw_artist(artist)
        self.blit(self.figure.bbox)

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
