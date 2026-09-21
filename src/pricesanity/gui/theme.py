"""Shared visual choices for long annotation sessions and read-only research views."""

from PySide6.QtCore import QEvent, QObject, QPointF, Qt
from PySide6.QtGui import QColor, QIcon, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QDateEdit,
    QLabel,
    QPushButton,
    QSizePolicy,
    QTableWidget,
    QToolButton,
)

# Share class order and meaning across annotation cards, label spans, and comparison bands.
REGIME_COLORS = {"bull": "#17765e", "bear": "#b7444e", "range": "#946513"}
REGIME_SYMBOLS = {"bull": "↗", "bear": "↘", "range": "↔"}
TIGHT_SPACING = 6
CONTROL_SPACING = 10
SECTION_SPACING = 16
PANEL_MARGIN = 20
# Backward-compatible names used by existing callers.
SPACING = CONTROL_SPACING
MARGIN = PANEL_MARGIN
PRIMARY_HEIGHT = 44
COMPACT_PRIMARY_HEIGHT = 30
MINIMUM_WINDOW_WIDTH = 760
MINIMUM_WINDOW_HEIGHT = 480
PREFERRED_WINDOW_WIDTH = 1200
PREFERRED_WINDOW_HEIGHT = 820
PLOT_TITLE_SIZE = 14
PLOT_LABEL_SIZE = 12
PLOT_TICK_SIZE = 11
PLOT_LEGEND_SIZE = 11
PLOT_TIMELINE_SIZE = 10
PLOT_TRACK_LABEL_SIZE = 8
PLOT_FIGURE_COLOR = "#22282c"
PLOT_AXES_COLOR = "#2d3439"
PLOT_TEXT_COLOR = "#e4ebef"
PLOT_GRID_COLOR = "#69757d"
PLOT_SPINE_COLOR = "#11171b"

# Semantic roles let windows request emphasis without duplicating platform-specific styling.
STYLE = """
QMainWindow, QDialog { background: #22282c; color: #e4ebef; }
QWidget { color: #e4ebef; font-size: 14px; }
QLabel { color: #e4ebef; }
QLabel[role="title"] { font-size: 24px; font-weight: 650; padding: 4px 0; }
QLabel[role="section"] { font-size: 16px; font-weight: 600; }
QLabel[role="muted"] { color: #a9b6be; font-size: 13px; }
QLabel[role="readonly"] {
    color: #c6dfed; background: #293943; padding: 9px;
    border: 2px solid #11171b; border-radius: 6px; font-weight: 550;
}
QLabel[role="openingGap"] {
    padding: 4px 10px; border: 2px solid #11171b; border-radius: 5px;
    font-size: 16px; font-weight: 600;
}
QLabel[gapDirection="positive"] { background: #233b34; color: #8fd3bd; }
QLabel[gapDirection="negative"] { background: #432b30; color: #e4a1a8; }
QLabel[gapDirection="neutral"] { background: #30383d; color: #c2cdd3; }
QLabel[role="regimeValue"] {
    background: transparent; color: #e4ebef; padding: 0px;
    border: none; font-weight: 600;
}
QFrame[role="card"] {
    background: #2d3439; border: 2px solid #11171b; border-radius: 8px;
}
QFrame[role="chart"] {
    background: #2d3439; border: 2px solid #11171b; border-radius: 8px;
}
QPushButton, QToolButton {
    background: #343d43; color: #e4ebef; border: 2px solid #11171b;
    border-radius: 6px; padding: 6px 12px; font-weight: 550; }
QPushButton:hover, QToolButton:hover { background: #46535b; border-color: #778b97; }
QPushButton:pressed, QToolButton:pressed { background: #53636d; }
QPushButton[compact="true"] { min-height: 24px; padding: 4px 9px; }
QPushButton[wideAction="true"] { min-width: 90px; padding-left: 18px; padding-right: 18px; }
QPushButton:checked { background: #1e5145; border: 2px solid #42a78b; font-weight: 600; }
QPushButton[regime="bear"]:checked { background: #5a2c33; border-color: #d06b75; }
QPushButton[regime="range"]:checked { background: #59451f; border-color: #c99745; }
QPushButton:focus, QToolButton:focus, QComboBox:focus, QDateEdit:focus {
    border: 2px solid #67a9d3;
}
QPushButton:disabled, QToolButton:disabled {
    color: #74818a; background: #2a3034; border-color: #171e22;
}
QComboBox, QDateEdit, QSpinBox {
    background: #30383d; color: #e4ebef; min-height: 32px;
    padding: 4px 8px; border: 2px solid #11171b; border-radius: 5px; }
QComboBox::drop-down, QDateEdit::drop-down {
    width: 24px; border-left: 2px solid #11171b;
}
QDateEdit::drop-down:hover { background: #46535b; }
QDateEdit::drop-down:pressed { background: #53636d; }
QDateEdit::down-arrow { image: none; width: 0px; height: 0px; }
QTabWidget::pane { background: #2d3439; border: 2px solid #11171b; }
QTabBar::tab { padding: 12px 18px; background: #30383d; color: #bac6cc; font-weight: 550; }
QTabBar::tab:selected { background: #3a444a; color: #f1f5f7; border-bottom: 3px solid #67a9d3; }
QTableWidget {
    background: #2d3439; color: #e4ebef; alternate-background-color: #333c42;
    gridline-color: #151c20; border: 2px solid #11171b;
}
QHeaderView::section {
    background: #374147; color: #e4ebef; padding: 10px;
    border: 0px; border-right: 1px solid #151c20; border-bottom: 2px solid #11171b;
    font-weight: 600;
}
QProgressBar {
    border: 2px solid #11171b; border-radius: 5px; background: #30383d;
    min-height: 20px; text-align: center; color: #e4ebef;
}
QProgressBar::chunk { background: #2f806c; border-radius: 4px; }
QCheckBox { spacing: 8px; font-weight: 550; }
QCheckBox::indicator { width: 18px; height: 18px; }
QCalendarWidget {
    background: #2d3439; border: 2px solid #11171b; border-radius: 7px;
}
QCalendarWidget QWidget#qt_calendar_navigationbar {
    background: #30383d; border-bottom: 2px solid #11171b; padding: 6px;
}
QCalendarWidget QToolButton {
    min-height: 30px; margin: 2px; padding: 3px 8px;
    background: #374147; border: 2px solid #11171b; font-weight: 600;
}
QCalendarWidget QAbstractItemView {
    background: #2d3439; color: #e4ebef; border: 1px solid #11171b;
    selection-background-color: #397da8; selection-color: white;
    outline: 0; alternate-background-color: #2d3439;
}
QCalendarWidget QAbstractItemView:item:hover {
    background: #46535b; color: #f1f5f7;
}
QCalendarWidget QMenu {
    background: #30383d; color: #e4ebef; border: 2px solid #11171b; padding: 4px;
}
QCalendarWidget QMenu::item { padding: 6px 18px 6px 10px; border-radius: 3px; }
QCalendarWidget QMenu::item:selected { background: #46535b; color: #f1f5f7; }
QToolButton#qt_calendar_monthbutton, QToolButton#qt_calendar_yearbutton {
    padding: 3px 8px; text-align: center;
}
QToolButton#qt_calendar_monthbutton::menu-indicator,
QToolButton#qt_calendar_yearbutton::menu-indicator {
    image: none; width: 0px;
}
"""


def regime_icon(regime: str, *, size: int = 24, device_pixel_ratio: float = 1.0) -> QIcon:
    """Draw semantic paths at device resolution without fonts or external icon packages."""

    # Render at the active device-pixel ratio so the small direction glyph stays sharp on
    # Retina and scaled Linux displays without depending on an installed symbol font.
    pixmap = QPixmap(round(size * device_pixel_ratio), round(size * device_pixel_ratio))
    pixmap.setDevicePixelRatio(device_pixel_ratio)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setPen(QPen(QColor(REGIME_COLORS[regime]), 2.4))
    if regime == "range":
        painter.drawLine(QPointF(4, 12), QPointF(20, 12))
        for x_position in (4, 20):
            painter.drawLine(QPointF(x_position, 7), QPointF(x_position, 17))
    else:
        start_y, end_y = (18, 6) if regime == "bull" else (6, 18)
        painter.drawLine(QPointF(4, start_y), QPointF(20, end_y))
        painter.drawLine(QPointF(20, end_y), QPointF(12, end_y))
        arrow_y = 14 if regime == "bull" else 10
        painter.drawLine(QPointF(20, end_y), QPointF(20, arrow_y))

    painter.end()
    return QIcon(pixmap)


def heading(text: str, *, role: str = "title") -> QLabel:
    """Create a wrapping label with one shared semantic theme role."""

    label = QLabel(text)
    label.setProperty("role", role)
    label.setWordWrap(True)
    return label


class _ResponsiveWindowFilter(QObject):
    """Reapply shared compact metrics whenever a themed window changes size."""

    def eventFilter(self, watched, event):
        """Update layout density after Qt commits a new window size."""

        if event.type() == QEvent.Type.Resize:
            _apply_responsive_metrics(watched)

        return False


def _configure_window_geometry(window) -> None:
    """Choose a multitasking-friendly initial size without exceeding the screen."""

    screen = window.screen()
    if screen is None:
        window.setMinimumSize(MINIMUM_WINDOW_WIDTH, MINIMUM_WINDOW_HEIGHT)
        window.resize(PREFERRED_WINDOW_WIDTH, PREFERRED_WINDOW_HEIGHT)
        return

    available = screen.availableGeometry()
    minimum_width = min(MINIMUM_WINDOW_WIDTH, available.width())
    minimum_height = min(MINIMUM_WINDOW_HEIGHT, available.height())
    window.setMinimumSize(minimum_width, minimum_height)

    # This is the restored size; launchers maximize without losing normal window controls.
    window.resize(
        max(minimum_width, round(available.width() * 0.5)),
        max(minimum_height, round(available.height() * 0.5)),
    )


def _apply_responsive_metrics(window) -> None:
    """Adjust only spacing and minimum control sizes at compact window dimensions."""

    compact = window.width() < 1100 or window.height() < 760
    # Restyling only when density changes avoids repolishing every child on each resize.
    if window.property("compactLayout") != compact:
        window.setProperty("compactLayout", compact)
        window.setStyleSheet(STYLE + (COMPACT_STYLE if compact else ""))

    if window.centralWidget() and window.centralWidget().layout():
        layout = window.centralWidget().layout()
        margin = 4 if compact else 10
        layout.setContentsMargins(margin, margin, margin, margin)
        layout.setSpacing(3 if compact else CONTROL_SPACING)

    # Dense annotation rows contract as a unit instead of allowing their final controls to be
    # clipped when a maximized window is restored onto a smaller monitor.
    for layout_name in ("date_range_layout", "navigation_layout", "regime_layout"):
        child_layout = getattr(window, layout_name, None)
        if child_layout is not None:
            child_layout.setSpacing(4 if compact else CONTROL_SPACING)

    for date_input in window.findChildren(QDateEdit):
        if date_input.property("responsiveDateInput"):
            date_input.setFixedWidth(128 if compact else 150)

    # Both chart types retain most of the flexible height while yielding enough room for their
    # controls on the supported minimum window size.
    chart = getattr(window, "chart", None)
    if chart is not None:
        chart.setMinimumHeight(170 if compact else 340)
    comparison_chart = getattr(window, "comparison_chart", None)
    if comparison_chart is not None:
        comparison_chart.setMinimumHeight(260 if compact else 400)

    for button in window.findChildren(QPushButton):
        # Every action in one window state keeps the same outer height; compact properties alter
        # padding and width only, so Stop and navigation controls remain visually aligned.
        button_height = COMPACT_PRIMARY_HEIGHT if compact else PRIMARY_HEIGHT
        button.setMinimumHeight(button_height)
        if button.property("responsiveNav"):
            full_label = button.property("fullLabel") or button.text()
            button.setProperty("fullLabel", full_label)
            button.setText(full_label.replace("Previous", "Prev") if compact else full_label)
            button.setMinimumWidth(button.sizeHint().width())
            button.setSizePolicy(
                QSizePolicy.Policy.Preferred,
                QSizePolicy.Policy.Fixed,
            )
        if button.property("wideAction"):
            button.setMinimumWidth(90 if compact else 112)

    opening_gap_value = getattr(window, "opening_gap_value", None)
    if opening_gap_value is not None:
        opening_gap_value.setMinimumWidth(72 if compact else 90)

    secondary_panel = getattr(window, "comparison_details", None)
    if secondary_panel is not None:
        secondary_panel.setMaximumHeight(65 if compact else 120)

    for table in window.findChildren(QTableWidget):
        table.verticalHeader().setDefaultSectionSize(32 if compact else 36)


def apply_theme(window) -> None:
    """Apply shared styling and compact-screen sizing to one top-level window."""

    window.setProperty("pricesanityTheme", "shared-v1")
    window.setProperty("compactLayout", None)
    window.setStyleSheet(STYLE)
    _configure_window_geometry(window)

    # Enforce consistent hit targets and read-only result tables after each window has built
    # its children, avoiding duplicated widget policy in every GUI module.
    for button in window.findChildren(QPushButton):
        button.setAttribute(Qt.WidgetAttribute.WA_Hover, True)
        button.setMouseTracking(True)
        button.setAccessibleName(button.text())

    # Qt tool buttons include the calendar popup controls. Explicit hover tracking keeps their
    # feedback responsive even while the chart or a date editor owns keyboard focus.
    for button in window.findChildren(QToolButton):
        button.setAttribute(Qt.WidgetAttribute.WA_Hover, True)
        button.setMouseTracking(True)

    for table in window.findChildren(QTableWidget):
        table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        table.setAlternatingRowColors(True)

    # Qt layouts already distribute width and height; this filter changes only density thresholds
    # as the user moves between a multitasking window and a larger workspace.
    old_filter = getattr(window, "_pricesanity_responsive_filter", None)
    if old_filter is not None:
        window.removeEventFilter(old_filter)
        old_filter.deleteLater()
    responsive_filter = _ResponsiveWindowFilter(window)
    window._pricesanity_responsive_filter = responsive_filter
    window.installEventFilter(responsive_filter)
    _apply_responsive_metrics(window)


# Keep borders and semantic colors intact while reclaiming padding before chart space.
COMPACT_STYLE = """
QWidget { font-size: 12px; }
QLabel[role="title"] { font-size: 18px; padding: 0; }
QLabel[role="section"] { font-size: 14px; }
QLabel[role="muted"] { font-size: 12px; }
QLabel[role="readonly"] { padding: 3px; }
QLabel[role="openingGap"] { font-size: 13px; padding: 2px 5px; }
QPushButton, QToolButton { padding: 2px 6px; }
QPushButton[compact="true"] { min-height: 22px; padding: 2px 4px; }
QPushButton[wideAction="true"] { padding-left: 6px; padding-right: 6px; }
QComboBox, QDateEdit, QSpinBox { min-height: 22px; padding: 2px 5px; }
QTabBar::tab { padding: 6px 8px; }
QHeaderView::section { padding: 5px; }
QProgressBar { min-height: 14px; }
"""


def compact_plot(canvas) -> bool:
    """Use the owning Qt window's density, including before a canvas is first displayed."""

    tier = canvas.window().property("compactLayout")
    return bool(tier) if tier is not None else canvas.height() < 450


def apply_plot_typography(canvas) -> None:
    """Scale every chart text category with the same compact decision as its Qt controls."""

    compact = compact_plot(canvas)
    for axis in canvas.figure.axes:
        axis.tick_params(labelsize=8 if compact else 10, pad=2 if compact else 4)
        for label in (axis.xaxis.label, axis.yaxis.label):
            label.set_fontsize(9 if compact else 11)
        axis.title.set_fontsize(10 if compact else 14)
        for label in axis.texts:
            label.set_fontsize(8 if compact else 10)
        legend = axis.get_legend()
        if legend is not None:
            for label in legend.get_texts():
                label.set_fontsize(8 if compact else 10)
