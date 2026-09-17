"""Shared visual choices for long annotation sessions and read-only research views."""

from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QColor, QIcon, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QLabel,
    QPushButton,
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

# Semantic roles let windows request emphasis without duplicating platform-specific styling.
STYLE = """
QMainWindow, QDialog { background: #f4f6f8; color: #172b3a; }
QWidget { font-size: 14px; }
QLabel { color: #172b3a; }
QLabel[role="title"] { font-size: 24px; font-weight: 650; padding: 4px 0; }
QLabel[role="section"] { font-size: 16px; font-weight: 600; }
QLabel[role="muted"] { color: #506473; font-size: 13px; }
QLabel[role="readonly"] {
    color: #264f67; background: #e5eff5; padding: 9px;
    border: 1px solid #9fb4c2; border-radius: 6px; font-weight: 550;
}
QFrame[role="card"] {
    background: white; border: 2px solid #b7c6d0; border-radius: 8px;
}
QFrame[role="chart"] {
    background: white; border: 2px solid #9fb1bd; border-radius: 7px;
}
QPushButton, QToolButton {
    background: white; color: #172b3a; border: 2px solid #aebfc9;
    border-radius: 6px; padding: 6px 12px; font-weight: 550; }
QPushButton:hover, QToolButton:hover { background: #edf3f7; border-color: #66869b; }
QPushButton:pressed, QToolButton:pressed { background: #dce8f0; }
QPushButton:checked { background: #dceee9; border: 2px solid #17765e; font-weight: 600; }
QPushButton[regime="bear"]:checked { background: #f8e5e7; border-color: #b7444e; }
QPushButton[regime="range"]:checked { background: #f7eddb; border-color: #946513; }
QPushButton:focus, QToolButton:focus, QComboBox:focus, QDateEdit:focus {
    border: 2px solid #246ba0;
}
QPushButton:disabled, QToolButton:disabled {
    color: #667681; background: #ebeff2; border-color: #d5dee5;
}
QComboBox, QDateEdit, QSpinBox {
    background: white; color: #172b3a; min-height: 32px;
    padding: 4px 8px; border: 2px solid #aebfc9; border-radius: 5px; }
QComboBox::drop-down, QDateEdit::drop-down {
    width: 24px; border-left: 1px solid #aebfc9;
}
QTabWidget::pane { background: white; border: 2px solid #b7c6d0; }
QTabBar::tab { padding: 12px 18px; background: #e6ecf1; color: #344e60; font-weight: 550; }
QTabBar::tab:selected { background: white; border-bottom: 3px solid #246ba0; }
QTableWidget {
    background: white; alternate-background-color: #f3f6f8;
    gridline-color: #c8d4dc; border: 2px solid #b7c6d0;
}
QHeaderView::section {
    background: #e9eff3; color: #29485b; padding: 10px;
    border: 0px; border-right: 1px solid #c3d0d8; border-bottom: 1px solid #aebfc9;
    font-weight: 600;
}
QProgressBar {
    border: 2px solid #afc0cb; border-radius: 5px; background: #e8eff3;
    min-height: 20px; text-align: center; color: #172b3a;
}
QProgressBar::chunk { background: #a7d5c7; border-radius: 4px; }
QCheckBox { spacing: 8px; font-weight: 550; }
QCheckBox::indicator { width: 18px; height: 18px; }
QCalendarWidget {
    background: white; border: 2px solid #9fb1bd; border-radius: 7px;
}
QCalendarWidget QWidget#qt_calendar_navigationbar {
    background: #e5edf2; border-bottom: 1px solid #9fb1bd; padding: 6px;
}
QCalendarWidget QToolButton {
    min-height: 30px; margin: 2px; padding: 3px 8px;
    background: white; border: 1px solid #9fb1bd; font-weight: 600;
}
QCalendarWidget QAbstractItemView {
    background: white; color: #172b3a; border: 1px solid #b7c6d0;
    selection-background-color: #246ba0; selection-color: white;
    outline: 0; alternate-background-color: white;
}
QCalendarWidget QAbstractItemView:item:hover {
    background: #dce8f0; color: #172b3a;
}
QCalendarWidget QMenu {
    background: white; color: #172b3a; border: 1px solid #9fb1bd; padding: 4px;
}
QToolButton#qt_calendar_monthbutton, QToolButton#qt_calendar_yearbutton {
    padding: 3px 8px; text-align: center;
}
QToolButton#qt_calendar_monthbutton::menu-indicator,
QToolButton#qt_calendar_yearbutton::menu-indicator { image: none; width: 0px; }
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


def apply_theme(window) -> None:
    """Apply shared styling and compact-screen sizing to one top-level window."""

    window.setProperty("pricesanityTheme", "shared-v1")
    window.setStyleSheet(STYLE)

    # At 150% scaling, a 1080p screen has only 720 logical pixels. Recover space
    # from margins and the chart before reducing the annotation hit targets.
    screen = window.screen()
    compact = screen is not None and screen.availableGeometry().height() < 900
    if window.centralWidget() and window.centralWidget().layout():
        layout = window.centralWidget().layout()
        margin = 4 if compact else PANEL_MARGIN
        layout.setContentsMargins(margin, margin, margin, margin)
        layout.setSpacing(2 if compact else CONTROL_SPACING)

    if compact and getattr(window, "chart", None) is not None:
        window.chart.setMinimumHeight(240)

    # Enforce consistent hit targets and read-only result tables after each window has built
    # its children, avoiding duplicated widget policy in every GUI module.
    for button in window.findChildren(QPushButton):
        button.setMinimumHeight(PRIMARY_HEIGHT)
        button.setAccessibleName(button.text())

    for table in window.findChildren(QTableWidget):
        table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        table.setAlternatingRowColors(True)
        table.verticalHeader().setDefaultSectionSize(36)
