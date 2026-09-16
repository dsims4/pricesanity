"""Shared visual choices for long annotation sessions and read-only research views."""
from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QColor, QIcon, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QLabel, QPushButton, QAbstractItemView, QTableWidget, QToolButton

REGIME_COLORS = {'bull': '#17765e', 'bear': '#b7444e', 'range': '#946513'}
REGIME_SYMBOLS = {'bull': '↗', 'bear': '↘', 'range': '↔'}
SPACING = 12
MARGIN = 20
PRIMARY_HEIGHT = 44

STYLE = '''
QMainWindow, QDialog { background: #f4f6f8; color: #172b3a; }
QWidget { font-family: sans-serif; font-size: 13px; }
QLabel { color: #172b3a; }
QLabel[role="title"] { font-size: 24px; font-weight: 600; padding: 4px 0; }
QLabel[role="muted"] { color: #506473; }
QLabel[role="readonly"] { color: #345b72; background: #e5eff5; padding: 8px; border-radius: 6px; }
QFrame[role="card"] { background: white; border: 1px solid #d5dee5; border-radius: 8px; }
QPushButton, QToolButton { background: white; color: #172b3a; border: 1px solid #b8c7d1;
    border-radius: 6px; padding: 6px 12px; }
QPushButton:hover, QToolButton:hover { background: #edf3f7; border-color: #66869b; }
QPushButton:pressed, QToolButton:pressed { background: #dce8f0; }
QPushButton:checked { background: #dceee9; border: 2px solid #17765e; font-weight: 600; }
QPushButton[regime="bear"]:checked { background: #f8e5e7; border-color: #b7444e; }
QPushButton[regime="range"]:checked { background: #f7eddb; border-color: #946513; }
QPushButton:focus, QToolButton:focus, QComboBox:focus, QDateEdit:focus { border: 2px solid #246ba0; }
QPushButton:disabled, QToolButton:disabled { color: #667681; background: #ebeff2; border-color: #d5dee5; }
QComboBox, QDateEdit, QSpinBox { background: white; color: #172b3a; min-height: 32px;
    padding: 4px 8px; border: 1px solid #b8c7d1; border-radius: 5px; }
QTabWidget::pane { background: white; border: 1px solid #d5dee5; }
QTabBar::tab { padding: 12px 18px; background: #e6ecf1; color: #344e60; }
QTabBar::tab:selected { background: white; border-bottom: 3px solid #246ba0; }
QTableWidget { background: white; alternate-background-color: #f3f6f8; gridline-color: #e1e7eb; }
QHeaderView::section { background: #e9eff3; color: #344e60; padding: 10px; border: none; }
QProgressBar { border: 1px solid #c4d2dc; border-radius: 5px; background: #e8eff3; min-height: 20px; text-align: center; color: #172b3a; }
QProgressBar::chunk { background: #a7d5c7; border-radius: 4px; }
QToolButton#qt_calendar_monthbutton, QToolButton#qt_calendar_yearbutton { padding: 0px; text-align: center; }
QToolButton#qt_calendar_monthbutton::menu-indicator, QToolButton#qt_calendar_yearbutton::menu-indicator { image: none; width: 0px; }
'''


def regime_icon(regime: str, *, size: int = 24, device_pixel_ratio: float = 1.0) -> QIcon:
    """Draw semantic paths at device resolution without fonts or external icon packages."""
    pixmap = QPixmap(round(size * device_pixel_ratio), round(size * device_pixel_ratio))
    pixmap.setDevicePixelRatio(device_pixel_ratio)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setPen(QPen(QColor(REGIME_COLORS[regime]), 2.4))
    if regime == 'range':
        painter.drawLine(QPointF(4,12),QPointF(20,12))
        for x in (4,20): painter.drawLine(QPointF(x,7),QPointF(x,17))
    else:
        y1,y2 = (18,6) if regime == 'bull' else (6,18)
        painter.drawLine(QPointF(4,y1),QPointF(20,y2))
        painter.drawLine(QPointF(20,y2),QPointF(12,y2))
        painter.drawLine(QPointF(20,y2),QPointF(20,14 if regime == 'bull' else 10))
    painter.end()
    return QIcon(pixmap)


def heading(text: str, *, role: str = 'title') -> QLabel:
    label = QLabel(text)
    label.setProperty('role',role)
    label.setWordWrap(True)
    return label


def apply_theme(window) -> None:
    window.setProperty('pricesanityTheme','shared-v1')
    window.setStyleSheet(STYLE)
    # At 150% scaling, a 1080p screen has only 720 logical pixels. Recover space
    # from margins and the chart before reducing the annotation hit targets.
    screen=window.screen()
    compact=screen is not None and screen.availableGeometry().height()<900
    if window.centralWidget() and window.centralWidget().layout():
        layout=window.centralWidget().layout()
        margin=8 if compact else MARGIN
        layout.setContentsMargins(margin,margin,margin,margin)
        layout.setSpacing(4 if compact else SPACING)
    if compact and getattr(window,'chart',None) is not None:
        window.chart.setMinimumHeight(240)
    for button in window.findChildren(QPushButton):
        button.setMinimumHeight(PRIMARY_HEIGHT)
        button.setAccessibleName(button.text())
    for table in window.findChildren(QTableWidget):
        table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        table.setAlternatingRowColors(True)
        table.verticalHeader().setDefaultSectionSize(36)
