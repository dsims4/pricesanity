"""Application window for annotating candlestick regimes."""

from datetime import date
from functools import partial
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from PySide6.QtCore import QDate, Qt
from PySide6.QtGui import QColor, QCloseEvent, QKeySequence, QShortcut, QTextCharFormat
from PySide6.QtWidgets import (
    QCalendarWidget,
    QDateEdit,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMenu,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from pricesanity.annotation.schema import CandlestickAnnotation, MarketRegime
from pricesanity.annotation.store import AnnotationStore
from pricesanity.gui.chart_widget import CandlestickChart

# Use one key map for shortcut creation and visible selection feedback.
REGIME_SHORTCUT_OPTIONS = {
    "1": MarketRegime.BULL,
    "2": MarketRegime.BEAR,
    "3": MarketRegime.RANGE,
}

# Reverse the shortcut map so saved regimes can show their original key.
REGIME_KEYS = {regime: number_key for number_key, regime in REGIME_SHORTCUT_OPTIONS.items()}

# Distinguish an untouched scalar from any of the three valid choices.
UNSELECTED_REGIME_TEXT = "Not selected"


class AnnotationWindow(QMainWindow):
    """Connect session data, chart navigation, and regime annotations."""

    def __init__(
        self,
        candlestick_data: pd.DataFrame,
        database_path: str | Path,
        *,
        timestamp_column: str = "ts_event",
        corpus_start_date: date | None = None,
        corpus_end_date: date | None = None,
    ) -> None:
        """Create the annotation window for prepared trading sessions.

        Args:
            candlestick_data: Dated, chronological 5-minute OHLC candlesticks.
            database_path: Path to the SQLite annotation database.
            timestamp_column: Column containing UTC candle timestamps.
            corpus_start_date: First calendar date named by the corpus.
            corpus_end_date: Last inclusive calendar date in the corpus.
        """

        # Initialize Qt ownership before adding controls that the window must later release.
        super().__init__()

        # Keep a private copy so the window cannot alter the source market data.
        self.all_candlestick_data = candlestick_data.copy()
        self.timestamp_column = timestamp_column

        # An empty date range cannot supply an active candle or annotation.
        if self.all_candlestick_data.empty:
            raise ValueError("Candlestick data cannot be empty.")

        # Require the chart fields and a persistent database key before any GUI
        # state is created around an incomplete table.
        required_columns = {
            "candlestick_id",
            "session_date",
            "open_gap",
            "open",
            "high",
            "low",
            "close",
            timestamp_column,
        }
        missing_columns = required_columns.difference(
            self.all_candlestick_data.columns
        )

        # Reject missing display fields before partially initializing the window.
        if missing_columns:
            raise ValueError(
                "Candlestick data is missing required columns: "
                f"{', '.join(sorted(missing_columns))}."
            )

        # Normalize date-like values once so comparisons from the Qt date
        # controls never depend on whether the loader supplied strings or dates.
        self.all_candlestick_data["session_date"] = pd.to_datetime(
            self.all_candlestick_data["session_date"],
            errors="raise",
        ).dt.date

        # Store opening gaps as numbers so every active-candle display uses the
        # exact causal feature aligned by the launcher.
        self.all_candlestick_data["open_gap"] = pd.to_numeric(
            self.all_candlestick_data["open_gap"],
            errors="raise",
        )

        # Invalid opening-gap features must not be presented as trustworthy model inputs.
        if not np.isfinite(
            self.all_candlestick_data["open_gap"].to_numpy(dtype=float)
        ).all():
            raise ValueError("Opening gaps must be finite numbers.")

        # Refuse blank or repeated identifiers because either condition could
        # overwrite the annotation belonging to a different candle.
        candlestick_ids = self.all_candlestick_data["candlestick_id"]
        has_blank_id = (
            candlestick_ids.isna()
            | candlestick_ids.astype(str).str.strip().eq("")
        ).any()

        # Each annotation key must identify exactly one displayed candlestick.
        if has_blank_id or candlestick_ids.duplicated().any():
            raise ValueError(
                "Candlestick identifiers must be unique and nonempty."
            )

        # Remember where annotations will be saved as the user works.
        self.database_path = Path(database_path)

        # Preserve the available exchange dates so applying a smaller range can
        # switch sessions without rereading the Parquet file.
        self.available_session_dates = sorted(
            set(self.all_candlestick_data["session_date"])
        )

        # Group indices once so crossing sessions does not scan the corpus.
        self._session_indices = self.all_candlestick_data.groupby(
            "session_date", sort=False
        ).indices

        # Use filename-derived boundaries when provided while keeping direct
        # Python callers compatible with the actual prepared session dates.
        self.corpus_start_date = (
            corpus_start_date if corpus_start_date is not None else self.available_session_dates[0]
        )
        self.corpus_end_date = (
            corpus_end_date if corpus_end_date is not None else self.available_session_dates[-1]
        )

        # Calendar navigation requires increasing corpus boundaries.
        if self.corpus_start_date > self.corpus_end_date:
            raise ValueError("Corpus start date cannot be after its end date.")

        # Available sessions must belong to the corpus represented by the date controls.
        if (
            self.available_session_dates[0] < self.corpus_start_date
            or self.available_session_dates[-1] > self.corpus_end_date
        ):
            raise ValueError(
                "Prepared session dates must remain inside corpus boundaries."
            )

        # The current session is replaced whenever the chosen range or active
        # trading date changes.
        self.candlestick_data = pd.DataFrame()

        # Begin with the first candlestick selected for annotation.
        self.active_candlestick_position = 0

        # A new two-key entry begins by asking for the current regime.
        self.is_selecting_current_regime = True

        # Keep incomplete choices in memory until both targets can be saved as
        # one complete annotation.
        self.selected_current_regime: MarketRegime | None = None
        self.selected_anticipated_regime: MarketRegime | None = None

        central_widget = QWidget(self)
        window_layout = QVBoxLayout(central_widget)

        # Keep the date range above the chart so changing the loaded sessions
        # does not require closing or restarting the application.
        date_range_layout = QHBoxLayout()
        window_layout.addLayout(date_range_layout)
        minimum_session_date = self.corpus_start_date
        maximum_session_date = self.corpus_end_date
        minimum_qdate = QDate(
            minimum_session_date.year,
            minimum_session_date.month,
            minimum_session_date.day,
        )
        maximum_qdate = QDate(
            maximum_session_date.year,
            maximum_session_date.month,
            maximum_session_date.day,
        )

        date_range_layout.addWidget(QLabel("Starting date:"))
        self.start_date_input = QDateEdit(minimum_qdate)
        self.start_date_input.setCalendarPopup(True)
        self.start_date_input.setDisplayFormat("yyyy-MM-dd")
        self.start_date_input.setReadOnly(False)
        self.start_date_input.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.start_date_input.setSizePolicy(
            QSizePolicy.Policy.Fixed,
            QSizePolicy.Policy.Fixed,
        )
        self.start_date_input.setFixedWidth(150)
        self.start_date_input.setDateRange(minimum_qdate, maximum_qdate)
        self._configure_date_calendar(self.start_date_input)
        date_range_layout.addWidget(self.start_date_input)

        date_range_layout.addWidget(QLabel("Ending date:"))
        self.end_date_input = QDateEdit(maximum_qdate)
        self.end_date_input.setCalendarPopup(True)
        self.end_date_input.setDisplayFormat("yyyy-MM-dd")
        self.end_date_input.setReadOnly(False)
        self.end_date_input.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.end_date_input.setSizePolicy(
            QSizePolicy.Policy.Fixed,
            QSizePolicy.Policy.Fixed,
        )
        self.end_date_input.setFixedWidth(150)
        self.end_date_input.setDateRange(minimum_qdate, maximum_qdate)
        self._configure_date_calendar(self.end_date_input)
        date_range_layout.addWidget(self.end_date_input)

        # Apply a new range from the already loaded data instead of reopening
        # the Parquet file or replacing the annotation database.
        self.apply_date_range_button = QPushButton("Apply date range")
        self.apply_date_range_button.clicked.connect(self.apply_date_range)
        date_range_layout.addWidget(self.apply_date_range_button)
        date_range_layout.addStretch()

        # Let the user end an annotation run explicitly without relying on the
        # operating system's window controls. Placing it after the stretch
        # keeps this terminating action separate at the top-right corner.
        self.stop_button = QPushButton("Stop")
        self.stop_button.clicked.connect(self.close)
        date_range_layout.addWidget(self.stop_button)

        # Show which trading date is active while the arrow identifies its
        # active candlestick below.
        session_information_layout = QHBoxLayout()
        window_layout.addLayout(session_information_layout)
        self.session_position_label = QLabel("")
        session_information_layout.addWidget(self.session_position_label)
        session_information_layout.addStretch()

        # Display the normalized opening move for whichever candle the arrow
        # currently identifies.
        session_information_layout.addWidget(QLabel("Opening gap:"))
        self.opening_gap_value = QLabel("")
        self.opening_gap_value.setFrameStyle(
            QFrame.Shape.Panel | QFrame.Shadow.Sunken
        )
        self.opening_gap_value.setMinimumWidth(90)
        session_information_layout.addWidget(self.opening_gap_value)

        # Give the plot a visible boundary while its layout continues to expand
        # and contract with the main window.
        self.chart_frame = QFrame(self)
        self.chart_frame.setFrameShape(QFrame.Shape.Box)
        self.chart_frame.setLineWidth(1)
        chart_layout = QVBoxLayout(self.chart_frame)
        chart_layout.setContentsMargins(0, 0, 0, 0)

        self.chart = CandlestickChart(
            self.chart_frame,
            timestamp_column=self.timestamp_column,
        )
        self.chart.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        chart_layout.addWidget(self.chart)
        window_layout.addWidget(self.chart_frame, stretch=1)

        # Keep both targets together at the lower-left so the chart retains
        # most of the horizontal space and the paired judgment reads naturally.
        regime_layout = QHBoxLayout()
        regime_layout.setSpacing(16)
        window_layout.addLayout(regime_layout)

        # Place each value below its label so their left edges remain aligned.
        current_regime_layout = QVBoxLayout()
        current_regime_layout.setSpacing(3)
        self.current_regime_label = QLabel("Current regime:")
        current_regime_layout.addWidget(self.current_regime_label)
        self.current_regime_value = QLabel(UNSELECTED_REGIME_TEXT)
        self.current_regime_value.setFrameStyle(
            QFrame.Shape.Panel | QFrame.Shadow.Sunken
        )
        self.current_regime_value.setFixedWidth(130)
        self.current_regime_value.setAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
        )
        current_regime_layout.addWidget(self.current_regime_value)
        regime_layout.addLayout(current_regime_layout)

        # A separate classification head is planned for this future target.
        anticipated_regime_layout = QVBoxLayout()
        anticipated_regime_layout.setSpacing(3)
        self.anticipated_regime_label = QLabel("Anticipated regime:")
        anticipated_regime_layout.addWidget(self.anticipated_regime_label)
        self.anticipated_regime_value = QLabel(UNSELECTED_REGIME_TEXT)
        self.anticipated_regime_value.setFrameStyle(
            QFrame.Shape.Panel | QFrame.Shadow.Sunken
        )
        self.anticipated_regime_value.setFixedWidth(130)
        self.anticipated_regime_value.setAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
        )
        anticipated_regime_layout.addWidget(self.anticipated_regime_value)
        regime_layout.addLayout(anticipated_regime_layout)
        regime_layout.addStretch()

        # Explain the complete keyboard workflow while identifying which of the
        # two choices the next number key will fill.
        self.selection_prompt = QLabel("")
        self.selection_prompt.setWordWrap(True)
        window_layout.addWidget(self.selection_prompt)

        self.setCentralWidget(central_widget)

        # Begin with the complete available range and its first trading session.
        self.annotation_store = AnnotationStore(self.database_path)
        self._annotation_load_failed = False

        # Initial navigation can fail after SQLite opens, so protect that connection during
        # setup.
        try:
            self.apply_date_range()

        # Release the store when construction fails because no window will exist to close it
        # later.
        except Exception:
            # Release the window-owned database connection before completing window teardown.
            self.annotation_store.close()
            raise

        # Attach navigation to the chart so date fields can use their own arrow
        # keys whenever they hold keyboard focus.
        self.previous_candlestick_shortcut = QShortcut(
            QKeySequence("Left"),
            self.chart,
        )
        self.previous_candlestick_shortcut.setContext(
            Qt.ShortcutContext.WidgetShortcut
        )

        # Reuse the controller action so keyboard and later button navigation
        # always follow the same session-boundary rules.
        self.previous_candlestick_shortcut.activated.connect(
            self.move_to_previous_candlestick
        )

        # Give the matching right-arrow shortcut the same chart-only scope.
        self.next_candlestick_shortcut = QShortcut(
            QKeySequence("Right"),
            self.chart,
        )
        self.next_candlestick_shortcut.setContext(
            Qt.ShortcutContext.WidgetShortcut
        )

        # Connect the shortcut to the same tested forward-navigation action.
        self.next_candlestick_shortcut.activated.connect(
            self.move_to_next_candlestick
        )

        # Retain each shortcut while limiting number-key annotation to a chart
        # that currently holds keyboard focus.
        self.regime_shortcuts: dict[str, QShortcut] = {}

        # Bind each key to its own regime now so later key presses cannot all select the final
        # loop value.
        for number_key, regime in REGIME_SHORTCUT_OPTIONS.items():
            regime_shortcut = QShortcut(QKeySequence(number_key), self.chart)
            regime_shortcut.setContext(Qt.ShortcutContext.WidgetShortcut)
            regime_shortcut.activated.connect(
                partial(self.select_regime, regime)
            )
            self.regime_shortcuts[number_key] = regime_shortcut

    @staticmethod
    def _configure_date_calendar(date_input: QDateEdit) -> None:
        """Make corpus boundaries clear in an editable calendar control.

        Args:
            date_input: Date field whose minimum and maximum dates are set.
        """

        # Customize the existing calendar so date selection keeps the field's configured bounds.
        calendar = date_input.calendarWidget()
        calendar.setMinimumWidth(300)

        # Keep the calendar's navigation bar visible so the month and year can
        # be selected through the field's dropdown instead of typed manually.
        calendar.setNavigationBarVisible(True)

        # Qt already presents the month as a non-editable menu. Give the year
        # the same interaction so neither calendar heading turns into a text
        # field that can accept an invalid partial value.
        month_button = calendar.findChild(
            QToolButton,
            "qt_calendar_monthbutton",
        )
        year_button = calendar.findChild(
            QToolButton,
            "qt_calendar_yearbutton",
        )
        year_editor = calendar.findChild(QSpinBox, "qt_calendar_yearedit")

        # The customized navigation depends on these Qt controls being present.
        if month_button is None or year_button is None or year_editor is None:
            raise RuntimeError("Qt calendar navigation controls are unavailable.")

        month_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        month_button.setFixedWidth(105)
        year_menu = QMenu(year_button)

        # Offer only years covered by this corpus and capture each year separately in its
        # callback.
        for year in range(
            date_input.minimumDate().year(),
            date_input.maximumDate().year() + 1,
        ):
            year_action = year_menu.addAction(str(year))
            year_action.triggered.connect(
                lambda checked=False, selected_year=year: calendar.setCurrentPage(
                    selected_year,
                    calendar.monthShown(),
                )
            )

        year_button.setMenu(year_menu)
        year_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        year_button.setFixedWidth(75)
        year_editor.hide()

        # The complete month and year button surfaces open their menus, so an
        # extra indicator is unnecessary. Centering the text keeps both compact
        # controls balanced in the calendar heading.
        calendar.setStyleSheet("""
            QToolButton#qt_calendar_monthbutton,
            QToolButton#qt_calendar_yearbutton {
                padding: 0px;
                text-align: center;
            }
            QToolButton#qt_calendar_monthbutton::menu-indicator,
            QToolButton#qt_calendar_yearbutton::menu-indicator {
                image: none;
                width: 0px;
            }
            """)

        # QDateEdit's date range prevents out-of-corpus cells from being
        # selected. Qt's disabled palette does not reliably color calendar
        # cells, so format the dates shown on every visited page explicitly.
        format_boundaries = partial(
            AnnotationWindow._format_calendar_boundaries,
            calendar,
            date_input.minimumDate(),
            date_input.maximumDate(),
        )
        calendar.currentPageChanged.connect(format_boundaries)
        format_boundaries(calendar.yearShown(), calendar.monthShown())

    @staticmethod
    def _format_calendar_boundaries(
        calendar: QCalendarWidget,
        minimum_date: QDate,
        maximum_date: QDate,
        visible_year: int,
        visible_month: int,
    ) -> None:
        """Color visible dates outside the corpus red.

        Args:
            calendar: Qt calendar receiving per-date text formats.
            minimum_date: First selectable corpus date.
            maximum_date: Last selectable corpus date.
            visible_year: Year currently displayed by the calendar.
            visible_month: Month currently displayed by the calendar.
        """

        # Include neighboring-month cells because Qt displays them in the same
        # six-week grid as the selected month.
        first_visible_date = QDate(visible_year, visible_month, 1).addDays(-7)
        last_visible_date = QDate(
            visible_year,
            visible_month,
            QDate(visible_year, visible_month, 1).daysInMonth(),
        ).addDays(7)
        red_date_format = QTextCharFormat()
        red_date_format.setForeground(QColor("red"))

        visible_date = first_visible_date

        # Mark every displayed calendar cell, including adjacent-month dates in the visible
        # grid.
        while visible_date <= last_visible_date:
            # Calendar cells outside the permitted corpus should not appear selectable.
            if visible_date < minimum_date or visible_date > maximum_date:
                calendar.setDateTextFormat(visible_date, red_date_format)

            visible_date = visible_date.addDays(1)

    def apply_date_range(self) -> None:
        """Use the selected inclusive dates without reopening the GUI."""

        # Convert the first Qt date explicitly so type checking can prove it is
        # comparable with the exchange-local Python session dates.
        starting_qdate = self.start_date_input.date()
        starting_date = date(
            starting_qdate.year(),
            starting_qdate.month(),
            starting_qdate.day(),
        )

        # Apply the same explicit conversion to the inclusive ending boundary.
        ending_qdate = self.end_date_input.date()
        ending_date = date(
            ending_qdate.year(),
            ending_qdate.month(),
            ending_qdate.day(),
        )

        # Keep the current chart intact when the controls describe an invalid
        # range, allowing the user to correct the dates without losing context.
        if starting_date > ending_date:
            self.session_position_label.setText(
                "Starting date must not be after ending date."
            )

            return

        # Retain only actual prepared trading dates, naturally skipping
        # weekends, holidays, and rejected sessions inside the calendar range.
        selected_session_dates = [
            session_date
            for session_date in self.available_session_dates
            if starting_date <= session_date <= ending_date
        ]

        # A range without eligible sessions cannot provide an active annotation candle.
        if not selected_session_dates:
            self.session_position_label.setText(
                "No prepared trading sessions are inside this date range."
            )

            return

        # Restart navigation at the beginning of the newly selected range.
        self.selected_session_dates = selected_session_dates
        self.active_session_position = 0
        self._load_active_session(candlestick_position=0)

        # Return keyboard control to annotation after the user applies dates.
        self.chart.setFocus(Qt.FocusReason.OtherFocusReason)

    def _load_active_session(self, *, candlestick_position: int) -> None:
        """Draw the active trading session at the requested candle.

        Args:
            candlestick_position: Zero-based candle to select after loading.
        """

        # Use the session position to retrieve the exact exchange date currently
        # selected within the inclusive GUI range.
        active_session_date = self.selected_session_dates[
            self.active_session_position
        ]
        self.candlestick_data = self.all_candlestick_data.iloc[
            self._session_indices[active_session_date]
        ].reset_index(drop=True)
        self.active_candlestick_position = candlestick_position

        # Make session progress visible without assigning equal lengths to
        # regular sessions and exchange-scheduled half days.
        self.session_position_label.setText(
            f"Session {active_session_date.isoformat()} "
            f"({self.active_session_position + 1} of "
            f"{len(self.selected_session_dates)})"
        )

        # Redraw the complete active day and restore any prior choices for the
        # candle identified by the arrow.
        self.chart.draw_session(
            self.candlestick_data,
            self.active_candlestick_position,
        )
        self._load_active_annotation()

    def select_regime(self, regime: MarketRegime) -> None:
        """Apply one number-key choice to the scalar currently requested.

        Args:
            regime: Bull, bear, or range choice mapped from the pressed key.
        """

        # Do not overwrite an existing judgment when its saved state could not be loaded.
        if self._annotation_load_failed:
            return

        # The first number describes the regime after the active candlestick.
        if self.is_selecting_current_regime:
            self.selected_current_regime = regime

            # Clear any old anticipated choice because changing the current regime
            # begins a new two-key annotation for this candle.
            self.selected_anticipated_regime = None
            self.current_regime_value.setText(
                self._format_regime_choice(regime)
            )
            self.anticipated_regime_value.setText(UNSELECTED_REGIME_TEXT)
            self.is_selecting_current_regime = False
            self._update_selection_prompt()

            return

        # The second number completes the future target for the same candle.
        self.selected_anticipated_regime = regime
        self.anticipated_regime_value.setText(
            self._format_regime_choice(regime)
        )

        # The current choice must exist because it is always collected first.
        if self.selected_current_regime is None:
            raise RuntimeError("Current regime must be selected first.")

        # Save both targets together so navigation never loads half of a human
        # judgment from the annotation database.
        try:
            self.annotation_store.save(
                CandlestickAnnotation(
                    candlestick_id=self._active_candlestick_id(),
                    current_regime=self.selected_current_regime,
                    anticipated_regime=self.selected_anticipated_regime,
                )
            )

        # Keep the current candle selected when saving fails so the user can retry the same
        # judgment.
        except sqlite3.Error as error:
            # Stay on this candle and keep both choices available for retry.
            self.statusBar().showMessage(
                f"Not saved: {error}. Press the anticipated regime key to retry."
            )

            return

        self.statusBar().clearMessage()

        # Begin the next two-key annotation with its current-regime choice.
        self.is_selecting_current_regime = True

        # Reload only at the end of the complete selected range so the final
        # saved choices remain visible; every earlier candle advances normally,
        # including from one session into the next.
        is_final_candlestick = self.active_candlestick_position >= len(self.candlestick_data) - 1
        is_final_session = self.active_session_position >= len(
            self.selected_session_dates
        ) - 1

        # Stop after saving the final candle instead of advancing beyond the selected data.
        if is_final_candlestick and is_final_session:
            self._load_active_annotation()
        else:
            self.move_to_next_candlestick()

    def _active_candlestick_id(self) -> str:
        """Return the stable identifier for the active candlestick."""

        # Use the persistent market-data identifier instead of the DataFrame's
        # replaceable row index.
        candlestick_id = self.candlestick_data.iloc[
            self.active_candlestick_position
        ][
            "candlestick_id"
        ]

        return str(candlestick_id)

    @staticmethod
    def _format_regime_choice(regime: MarketRegime) -> str:
        """Format a saved regime with its matching number key.

        Args:
            regime: Regime selected for one annotation scalar.

        Returns:
            Number and regime shown in the GUI.
        """

        # Show both parts so the display confirms the exact key that was chosen.
        number_key = REGIME_KEYS[regime]

        return f"{number_key} - {regime.value.title()}"

    def _load_active_annotation(self) -> None:
        """Display the saved choices for the active candlestick."""

        # Convert the stored ratio into a signed percentage that is easier to
        # interpret while reading the active candle's real price geometry.
        active_opening_gap = float(
            self.candlestick_data.iloc[self.active_candlestick_position][
                "open_gap"
            ]
        )
        self.opening_gap_value.setText(f"{active_opening_gap:+.3%}")

        # Read only the active candle so navigation remains inexpensive even
        # when the annotation database eventually contains many sessions.
        try:
            annotation = self.annotation_store.load(self._active_candlestick_id())

        # A failed read must not appear to be an unannotated candle and permit accidental
        # overwrite.
        except (sqlite3.Error, ValueError) as error:
            # An unreadable judgment must not appear to be an unlabeled candle.
            self._annotation_load_failed = True
            self.selected_current_regime = None
            self.selected_anticipated_regime = None
            self.current_regime_value.setText("Unavailable")
            self.anticipated_regime_value.setText("Unavailable")
            self.selection_prompt.setText("Reload this candle before annotating.")
            self.statusBar().showMessage(f"Could not load annotation: {error}")

            return

        self._annotation_load_failed = False
        self.statusBar().clearMessage()

        # Navigation always starts a fresh two-key entry if the user decides to
        # replace the choices currently displayed.
        self.is_selecting_current_regime = True

        # An untouched candle has no defensible default label, so both boxes
        # remain blank until the annotator supplies one.
        if annotation is None:
            self.selected_current_regime = None
            self.selected_anticipated_regime = None
            self.current_regime_value.setText(UNSELECTED_REGIME_TEXT)
            self.anticipated_regime_value.setText(UNSELECTED_REGIME_TEXT)
            self._update_selection_prompt()

            return

        # Restore both stored choices when revisiting an annotated candle.
        self.selected_current_regime = annotation.current_regime
        self.selected_anticipated_regime = annotation.anticipated_regime
        self.current_regime_value.setText(
            self._format_regime_choice(annotation.current_regime)
        )
        self.anticipated_regime_value.setText(
            self._format_regime_choice(annotation.anticipated_regime)
        )
        self._update_selection_prompt()

    def _update_selection_prompt(self) -> None:
        """Show which scalar the next number key will fill."""

        # State the physical action, key meanings, save point, and automatic
        # movement so a first-time annotator can complete the workflow unaided.
        if self.is_selecting_current_regime:
            instruction = (
                "Step 1 of 2: Click the chart, then press 1 for Bull, "
                "2 for Bear, or 3 for Range to choose the current regime."
            )
        else:
            instruction = (
                "Step 2 of 2: Press 1 for Bull, 2 for Bear, or 3 for Range "
                "to choose the anticipated regime. This saves both choices "
                "and moves to the next candle."
            )

        self.selection_prompt.setText(instruction)

    def move_to_next_candlestick(self) -> None:
        """Select the next candlestick when one exists."""

        # Move from the final candle into the next selected trading date so one
        # annotation run can continue across the complete chosen date range.
        if self.active_candlestick_position >= len(self.candlestick_data) - 1:
            # The next-session action must stay within the selected session range.
            if self.active_session_position >= len(
                self.selected_session_dates
            ) - 1:
                return

            self.active_session_position += 1
            self._load_active_session(candlestick_position=0)

            return

        self.active_candlestick_position += 1

        # Move only the arrow; the session geometry has not changed.
        self.chart.set_active_candlestick(self.active_candlestick_position)

        # Replace the previous candle's display with this candle's saved
        # choices, or blank boxes when it has not been annotated.
        self._load_active_annotation()

    def move_to_previous_candlestick(self) -> None:
        """Select the previous candlestick when one exists."""

        # Move from the opening candle to the end of the previous selected
        # session when one exists in the current date range.
        if self.active_candlestick_position <= 0:
            # The previous-session action must not move before the first selected session.
            if self.active_session_position <= 0:
                return

            self.active_session_position -= 1
            previous_session_date = self.selected_session_dates[
                self.active_session_position
            ]
            previous_session_size = len(self._session_indices[previous_session_date])
            self._load_active_session(
                candlestick_position=previous_session_size - 1
            )

            return

        self.active_candlestick_position -= 1

        # Move only the arrow; the session geometry has not changed.
        self.chart.set_active_candlestick(self.active_candlestick_position)

        # Replace the next candle's display with this candle's saved choices,
        # or blank boxes when it has not been annotated.
        self._load_active_annotation()

    def closeEvent(self, event: QCloseEvent) -> None:
        """Release the window's database connection on close.

        Args:
            event: Qt close event passed to the base window handler.
        """

        # Release the window-owned database connection before completing window teardown.
        self.annotation_store.close()
        super().closeEvent(event)
