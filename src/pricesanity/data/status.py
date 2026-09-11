"""Convert Databento status information into RTH session schedules."""

from datetime import date, time

import pandas as pd
from pandas.api.types import is_bool_dtype


# Schedule construction needs only the normalized trading state because
# extraction has already removed unrelated Databento status events.
REQUIRED_STATUS_COLUMNS = ("is_trading",)

# Each condition must include its date so the resulting session can preserve
# Databento's quality decision for later acceptance or rejection.
REQUIRED_CONDITION_COLUMNS = ("date", "condition")

# Raw status records need their cause, related event, and resulting trading
# state to distinguish scheduled session boundaries from other market updates.
REQUIRED_RAW_STATUS_COLUMNS = ("reason", "trading_event", "is_trading")


def build_session_schedule(
    scheduled_status_data: pd.DataFrame,
    data_conditions: pd.DataFrame,
    *,
    timestamp_column: str,
    session_timezone: str,
    session_start_time: str,
    session_end_time: str,
) -> pd.DataFrame:
    """Build each trading date's RTH schedule and preserve its data condition.

    Args:
        scheduled_status_data: Scheduled trading transitions to organize.
        data_conditions: Databento data condition for each trading date.
        timestamp_column: Column containing transition timestamps.
        session_timezone: Timezone defining each trading date.
        session_start_time: Earliest configured RTH time.
        session_end_time: Latest configured RTH time.

    Returns:
        Session date, open time, close time, and data condition for each
        trading date.

    Raises:
        ValueError: If required fields, timestamps, conditions, or session
            settings are invalid.
    """
    # The timestamp places a transition on a trading date, while its boolean
    # state identifies whether that transition closes the session.
    required_status_columns = (
        timestamp_column,
        *REQUIRED_STATUS_COLUMNS,
    )

    # Find every absent status field before building dates because incomplete
    # transitions cannot define reliable session boundaries.
    missing_status_columns = set(required_status_columns) - set(
        scheduled_status_data.columns
    )

    # Without both time and trading state, the function cannot determine when a
    # scheduled session ended.
    if missing_status_columns:
        # Include every missing field in one stable message so the transition
        # data can be corrected without repeated validation attempts.
        missing_column_names = ", ".join(sorted(missing_status_columns))
        raise ValueError(
            f"Status data is missing required columns: {missing_column_names}"
        )

    # Find every absent condition field before matching quality information to
    # the dates created from scheduled closes.
    missing_condition_columns = set(REQUIRED_CONDITION_COLUMNS) - set(
        data_conditions.columns
    )

    # Without both date and condition, trustworthy and degraded sessions cannot
    # be separated later.
    if missing_condition_columns:
        # Include every missing field in one stable message so the condition
        # data can be corrected without repeated validation attempts.
        missing_column_names = ", ".join(sorted(missing_condition_columns))
        raise ValueError(
            f"Condition data is missing required columns: "
            f"{missing_column_names}"
        )

    # Work from a separate table because UTC parsing should not alter the
    # extracted transitions supplied by the caller.
    parsed_status_data = scheduled_status_data.copy()

    # Convert transitions to UTC so their absolute times can be compared with
    # the date-specific RTH boundaries created below.
    parsed_status_data[timestamp_column] = pd.to_datetime(
        parsed_status_data[timestamp_column],
        errors="coerce",
        utc=True,
    )

    # A transition without a valid timestamp cannot be assigned to a trading
    # date or used as its closing boundary.
    if parsed_status_data[timestamp_column].isna().any():
        raise ValueError("Status data contains invalid timestamps.")

    # Require booleans because truth and falsehood must unambiguously represent
    # trading and not trading before closing transitions are selected.
    if not is_bool_dtype(parsed_status_data["is_trading"]):
        raise ValueError("Status is_trading values must be booleans.")

    # Work from a separate condition table because date parsing and text
    # normalization should not alter the caller's Databento metadata.
    parsed_data_conditions = data_conditions.copy()

    # Keep only the calendar date because one quality condition applies to the
    # entire session rather than a particular moment within it.
    parsed_data_conditions["session_date"] = pd.to_datetime(
        parsed_data_conditions["date"],
        errors="coerce",
    ).dt.date

    # An invalid date cannot be matched safely with a scheduled trading session.
    if parsed_data_conditions["session_date"].isna().any():
        raise ValueError("Condition data contains invalid dates.")

    # More than one condition for a date would leave the session's accepted
    # quality state unclear.
    if parsed_data_conditions["session_date"].duplicated().any():
        raise ValueError("Condition data contains duplicate dates.")

    # Normalize capitalization so equivalent Databento condition labels receive
    # the same decision during session filtering.
    parsed_data_conditions["condition"] = (
        parsed_data_conditions["condition"].astype(str).str.lower()
    )

    # Index conditions by date because the schedule loop retrieves one quality
    # label for each date derived from a close.
    condition_by_date = parsed_data_conditions.set_index("session_date")[
        "condition"
    ]

    # Convert configured clock text into time values that can be attached to
    # each scheduled trading date.
    parsed_session_start_time = time.fromisoformat(session_start_time)
    parsed_session_end_time = time.fromisoformat(session_end_time)

    # This project models same-day RTH sessions, so an open at or after the close
    # cannot define a valid configured window.
    if parsed_session_start_time >= parsed_session_end_time:
        raise ValueError("RTH session start must occur before its end.")

    # Session length is determined by its ending boundary, so opening
    # transitions are no longer needed after extraction.
    scheduled_close_data = parsed_status_data.loc[
        ~parsed_status_data["is_trading"]
    ].copy()

    # A UTC close may fall on a different calendar date elsewhere, so convert it
    # to the session timezone before deciding which trading date it ends.
    session_close_timestamps = scheduled_close_data[
        timestamp_column
    ].dt.tz_convert(session_timezone)

    # Store the local closing date so all transitions belonging to the same
    # trading session can be considered together.
    scheduled_close_data["session_date"] = session_close_timestamps.dt.date

    # Build rows separately because each date can have a different close and
    # data condition.
    session_schedule_rows: list[dict[str, object]] = []

    # Process closes by local date so a normal session and an official half-day
    # each receive boundaries based on their own schedule.
    for session_date, daily_close_data in scheduled_close_data.groupby(
        "session_date",
        sort=True,
    ):
        # Combining a clock time with an unexpected grouping value would create
        # an invalid session boundary.
        if not isinstance(session_date, date):
            raise ValueError("Status data contains an invalid session date.")

        # Use the configured RTH open for every eligible date because status
        # transitions are used here to discover variable closing times.
        configured_session_open = pd.Timestamp.combine(
            session_date,
            parsed_session_start_time,
        ).tz_localize(session_timezone)

        # Build the latest allowed RTH close so overnight trading cannot enter
        # the intraday model sequence.
        configured_session_close = pd.Timestamp.combine(
            session_date,
            parsed_session_end_time,
        ).tz_localize(session_timezone)

        # The earliest scheduled close ends the eligible session; later closes
        # may belong to trading outside the chosen RTH window.
        scheduled_session_close = daily_close_data[timestamp_column].min()

        # Choose whichever close occurs first so normal sessions stop at the
        # configured RTH end while official half-days retain their early close.
        effective_session_close = min(
            configured_session_close.tz_convert("UTC"),
            scheduled_session_close,
        )

        # A close before the configured open means that no eligible RTH
        # candlestick interval exists for this date.
        if effective_session_close <= configured_session_open.tz_convert("UTC"):
            continue

        # Treat a missing condition as unknown so absent quality evidence can
        # never be mistaken for confirmed availability.
        data_condition = condition_by_date.get(session_date, "unknown")

        # Store one UTC schedule row so session validation can compare these
        # boundaries directly with UTC candlestick timestamps.
        session_schedule_rows.append(
            {
                "session_date": session_date,
                "session_open": configured_session_open.tz_convert("UTC"),
                "session_close": effective_session_close,
                "data_condition": data_condition,
            }
        )

    # Preserve the schedule schema even when no date qualifies so downstream
    # validation can handle an empty result without guessing its columns.
    return pd.DataFrame(
        session_schedule_rows,
        columns=(
            "session_date",
            "session_open",
            "session_close",
            "data_condition",
        ),
    )


def extract_session_transitions(
    status_data: pd.DataFrame,
    *,
    timestamp_column: str,
) -> pd.DataFrame:
    """Extract scheduled opening and closing transitions from raw Databento
    status records.

    Args:
        status_data: Raw Databento status records.
        timestamp_column: Column containing status-event timestamps.

    Returns:
        UTC timestamps and boolean trading states for scheduled changes.

    Raises:
        ValueError: If required fields, timestamps, or trading states are
            invalid.
    """
    # The timestamp locates each status change, while the Databento fields
    # determine whether that change represents a scheduled session transition.
    required_status_columns = (
        timestamp_column,
        *REQUIRED_RAW_STATUS_COLUMNS
    )

    # Find any missing fields before trying to interpret the raw status data,
    # because an incomplete record cannot reliably define a session boundary.
    missing_status_columns = set(required_status_columns) - set(
        status_data.columns
    )

    # Stop before building session transitions when the input cannot describe
    # when a status changed, why it changed, or whether trading became active.
    if missing_status_columns:
        # Include every missing field in the error so the input can be corrected
        # without repeatedly discovering one missing column at a time.
        missing_column_names = ", ".join(sorted(missing_status_columns))

        raise ValueError(
            f"Status data is missing required columns: {missing_column_names}"
        )

    # Copy only the fields used to identify transitions so unrelated Databento
    # metadata does not pass through this function or alter the original data.
    extracted_status_data = status_data.loc[
        :, list(required_status_columns)
    ].copy()

    # Every transition needs one shared timezone so midnight snapshots can be
    # identified and the remaining status changes can be placed in exact order.
    # Invalid timestamps become NaT so they can all be rejected in the next check.
    extracted_status_data[timestamp_column] = pd.to_datetime(
        extracted_status_data[timestamp_column],
        errors="coerce",
        utc=True,
    )

    # A status record without a valid timestamp cannot be ordered or used
    # to define when a trading session opened or closed.
    if extracted_status_data[timestamp_column].isna().any():
        raise ValueError("Status data contains invalid timestamps.")

    # Trading states must be read in chronological order because each
    # transition is identified by comparing a record with the state
    # immediately before it.
    extracted_status_data = extracted_status_data.sort_values(
        timestamp_column,
        kind="stable",
    )

    # Databento repeats the current state at midnight so requests crossing UTC
    # dates still know the active state. These snapshots are not new session
    # transitions and must be removed before comparing changes.
    is_not_midnight_snapshot = (
        extracted_status_data[timestamp_column].dt.time != time.min
    )
    extracted_status_data = extracted_status_data.loc[
        is_not_midnight_snapshot
    ].copy()

    # Databento may provide reason and event codes as enum objects, names, or
    # numbers. Converting each form to lowercase text gives the filters below
    # one consistent representation.
    normalized_reasons = extracted_status_data["reason"].map(
        lambda value: getattr(value, "name", value)
    ).astype(str).str.strip().str.lower()
    normalized_trading_events = extracted_status_data["trading_event"].map(
        lambda value: getattr(value, "name", value)
    ).astype(str).str.strip().str.lower()

    # Only scheduled changes describe normal session boundaries. Unscheduled
    # changes, such as operational halts, must not shorten a planned session.
    is_scheduled_transition = normalized_reasons.isin(("scheduled", "1"))

    # A separate trading event describes another exchange action instead of a
    # direct change between trading and not trading, so it cannot define the
    # opening or closing boundary used by this project.
    has_no_trading_event = normalized_trading_events.isin(("none", "0"))

    # Apply both requirements together so only potential scheduled session
    # openings and closings remain for trading-state normalization.
    extracted_status_data = extracted_status_data.loc[
        is_scheduled_transition & has_no_trading_event
    ].copy()

    # Databento represents active and inactive trading with Y and N, while a
    # caller may supply booleans that have already been normalized. Both valid
    # forms are converted into one boolean type for schedule construction.
    normalized_trading_states = extracted_status_data["is_trading"].map(
        {
            "Y": True,
            "N": False,
            True: True,
            False: False,
        }
    )

    # Databento uses '~' when the state is unknown. An unknown or unsupported
    # value cannot prove whether a scheduled session opened or closed.
    if normalized_trading_states.isna().any():
        raise ValueError("Status is_trading values must be Y, N, or booleans.")

    # Store every accepted state as a boolean so the next record can be compared
    # directly with the trading state immediately before it.
    extracted_status_data["is_trading"] = normalized_trading_states.astype(
        bool
    )

    # Repeated records describe a market that remained in the same state rather
    # than a new opening or closing, so only changes from the prior state remain.
    is_new_trading_state = extracted_status_data["is_trading"].ne(
        extracted_status_data["is_trading"].shift()
    )
    extracted_status_data = extracted_status_data.loc[
        is_new_trading_state,
        [timestamp_column, "is_trading"],
    ]

    # Reset the row labels so the extracted transitions form a clean table for
    # build_session_schedule without retaining gaps from filtered status rows.
    return extracted_status_data.reset_index(drop=True)
