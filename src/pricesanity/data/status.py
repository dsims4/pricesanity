"""Convert Databento status information into RTH session schedules."""

from datetime import date, time

import pandas as pd
from pandas.api.types import is_bool_dtype


# Define normalized status fields needed to identify scheduled market closes.
REQUIRED_STATUS_COLUMNS = ("is_trading",)

# Define Databento condition fields needed to preserve daily data quality.
REQUIRED_CONDITION_COLUMNS = ("date", "condition")


def build_session_schedule(
    scheduled_status_data: pd.DataFrame,
    data_conditions: pd.DataFrame,
    *,
    timestamp_column: str,
    session_timezone: str,
    session_start_time: str,
    session_end_time: str,
) -> pd.DataFrame:
    """Build date-specific RTH boundaries from scheduled status changes.

    Args:
        scheduled_status_data: Normalized scheduled trading-state changes.
        data_conditions: Databento data condition for each date.
        timestamp_column: Column containing status-event timestamps.
        session_timezone: Timezone used to define each RTH date.
        session_start_time: Configured RTH starting time.
        session_end_time: Maximum configured RTH ending time.

    Returns:
        Session date, open time, close time, and data condition for each
        trading date.

    Raises:
        ValueError: If required fields, timestamps, conditions, or session
            settings are invalid.
    """
    # Combine the configurable timestamp name with required status fields.
    required_status_columns = (
        timestamp_column,
        *REQUIRED_STATUS_COLUMNS,
    )

    # Find missing status fields to check if the function can proceed.
    missing_status_columns = set(required_status_columns) - set(
        scheduled_status_data.columns
    )

    # Reject status data that cannot identify closing transitions.
    if missing_status_columns:
        # Sort and join missing names so the error is stable and readable.
        missing_column_names = ", ".join(sorted(missing_status_columns))
        raise ValueError(
            f"Status data is missing required columns: {missing_column_names}"
        )

    # Find missing condition fields to check if the function can proceed.
    missing_condition_columns = set(REQUIRED_CONDITION_COLUMNS) - set(
        data_conditions.columns
    )

    # Reject condition data that cannot describe quality by date.
    if missing_condition_columns:
        # Sort and join missing names so the error is stable and readable.
        missing_column_names = ", ".join(sorted(missing_condition_columns))
        raise ValueError(
            f"Condition data is missing required columns: "
            f"{missing_column_names}"
        )

    # Copy status data so timestamp parsing does not modify the caller's data.
    parsed_status_data = scheduled_status_data.copy()

    # Parse status timestamps in UTC for comparison with configured boundaries.
    parsed_status_data[timestamp_column] = pd.to_datetime(
        parsed_status_data[timestamp_column],
        errors="coerce",
        utc=True,
    )

    # Reject any status timestamp that could not be interpreted.
    if parsed_status_data[timestamp_column].isna().any():
        raise ValueError("Status data contains invalid timestamps.")

    # Require normalized booleans from Databento or the user for clear states.
    if not is_bool_dtype(parsed_status_data["is_trading"]):
        raise ValueError("Status is_trading values must be booleans.")

    # Copy condition data so date parsing does not modify the caller's data.
    parsed_data_conditions = data_conditions.copy()

    # Parse condition dates without attaching a time or timezone.
    parsed_data_conditions["session_date"] = pd.to_datetime(
        parsed_data_conditions["date"],
        errors="coerce",
    ).dt.date

    # Reject any condition date that could not be interpreted.
    if parsed_data_conditions["session_date"].isna().any():
        raise ValueError("Condition data contains invalid dates.")

    # Require one quality condition for each date to prevent ambiguous matches.
    if parsed_data_conditions["session_date"].duplicated().any():
        raise ValueError("Condition data contains duplicate dates.")

    # Normalize condition text for direct comparison in session filtering.
    parsed_data_conditions["condition"] = (
        parsed_data_conditions["condition"].astype(str).str.lower()
    )

    # Index conditions by date for quick lookup while constructing sessions.
    condition_by_date = parsed_data_conditions.set_index("session_date")[
        "condition"
    ]

    # Parse the configured RTH boundaries for date-specific timestamps.
    parsed_session_start_time = time.fromisoformat(session_start_time)
    parsed_session_end_time = time.fromisoformat(session_end_time)

    # Reject configured RTH boundaries that do not describe a same-day session.
    if parsed_session_start_time >= parsed_session_end_time:
        raise ValueError("RTH session start must occur before its end.")

    # Keep only False transitions because they identify scheduled market closes.
    scheduled_close_data = parsed_status_data.loc[
        ~parsed_status_data["is_trading"]
    ].copy()

    # Convert closes to session time before assigning their local trading dates.
    session_close_timestamps = scheduled_close_data[
        timestamp_column
    ].dt.tz_convert(session_timezone)

    # Assign each scheduled close to the date it closes in the session timezone.
    scheduled_close_data["session_date"] = session_close_timestamps.dt.date

    # Collect one output record for each date containing an RTH session.
    session_schedule_rows: list[dict[str, object]] = []

    # Build date-specific boundaries from each date's earliest scheduled close.
    for session_date, daily_close_data in scheduled_close_data.groupby(
        "session_date",
        sort=True,
    ):
        # Confirm the grouping value is a date before combining it with a time.
        if not isinstance(session_date, date):
            raise ValueError("Status data contains an invalid session date.")

        # Attach the session timezone to the configured RTH opening boundary.
        configured_session_open = pd.Timestamp.combine(
            session_date,
            parsed_session_start_time,
        ).tz_localize(session_timezone)

        # Attach the session timezone to the maximum RTH closing boundary.
        configured_session_close = pd.Timestamp.combine(
            session_date,
            parsed_session_end_time,
        ).tz_localize(session_timezone)

        # Select the first scheduled close in case the date contains later ones.
        scheduled_session_close = daily_close_data[timestamp_column].min()

        # Clip a normal close to configured RTH while preserving an early close.
        effective_session_close = min(
            configured_session_close.tz_convert("UTC"),
            scheduled_session_close,
        )

        # Omit dates whose scheduled close occurs before RTH begins.
        if effective_session_close <= configured_session_open.tz_convert("UTC"):
            continue

        # Preserve unknown quality as unusable instead of assuming availability.
        data_condition = condition_by_date.get(session_date, "unknown")

        # Store UTC boundaries in the schema consumed by session validation.
        session_schedule_rows.append(
            {
                "session_date": session_date,
                "session_open": configured_session_open.tz_convert("UTC"),
                "session_close": effective_session_close,
                "data_condition": data_condition,
            }
        )

    # Return stable columns even when no scheduled date contains an RTH session.
    return pd.DataFrame(
        session_schedule_rows,
        columns=(
            "session_date",
            "session_open",
            "session_close",
            "data_condition",
        ),
    )
