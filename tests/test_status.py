import pandas as pd
import pytest

from pricesanity.data.status import (
    build_session_schedule,
    extract_session_transitions,
)


def test_extract_session_transitions_keeps_scheduled_state_changes() -> None:
    # Mix valid boundaries with a midnight snapshot, a repeated state, an
    # unscheduled halt, and an unrelated trading event.
    status_data = pd.DataFrame(
        {
            "ts_event": [
                "2024-12-23 00:00:00Z",
                "2024-12-23 22:00:00Z",
                "2024-12-23 22:01:00Z",
                "2024-12-23 22:02:00Z",
                "2024-12-23 22:03:00Z",
                "2024-12-23 23:00:00Z",
            ],
            "reason": [
                "scheduled",
                "scheduled",
                "scheduled",
                "operational",
                "scheduled",
                "scheduled",
            ],
            "trading_event": ["none", "none", "none", "none", 2, 0],
            "is_trading": ["Y", "N", "N", "Y", "Y", "Y"],
        }
    )

    # Extract the only two records that prove a scheduled close and later open.
    session_transitions = extract_session_transitions(
        status_data,
        timestamp_column="ts_event",
    )

    # The result must preserve both transition times and normalize their states.
    assert session_transitions["ts_event"].tolist() == list(
        pd.to_datetime(
            ["2024-12-23 22:00:00Z", "2024-12-23 23:00:00Z"],
            utc=True,
        )
    )
    assert session_transitions["is_trading"].tolist() == [False, True]


def test_extract_session_transitions_rejects_unknown_trading_state() -> None:
    # An unknown state cannot prove which side of a session boundary is active.
    status_data = pd.DataFrame(
        {
            "ts_event": ["2024-12-23 22:00:00Z"],
            "reason": [1],
            "trading_event": [0],
            "is_trading": ["~"],
        }
    )

    # Reject the ambiguous state before it can enter session construction.
    with pytest.raises(ValueError, match="must be Y, N, or booleans"):
        extract_session_transitions(
            status_data,
            timestamp_column="ts_event",
        )


def test_build_session_schedule_preserves_scheduled_early_close() -> None:
    # Reproduce scheduled ES trading-state changes around Christmas Eve 2024.
    scheduled_status_data = pd.DataFrame(
        {
            "ts_event": pd.to_datetime(
                [
                    "2024-12-22 23:00:00Z",
                    "2024-12-23 22:00:00Z",
                    "2024-12-23 23:00:00Z",
                    "2024-12-24 18:15:00Z",
                ],
                utc=True,
            ),
            "is_trading": [True, False, True, False],
        }
    )

    # Mark both trading dates as available in Databento's condition data.
    data_conditions = pd.DataFrame(
        {
            "date": ["2024-12-23", "2024-12-24"],
            "condition": ["available", "available"],
        }
    )

    # Build RTH boundaries using the normal close as a maximum.
    session_schedule = build_session_schedule(
        scheduled_status_data,
        data_conditions,
        timestamp_column="ts_event",
        session_timezone="America/New_York",
        session_start_time="09:30",
        session_end_time="16:15",
    )

    # Create the UTC boundaries expected for one regular and one early close.
    expected_session_opens = pd.to_datetime(
        [
            "2024-12-23 14:30:00Z",
            "2024-12-24 14:30:00Z",
        ],
        utc=True,
    )
    expected_session_closes = pd.to_datetime(
        [
            "2024-12-23 21:15:00Z",
            "2024-12-24 18:15:00Z",
        ],
        utc=True,
    )

    # Clip the regular close to 16:15 while retaining the scheduled early close.
    assert session_schedule["session_open"].tolist() == list(
        expected_session_opens
    )
    assert session_schedule["session_close"].tolist() == list(
        expected_session_closes
    )
    assert session_schedule["data_condition"].tolist() == [
        "available",
        "available",
    ]


def test_build_session_schedule_preserves_degraded_condition() -> None:
    # Create one scheduled close for a normal trading date.
    scheduled_status_data = pd.DataFrame(
        {
            "ts_event": pd.to_datetime(
                ["2026-09-09 21:00:00Z"],
                utc=True,
            ),
            "is_trading": [False],
        }
    )

    # Mark the same date degraded so downstream filtering can reject it.
    data_conditions = pd.DataFrame(
        {
            "date": ["2026-09-09"],
            "condition": ["DEGRADED"],
        }
    )

    # Build the session schedule while normalizing condition capitalization.
    session_schedule = build_session_schedule(
        scheduled_status_data,
        data_conditions,
        timestamp_column="ts_event",
        session_timezone="America/New_York",
        session_start_time="09:30",
        session_end_time="16:15",
    )

    # Retain the degraded label rather than treating the date as unavailable.
    assert session_schedule.loc[0, "data_condition"] == "degraded"


def test_build_session_schedule_omits_close_before_rth() -> None:
    # Create a scheduled market close that occurs before the configured RTH open.
    scheduled_status_data = pd.DataFrame(
        {
            "ts_event": pd.to_datetime(
                ["2026-04-03 12:15:00Z"],
                utc=True,
            ),
            "is_trading": [False],
        }
    )

    # Supply an available condition even though no RTH session occurred.
    data_conditions = pd.DataFrame(
        {
            "date": ["2026-04-03"],
            "condition": ["available"],
        }
    )

    # Build the schedule using a 09:30 New York RTH opening boundary.
    session_schedule = build_session_schedule(
        scheduled_status_data,
        data_conditions,
        timestamp_column="ts_event",
        session_timezone="America/New_York",
        session_start_time="09:30",
        session_end_time="16:15",
    )

    # Omit the date because its market close occurred before RTH began.
    assert session_schedule.empty


def test_build_session_schedule_rejects_nonboolean_trading_state() -> None:
    # Create an unnormalized Databento trading-state value.
    scheduled_status_data = pd.DataFrame(
        {
            "ts_event": pd.to_datetime(
                ["2026-09-09 21:00:00Z"],
                utc=True,
            ),
            "is_trading": ["N"],
        }
    )

    # Supply valid condition data so validation reaches the trading state.
    data_conditions = pd.DataFrame(
        {
            "date": ["2026-09-09"],
            "condition": ["available"],
        }
    )

    # Require SDK-specific status values to be normalized before schedule use.
    with pytest.raises(ValueError, match="must be booleans"):
        build_session_schedule(
            scheduled_status_data,
            data_conditions,
            timestamp_column="ts_event",
            session_timezone="America/New_York",
            session_start_time="09:30",
            session_end_time="16:15",
        )
