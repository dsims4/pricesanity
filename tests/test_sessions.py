import pandas as pd
import pytest

from pricesanity.data.sessions import filter_complete_sessions


def test_filter_complete_sessions_keeps_regular_and_half_sessions() -> None:
    # Generate the timestamps expected for one regular RTH session.
    regular_session_timestamps = pd.date_range(
        "2026-09-09 09:30:00",
        "2026-09-09 16:00:00",
        freq="5min",
        inclusive="left",
        tz="America/New_York",
    )

    # Generate the timestamps expected for one scheduled early-close session.
    half_session_timestamps = pd.date_range(
        "2026-09-10 09:30:00",
        "2026-09-10 13:00:00",
        freq="5min",
        inclusive="left",
        tz="America/New_York",
    )

    # Combine both valid sessions into one UTC candlestick dataset.
    candlestick_data = pd.DataFrame(
        {
            "ts_event": regular_session_timestamps.append(
                half_session_timestamps
            ).tz_convert("UTC"),
            "session_marker": (
                ["regular"] * len(regular_session_timestamps)
                + ["half"] * len(half_session_timestamps)
            ),
        }
    )

    # Describe each session using boundaries derived from scheduled status data.
    session_schedule = pd.DataFrame(
        {
            "session_open": pd.to_datetime(
                [
                    "2026-09-09 09:30:00-04:00",
                    "2026-09-10 09:30:00-04:00",
                ],
                utc=True,
            ),
            "session_close": pd.to_datetime(
                [
                    "2026-09-09 16:00:00-04:00",
                    "2026-09-10 13:00:00-04:00",
                ],
                utc=True,
            ),
            "data_condition": ["available", "available"],
        }
    )

    # Validate both dates against their own scheduled session boundaries.
    complete_session_data = filter_complete_sessions(
        candlestick_data,
        session_schedule,
        timestamp_column="ts_event",
        target_interval="5min",
    )

    # Keep every candle from both the regular and scheduled half-session.
    assert len(complete_session_data) == len(candlestick_data)
    assert complete_session_data["session_marker"].unique().tolist() == [
        "regular",
        "half",
    ]


def test_filter_complete_sessions_discards_incomplete_and_degraded() -> None:
    # Remove the final timestamp from an otherwise normal scheduled session.
    incomplete_session_timestamps = pd.date_range(
        "2026-09-11 09:30:00",
        "2026-09-11 16:00:00",
        freq="5min",
        inclusive="left",
        tz="America/New_York",
    )[:-1]

    # Generate a complete session whose date is marked degraded by Databento.
    degraded_session_timestamps = pd.date_range(
        "2026-09-14 09:30:00",
        "2026-09-14 16:00:00",
        freq="5min",
        inclusive="left",
        tz="America/New_York",
    )

    # Combine the incomplete and degraded sessions into one UTC dataset.
    candlestick_data = pd.DataFrame(
        {
            "ts_event": incomplete_session_timestamps.append(
                degraded_session_timestamps
            ).tz_convert("UTC"),
            "session_marker": (
                ["incomplete"] * len(incomplete_session_timestamps)
                + ["degraded"] * len(degraded_session_timestamps)
            ),
        }
    )

    # Describe the normal boundaries and separate quality of both dates.
    session_schedule = pd.DataFrame(
        {
            "session_open": pd.to_datetime(
                [
                    "2026-09-11 09:30:00-04:00",
                    "2026-09-14 09:30:00-04:00",
                ],
                utc=True,
            ),
            "session_close": pd.to_datetime(
                [
                    "2026-09-11 16:00:00-04:00",
                    "2026-09-14 16:00:00-04:00",
                ],
                utc=True,
            ),
            "data_condition": ["available", "degraded"],
        }
    )

    # Apply timestamp completeness and Databento quality validation together.
    complete_session_data = filter_complete_sessions(
        candlestick_data,
        session_schedule,
        timestamp_column="ts_event",
        target_interval="5min",
    )

    # Reject the available but incomplete date and the complete degraded date.
    assert complete_session_data.empty


def test_filter_complete_sessions_rejects_misaligned_boundaries() -> None:
    # Create one valid timestamp so input validation reaches the schedule check.
    candlestick_data = pd.DataFrame(
        {
            "ts_event": pd.to_datetime(
                ["2026-09-15 13:30:00Z"],
                utc=True,
            )
        }
    )

    # Create a seven-minute session that cannot contain whole five-minute bars.
    session_schedule = pd.DataFrame(
        {
            "session_open": pd.to_datetime(
                ["2026-09-15 13:30:00Z"],
                utc=True,
            ),
            "session_close": pd.to_datetime(
                ["2026-09-15 13:37:00Z"],
                utc=True,
            ),
            "data_condition": ["available"],
        }
    )

    # Reject boundaries that would create a partial target candlestick.
    with pytest.raises(ValueError, match="integer multiple"):
        filter_complete_sessions(
            candlestick_data,
            session_schedule,
            timestamp_column="ts_event",
            target_interval="5min",
        )
