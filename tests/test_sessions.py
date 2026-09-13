import pandas as pd
import pytest

from pricesanity.data.sessions import filter_complete_sessions


def test_filter_complete_sessions_keeps_regular_and_half_sessions() -> None:
    """Verify filter complete sessions keeps regular and half sessions."""

    # Generate a complete earlier session whose close provides the first
    # retained session with a trustworthy normalization reference.
    reference_session_timestamps = pd.date_range(
        "2026-09-08 09:30:00",
        "2026-09-08 16:15:00",
        freq="5min",
        inclusive="left",
        tz="America/New_York",
    )

    # Generate the timestamps expected for one regular RTH session.
    regular_session_timestamps = pd.date_range(
        "2026-09-09 09:30:00",
        "2026-09-09 16:15:00",
        freq="5min",
        inclusive="left",
        tz="America/New_York",
    )

    # Generate the timestamps expected for one scheduled early-close session.
    half_session_timestamps = pd.date_range(
        "2026-09-10 09:30:00",
        "2026-09-10 13:15:00",
        freq="5min",
        inclusive="left",
        tz="America/New_York",
    )

    # Combine the reference, regular, and half-day sessions into one UTC
    # candlestick dataset.
    candlestick_data = pd.DataFrame(
        {
            "ts_event": reference_session_timestamps.append(regular_session_timestamps)
            .append(half_session_timestamps)
            .tz_convert("UTC"),
            "session_marker": (
                ["reference"] * len(reference_session_timestamps)
                + ["regular"] * len(regular_session_timestamps)
                + ["half"] * len(half_session_timestamps)
            ),
        }
    )

    # Describe all three sessions using boundaries derived from scheduled status
    # data.
    session_schedule = pd.DataFrame(
        {
            "session_open": pd.to_datetime(
                [
                    "2026-09-08 09:30:00-04:00",
                    "2026-09-09 09:30:00-04:00",
                    "2026-09-10 09:30:00-04:00",
                ],
                utc=True,
            ),
            "session_close": pd.to_datetime(
                [
                    "2026-09-08 16:15:00-04:00",
                    "2026-09-09 16:15:00-04:00",
                    "2026-09-10 13:15:00-04:00",
                ],
                utc=True,
            ),
            "data_condition": ["available", "available", "available"],
        }
    )

    # Validate every date and its preceding closing reference.
    complete_session_data = filter_complete_sessions(
        candlestick_data,
        session_schedule,
        timestamp_column="ts_event",
        target_interval="5min",
    )

    # Use the first date only as a reference, then keep the regular and
    # scheduled half-session because both have trustworthy predecessors.
    assert len(complete_session_data) == (
        len(regular_session_timestamps) + len(half_session_timestamps)
    )
    assert complete_session_data["session_marker"].unique().tolist() == [
        "regular",
        "half",
    ]


def test_filter_complete_sessions_discards_incomplete_and_degraded() -> None:
    """Verify filter complete sessions discards incomplete and degraded."""

    # Remove the final timestamp from an otherwise normal scheduled session.
    incomplete_session_timestamps = pd.date_range(
        "2026-09-11 09:30:00",
        "2026-09-11 16:15:00",
        freq="5min",
        inclusive="left",
        tz="America/New_York",
    )[:-1]

    # Generate a complete session whose date is marked degraded by Databento.
    degraded_session_timestamps = pd.date_range(
        "2026-09-14 09:30:00",
        "2026-09-14 16:15:00",
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
                    "2026-09-11 16:15:00-04:00",
                    "2026-09-14 16:15:00-04:00",
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
    """Verify filter complete sessions rejects misaligned boundaries."""

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


def test_filter_complete_sessions_requires_trustworthy_predecessor() -> None:
    """Verify filter complete sessions requires trustworthy predecessor."""

    # Give four complete dates different quality roles so the test can separate
    # a session's own trustworthiness from its eligibility as model input.
    session_dates = pd.to_datetime(["2026-09-08", "2026-09-09", "2026-09-10", "2026-09-11"])
    session_timestamps = [
        pd.date_range(
            date + pd.Timedelta(hours=9, minutes=30),
            date + pd.Timedelta(hours=16, minutes=15),
            freq="5min",
            inclusive="left",
            tz="America/New_York",
        )
        for date in session_dates
    ]

    # Preserve each date's role so the returned sessions can be identified
    # without inferring their dates from timestamps.
    session_markers = ("reference", "degraded", "recovery", "eligible")
    candlestick_data = pd.DataFrame(
        {
            "ts_event": session_timestamps[0]
            .append(session_timestamps[1])
            .append(session_timestamps[2])
            .append(session_timestamps[3])
            .tz_convert("UTC"),
            "session_marker": [
                marker
                for marker, timestamps in zip(
                    session_markers,
                    session_timestamps,
                    strict=True,
                )
                for _ in timestamps
            ],
        }
    )

    # Mark only the second date degraded while leaving all timestamp sequences
    # complete.
    session_schedule = pd.DataFrame(
        {
            "session_open": [timestamps[0] for timestamps in session_timestamps],
            "session_close": [
                timestamps[-1] + pd.Timedelta(minutes=5) for timestamps in session_timestamps
            ],
            "data_condition": [
                "available",
                "degraded",
                "available",
                "available",
            ],
        }
    )

    # Require each retained date to have its own trustworthy data and a
    # trustworthy close from the scheduled date immediately before it.
    complete_session_data = filter_complete_sessions(
        candlestick_data,
        session_schedule,
        timestamp_column="ts_event",
        target_interval="5min",
    )

    # The recovery date repairs the causal reference chain but cannot use the
    # degraded close itself; only the following date becomes eligible.
    assert complete_session_data["session_marker"].unique().tolist() == ["eligible"]
