import pandas as pd

from pricesanity.data.clean import filter_regular_trading_hours


def test_filter_regular_trading_hours_respects_boundaries() -> None:
    """Verify filter regular trading hours respects boundaries."""

    # Build rows immediately before, at, and after the RTH boundaries.
    candlestick_data = pd.DataFrame(
        {
            "ts_event": pd.to_datetime(
                [
                    "2026-09-09T13:29:00Z",  # 09:29 New York
                    "2026-09-09T13:30:00Z",  # 09:30 New York
                    "2026-09-09T20:14:00Z",  # 16:14 New York
                    "2026-09-09T20:15:00Z",  # 16:15 New York
                ],
                utc=True,
            ),
            "marker": ["before", "open", "last", "close"],
        }
    )

    # Filter the UTC rows using their corresponding New York session times.
    filtered_candlestick_data = filter_regular_trading_hours(
        candlestick_data,
        timestamp_column="ts_event",
        session_timezone="America/New_York",
        session_start_time="09:30",
        session_end_time="16:15",
        trading_weekdays=(0, 1, 2, 3, 4),
    )

    # Keep the opening minute and final in-session minute only.
    assert filtered_candlestick_data["marker"].tolist() == ["open", "last"]
