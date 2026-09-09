import pandas as pd

from pricesanity.data.clean import filter_regular_trading_hours


def test_filter_regular_trading_hours_respects_boundaries() -> None:
    frame = pd.DataFrame(
        {
            "ts_event": pd.to_datetime(
                [
                    "2026-09-09T13:29:00Z",  # 09:29 New York
                    "2026-09-09T13:30:00Z",  # 09:30 New York
                    "2026-09-09T19:59:00Z",  # 15:59 New York
                    "2026-09-09T20:00:00Z",  # 16:00 New York
                ],
                utc=True,
            ),
            "marker": ["before", "open", "last", "close"],
        }
    )

    result = filter_regular_trading_hours(
        frame,
        timestamp_column="ts_event",
        market_timezone="America/New_York",
        session_start="09:30",
        session_end="16:00",
        weekdays=(0, 1, 2, 3, 4),
    )

    assert result["marker"].tolist() == ["open", "last"]
    