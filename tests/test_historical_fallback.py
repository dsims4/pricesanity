"""Historical fallback sessions must remain conservative and causal."""

from dataclasses import replace

import pandas as pd

from pricesanity.config import load_config
from pricesanity.data.databento_ingest import iter_ohlc_csv
from pricesanity.data.pipeline import (
    prepare_candlestick_session_tables,
    prepare_csv_session_tables,
)


def historical_corpus():
    """Build pre-coverage cases and an authoritative post-coverage boundary."""

    config = load_config("configs/default.yaml")
    config = replace(config, session=replace(config.session, end_time="09:45"))
    specifications = [
        ("2014-06-09", "available", 15, False, 100.0),
        ("2014-06-10", "available", 15, False, 101.0),
        ("2014-06-11", "degraded", 15, False, 102.0),
        ("2014-06-12", "available", 15, False, 103.0),
        ("2014-06-13", "available", 15, False, 104.0),
        ("2014-06-16", "unavailable", 15, False, 105.0),
        ("2014-06-17", None, 15, False, 106.0),
        # Without a scheduled close, an early-close-shaped date must fail the
        # configured normal-session grid instead of receiving an invented close.
        ("2014-07-03", "available", 10, False, 107.0),
        ("2015-11-19", "available", 15, False, 110.0),
        # This is the first authoritative status-derived session and therefore
        # the exact date on which historical fallback stops.
        ("2015-11-20", "available", 15, True, 112.0),
        ("2015-11-23", "available", 15, False, 113.0),
        ("2015-11-24", "available", 15, True, 114.0),
        ("2015-11-25", "available", 10, True, 115.0),
    ]
    candle_frames = []
    status_rows = []
    condition_rows = []
    for day, condition, minutes, has_status, price in specifications:
        timestamps = pd.date_range(
            day + " 09:30",
            periods=minutes,
            freq="1min",
            tz="America/New_York",
        ).tz_convert("UTC")
        candle_frames.append(
            pd.DataFrame(
                {
                    "ts_event": timestamps,
                    "open": price,
                    "high": price,
                    "low": price,
                    "close": price,
                }
            )
        )
        if condition is not None:
            condition_rows.append({"date": day, "condition": condition})
        if has_status:
            close_time = "09:40" if minutes == 10 else "09:45"
            status_rows.extend(
                [
                    {
                        "ts_event": pd.Timestamp(
                            day + " 08:00", tz="America/New_York"
                        ).tz_convert("UTC"),
                        "reason": "scheduled",
                        "trading_event": "none",
                        "is_trading": "Y",
                    },
                    {
                        "ts_event": pd.Timestamp(
                            day + " " + close_time, tz="America/New_York"
                        ).tz_convert("UTC"),
                        "reason": "scheduled",
                        "trading_event": "none",
                        "is_trading": "N",
                    },
                ]
            )
    return (
        pd.concat(candle_frames, ignore_index=True),
        pd.DataFrame(status_rows),
        pd.DataFrame(condition_rows),
        config,
    )


def local_dates(table: pd.DataFrame) -> list[str]:
    """Return unique New York session dates from a prepared table."""

    return list(
        dict.fromkeys(
            table.ts_event.dt.tz_convert("America/New_York").dt.date.astype(str)
        )
    )


def test_pre_status_fallback_keeps_only_complete_available_normal_sessions():
    """Verify available full sessions use fallback without weakening quality."""

    candles, status, conditions, config = historical_corpus()
    result = prepare_candlestick_session_tables(
        candles, status, conditions, config=config
    )

    # The first available date is reference-only. Degraded, unavailable,
    # missing-quality, and incomplete early-close-shaped dates never appear.
    prepared_dates = local_dates(result.ohlc)
    assert "2014-06-09" not in prepared_dates
    assert "2014-06-10" in prepared_dates
    assert "2014-06-11" not in prepared_dates
    assert "2014-06-12" not in prepared_dates
    assert "2014-06-13" in prepared_dates
    assert "2014-06-16" not in prepared_dates
    assert "2014-06-17" not in prepared_dates
    assert "2014-07-03" not in prepared_dates


def test_fallback_stops_at_first_status_derived_session_and_keeps_early_close():
    """Verify authoritative coverage replaces fallback from its first date onward."""

    candles, status, conditions, config = historical_corpus()
    result = prepare_candlestick_session_tables(
        candles, status, conditions, config=config
    )
    prepared_dates = local_dates(result.ohlc)

    # November 19 receives the last fallback schedule and supplies the causal
    # close used by the first authoritative session on November 20.
    assert "2015-11-20" in prepared_dates
    first_boundary_candle = result.normalized.loc[
        result.normalized.ts_event
        == pd.Timestamp("2015-11-20 09:30", tz="America/New_York").tz_convert("UTC")
    ].iloc[0]
    assert first_boundary_candle.open_gap == (112.0 - 110.0) / 110.0

    # A missing post-coverage schedule breaks the chain instead of invoking
    # fallback. The next scheduled day restores it, and the official early
    # close after that remains eligible with two five-minute candles.
    assert "2015-11-23" not in prepared_dates
    assert "2015-11-24" not in prepared_dates
    assert prepared_dates.count("2015-11-25") == 1
    assert sum(
        result.ohlc.ts_event.dt.tz_convert("America/New_York")
        .dt.date.astype(str).eq("2015-11-25")
    ) == 2


def test_chunked_and_eager_historical_fallback_are_identical(tmp_path):
    """Verify CSV read boundaries do not alter fallback or normalization."""

    candles, status, conditions, config = historical_corpus()
    csv_path = tmp_path / "candlesticks.csv"
    candles.to_csv(csv_path, index=False)
    decoded = pd.concat(
        list(iter_ohlc_csv(csv_path, chunk_rows=100_000)), ignore_index=True
    )
    eager = prepare_candlestick_session_tables(
        decoded, status, conditions, config=config
    )
    chunked = prepare_csv_session_tables(
        csv_path,
        status,
        conditions,
        config=config,
        csv_chunk_rows=7,
    )

    pd.testing.assert_frame_equal(chunked.ohlc, eager.ohlc)
    pd.testing.assert_frame_equal(chunked.normalized, eager.normalized)
