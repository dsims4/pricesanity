"""Available historical prices still require scheduled session evidence."""

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
        # Short prices cannot establish an early close without a scheduled
        # boundary; both partial and complete unscheduled dates remain unusable.
        ("2014-07-03", "available", 10, False, 107.0),
        ("2015-11-19", "available", 15, False, 110.0),
        # This is the first authoritative status-derived session and therefore
        # the first date that can supply a trustworthy closing reference.
        ("2015-11-20", "available", 15, True, 112.0),
        ("2015-11-23", "available", 15, False, 113.0),
        ("2015-11-24", "available", 15, True, 114.0),
        ("2015-11-25", "available", 10, True, 115.0),
    ]
    candle_frames = []
    status_rows = []
    condition_rows = []

    # Vary prices, quality, and scheduled evidence independently so availability
    # alone cannot accidentally qualify the pre-coverage examples.
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
        # Omitting quality evidence must remain distinct from an available day.
        if condition is not None:
            condition_rows.append({"date": day, "condition": condition})

        # Only these explicit transitions may establish session boundaries.
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


def test_available_prices_without_scheduled_status_are_not_eligible():
    """Dataset availability alone cannot establish a session or its previous close."""

    candles, status, conditions, config = historical_corpus()
    result = prepare_candlestick_session_tables(candles, status, conditions, config=config)

    # November 20 is the first complete scheduled session and is reference-only.
    # Missing November 23 status breaks the chain; November 24 restores it.
    assert local_dates(result.ohlc) == ["2015-11-25"]
    assert len(result.ohlc) == 2
    assert result.normalized.iloc[0].open_gap == (115.0 - 114.0) / 114.0


def test_no_scheduled_status_produces_no_annotation_targets():
    """Complete available candles remain unusable when no scheduled evidence exists."""

    candles, status, conditions, config = historical_corpus()
    result = prepare_candlestick_session_tables(
        candles, status.iloc[:0], conditions, config=config
    )

    assert result.ohlc.empty
    assert result.normalized.empty


def test_chunked_and_eager_status_eligibility_are_identical(tmp_path):
    """Verify CSV read boundaries do not alter scheduled eligibility or normalization."""

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
