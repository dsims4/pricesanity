"""Displayed sessions must retain a valid previous-session close across weekends."""

from dataclasses import replace

import pandas as pd
import pytest

from pricesanity.config import load_config
from pricesanity.data.pipeline import (
    prepare_candlestick_session_tables,
    prepare_csv_session_tables,
)
from pricesanity.gui.annotation_app import AnnotationWindow
from pricesanity.gui.annotation_launcher import load_annotation_sessions


@pytest.mark.parametrize("friday_state", ["complete", "missing", "partial", "degraded"])
@pytest.mark.parametrize("chunked", [False, True])
def test_first_displayed_session_requires_previous_close(
    tmp_path,
    qt_application,
    friday_state,
    chunked,
):
    """Reject stale Monday-to-Monday references while allowing complete Friday closes."""

    # A prior Monday cannot substitute for missing weekday candles. Friday is
    # the only valid reference for June 14; June 15 can instead use June 14.
    config = load_config("configs/default.yaml")
    config = replace(config, session=replace(config.session, end_time="09:45"))
    friday_minutes = {"complete": 15, "missing": 0, "partial": 14, "degraded": 15}
    specifications = [
        ("2010-06-07", 15, 100.0),
        ("2010-06-11", friday_minutes[friday_state], 105.0),
        ("2010-06-14", 15, 110.0),
        ("2010-06-15", 15, 120.0),
    ]
    frames = []

    # Distinct closes make using the wrong reference observable in the actual
    # normalized opening gap, rather than checking session dates alone.
    for day, minutes, price in specifications:
        frames.append(
            pd.DataFrame(
                {
                    "ts_event": pd.date_range(
                        day + " 09:30",
                        periods=minutes,
                        freq="1min",
                        tz="America/New_York",
                    ).tz_convert("UTC"),
                    "open": price,
                    "high": price,
                    "low": price,
                    "close": price,
                }
            )
        )

    candles = pd.concat(frames, ignore_index=True)

    # Include available weekend metadata too: it must not introduce a missing
    # Saturday or Sunday session between a valid Friday and Monday.
    conditions = pd.DataFrame(
        {
            "date": pd.date_range("2010-06-07", "2010-06-15").strftime("%Y-%m-%d"),
            "condition": "available",
        }
    )

    # A complete price grid still cannot supply a reference when its quality
    # condition is degraded.
    if friday_state == "degraded":
        conditions.loc[conditions.date.eq("2010-06-11"), "condition"] = "degraded"

    # Supply actual scheduled transitions for every evidenced weekday, even
    # when that instrument's candles are missing. Weekends remain unscheduled.
    status_rows = []

    # Synthetic pre-2015 scheduled evidence verifies that eligibility depends
    # on records rather than a hard-coded date cutoff.
    for day in pd.bdate_range("2010-06-07", "2010-06-15"):
        # Provide both states so each closing transition is independently visible.
        for clock, state in (("09:00", "Y"), ("09:45", "N")):
            status_rows.append(
                {
                    "ts_event": pd.Timestamp(
                        f"{day.date()} {clock}", tz="America/New_York"
                    ).tz_convert("UTC"),
                    "reason": "scheduled",
                    "trading_event": "none",
                    "is_trading": state,
                }
            )
    status = pd.DataFrame(status_rows)

    # Seven-row reads split five-minute buckets so the same eligibility rule
    # must survive both source loading paths.
    if chunked:
        csv_path = tmp_path / "source.csv"
        candles.to_csv(csv_path, index=False)
        prepared = prepare_csv_session_tables(
            csv_path,
            status,
            conditions,
            config=config,
            csv_chunk_rows=7,
        )
    else:
        prepared = prepare_candlestick_session_tables(
            candles,
            status,
            conditions,
            config=config,
        )

    expected_date = "2010-06-14" if friday_state == "complete" else "2010-06-15"
    expected_gap = (
        (110.0 - 105.0) / 105.0
        if friday_state == "complete"
        else (120.0 - 110.0) / 110.0
    )
    prepared_dates = prepared.ohlc.ts_event.dt.tz_convert("America/New_York").dt.date

    assert str(prepared_dates.min()) == expected_date
    assert prepared.normalized.iloc[0].open_gap == pytest.approx(expected_gap)
    assert prepared.ohlc.ts_event.equals(prepared.normalized.ts_event)

    # Exercise the actual saved-artifact loader and window, not just the
    # preparation result, so a reference-only date cannot reappear in navigation.
    ohlc_path = tmp_path / "ohlc.parquet"
    normalized_path = tmp_path / "normalized.parquet"
    prepared.ohlc.to_parquet(ohlc_path, index=False)
    prepared.normalized.to_parquet(normalized_path, index=False)
    displayed = load_annotation_sessions(ohlc_path, normalized_path, config=config)
    window = AnnotationWindow(displayed, tmp_path / "annotations.sqlite3")

    # Release the real SQLite connection even if a display assertion fails.
    try:
        assert str(window.available_session_dates[0]) == expected_date
        assert str(window.selected_session_dates[0]) == expected_date
        assert window.opening_gap_value.text() == f"{expected_gap:+.3%}"
    finally:
        window.close()
