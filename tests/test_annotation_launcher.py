"""Tests for loading prepared sessions into the annotation GUI."""

import pandas as pd
import pytest

from pricesanity.config import load_config
from pricesanity.data.annotation_evidence import attach_annotation_evidence, EVIDENCE_ATTRIBUTE
from pricesanity.gui.annotation_launcher import (
    build_argument_parser,
    load_annotation_sessions,
    parse_corpus_date_bounds,
)


@pytest.fixture
def config():
    """Load the project configuration used by prepared candlesticks."""

    return load_config("configs/default.yaml")


def add_fixture_evidence(candlestick_path, normalized_path, config):
    """Give small loader fixtures explicit scheduled bounds and a real reference day."""

    candles = pd.read_parquet(candlestick_path)
    normalized = pd.read_parquet(normalized_path)

    # Match the source export's nanosecond representation before concatenating
    # reference history; mixed pandas resolutions can otherwise become objects.
    candles["ts_event"] = candles.ts_event.dt.as_unit("ns")
    normalized["ts_event"] = normalized.ts_event.dt.as_unit("ns")
    duration = pd.Timedelta(config.data.target_interval)
    dates = candles.ts_event.dt.tz_convert(config.data.session_timezone).dt.date
    first_date = pd.Timestamp(min(dates)) - pd.offsets.BDay(1)
    reference_times = pd.date_range(
        f"{first_date.date()} 09:30",
        f"{first_date.date()} 16:15",
        freq=duration,
        inclusive="left",
        tz=config.data.session_timezone,
    ).tz_convert("UTC").as_unit("ns")
    reference = pd.DataFrame(
        {"ts_event": reference_times, "open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0}
    )
    history = pd.concat([reference, candles], ignore_index=True)
    schedule_rows = []

    # Short fixtures represent scheduled early closes. The reference day is
    # complete and remains outside the annotation table, as in real preparation.
    for session_date, group in history.groupby(
        history.ts_event.dt.tz_convert(config.data.session_timezone).dt.date,
        sort=True,
    ):
        schedule_rows.append(
            {
                "session_date": session_date,
                "session_open": group.ts_event.iloc[0],
                "session_close": group.ts_event.iloc[-1] + duration,
                "data_condition": "available",
            }
        )

    attach_annotation_evidence(
        candles,
        normalized,
        history,
        pd.DataFrame(schedule_rows),
        config=config,
    )
    candles.to_parquet(candlestick_path, index=False)
    normalized.to_parquet(normalized_path, index=False)


def test_load_annotation_sessions_creates_stable_identifiers(
    tmp_path,
    config,
) -> None:
    """One-session interim data is identified without a date flag."""

    candlestick_path = tmp_path / "ohlc.parquet"
    normalized_path = tmp_path / "normalized.parquet"

    # Prepared artifacts retain their chronological pipeline order before IDs are assigned.
    pd.DataFrame(
        {
            "ts_event": pd.to_datetime(["2026-09-09T13:30:00Z", "2026-09-09T13:35:00Z"]),
            "open": [100.0, 101.0],
            "high": [101.0, 102.0],
            "low": [99.0, 100.0],
            "close": [100.5, 101.5],
        }
    ).to_parquet(candlestick_path, index=False)

    # Store aligned opening gaps separately, matching the real pipeline output.
    pd.DataFrame(
        {
            "ts_event": pd.to_datetime(["2026-09-09T13:30:00Z", "2026-09-09T13:35:00Z"]),
            "instrument": ["ES.v.0", "ES.v.0"],
            "open_gap": [0.0, (101.0 - 100.5) / 100.5],
        }
    ).to_parquet(normalized_path, index=False)

    add_fixture_evidence(candlestick_path, normalized_path, config)
    candlestick_data = load_annotation_sessions(
        candlestick_path,
        normalized_path,
        config=config,
    )

    assert candlestick_data["ts_event"].is_monotonic_increasing
    assert candlestick_data["candlestick_id"].tolist() == [
        "ES.v.0:5min:2026-09-09T13:30:00+00:00",
        "ES.v.0:5min:2026-09-09T13:35:00+00:00",
    ]
    assert candlestick_data["open_gap"].tolist() == [0.0, (101.0 - 100.5) / 100.5]


def test_load_annotation_sessions_keeps_multiple_sessions(
    tmp_path,
    config,
) -> None:
    """A multi-session file remains available to the GUI date controls."""

    candlestick_path = tmp_path / "ohlc.parquet"
    normalized_path = tmp_path / "normalized.parquet"
    pd.DataFrame(
        {
            "ts_event": pd.to_datetime(["2026-09-09T13:30:00Z", "2026-09-10T13:30:00Z"]),
            "open": [100.0, 101.0],
            "high": [101.0, 102.0],
            "low": [99.0, 100.0],
            "close": [100.5, 101.5],
        }
    ).to_parquet(candlestick_path, index=False)
    pd.DataFrame(
        {
            "ts_event": pd.to_datetime(["2026-09-09T13:30:00Z", "2026-09-10T13:30:00Z"]),
            "instrument": ["ES.v.0", "ES.v.0"],
            "open_gap": [0.0, (101.0 - 100.5) / 100.5],
        }
    ).to_parquet(normalized_path, index=False)

    add_fixture_evidence(candlestick_path, normalized_path, config)
    candlestick_data = load_annotation_sessions(
        candlestick_path,
        normalized_path,
        config=config,
    )

    assert len(candlestick_data) == 2
    assert [value.isoformat() for value in candlestick_data["session_date"]] == [
        "2026-09-09",
        "2026-09-10",
    ]


def test_load_annotation_sessions_requires_row_order_and_instrument(tmp_path, config) -> None:
    """Prepared representations cannot be reordered or assigned to another instrument."""

    timestamps = pd.to_datetime(["2026-09-09T13:30:00Z", "2026-09-09T13:35:00Z"])
    candlestick_path = tmp_path / "ohlc.parquet"
    normalized_path = tmp_path / "normalized.parquet"
    pd.DataFrame(
        {
            "ts_event": timestamps,
            "open": [100.0, 100.5],
            "high": [101.0, 101.5],
            "low": [99.0, 99.5],
            "close": [100.5, 101.0],
        }
    ).to_parquet(candlestick_path, index=False)

    normalized = pd.DataFrame(
        {
            "ts_event": timestamps[::-1],
            "instrument": ["ES.v.0", "ES.v.0"],
            "open_gap": [0.001, 0.002],
        }
    )
    normalized.to_parquet(normalized_path, index=False)

    # Matching timestamp sets are insufficient because the two artifacts promise row alignment.
    with pytest.raises(ValueError, match="row for row"):
        load_annotation_sessions(candlestick_path, normalized_path, config=config)

    normalized["ts_event"] = timestamps
    normalized["instrument"] = "NQ.v.0"
    normalized.to_parquet(normalized_path, index=False)

    # The configured instrument becomes part of every persistent annotation identifier.
    with pytest.raises(ValueError, match="configured instrument"):
        load_annotation_sessions(candlestick_path, normalized_path, config=config)

    spaced_timestamps = pd.to_datetime(
        ["2026-09-09T13:30:00Z", "2026-09-09T13:40:00Z"]
    )
    candlesticks = pd.read_parquet(candlestick_path)
    candlesticks["ts_event"] = spaced_timestamps
    candlesticks.to_parquet(candlestick_path, index=False)
    normalized["ts_event"] = spaced_timestamps
    normalized["instrument"] = "ES.v.0"
    normalized.to_parquet(normalized_path, index=False)

    # A different cadence would create stable IDs that falsely claim the configured interval.
    with pytest.raises(ValueError, match="configured target interval"):
        load_annotation_sessions(candlestick_path, normalized_path, config=config)


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("high", 99.0, "OHLC geometry"),
        ("low", float("inf"), "finite numbers"),
    ],
)
def test_load_annotation_sessions_rejects_invalid_chart_prices(
    tmp_path, config, column, value, message
) -> None:
    """The GUI refuses market geometry that could render a misleading candle."""

    timestamp = pd.Timestamp("2026-09-09T13:30:00Z")
    candlestick_path = tmp_path / "ohlc.parquet"
    normalized_path = tmp_path / "normalized.parquet"
    prices = {"open": [100.0], "high": [101.0], "low": [99.0], "close": [100.5]}
    prices[column] = [value]
    pd.DataFrame({"ts_event": [timestamp], **prices}).to_parquet(
        candlestick_path, index=False
    )
    pd.DataFrame(
        {"ts_event": [timestamp], "instrument": ["ES.v.0"], "open_gap": [0.0]}
    ).to_parquet(normalized_path, index=False)

    with pytest.raises(ValueError, match=message):
        load_annotation_sessions(candlestick_path, normalized_path, config=config)


def test_annotation_parser_defaults_to_ignored_database() -> None:
    """The terminal command keeps annotations under the ignored data tree."""

    parsed_arguments = build_argument_parser().parse_args(
        [
            "--candlesticks",
            "data/interim/example.parquet",
            "--normalized",
            "data/processed/example.parquet",
            "--config",
            "configs/default.yaml",
        ]
    )

    assert str(parsed_arguments.database) == ("data/annotations/pricesanity.sqlite3")


def test_load_annotation_sessions_rejects_misaligned_opening_gaps(
    tmp_path,
    config,
) -> None:
    """The GUI cannot pair an opening gap with a different candle."""

    candlestick_path = tmp_path / "ohlc.parquet"
    normalized_path = tmp_path / "normalized.parquet"
    pd.DataFrame(
        {
            "ts_event": pd.to_datetime(["2026-09-09T13:30:00Z"]),
            "open": [100.0],
            "high": [101.0],
            "low": [99.0],
            "close": [100.5],
        }
    ).to_parquet(candlestick_path, index=False)
    pd.DataFrame(
        {
            "ts_event": pd.to_datetime(["2026-09-09T13:35:00Z"]),
            "instrument": ["ES.v.0"],
            "open_gap": [0.001],
        }
    ).to_parquet(normalized_path, index=False)

    # Matching row counts alone cannot prove that raw candles and normalized features refer to
    # the same times.
    with pytest.raises(ValueError, match="timestamps must match exactly"):
        load_annotation_sessions(
            candlestick_path,
            normalized_path,
            config=config,
        )


def test_parse_corpus_date_bounds_converts_exclusive_end() -> None:
    """Filename dates become inclusive GUI calendar boundaries."""

    starting_date, ending_date = parse_corpus_date_bounds(
        "ES-v-0_2026-09-08_2026-09-10_ohlc.parquet"
    )

    assert starting_date.isoformat() == "2026-09-08"
    assert ending_date.isoformat() == "2026-09-09"


def test_load_annotation_sessions_reports_missing_projected_column(tmp_path, config):
    """Verify load annotation sessions reports missing projected column."""

    timestamps = pd.to_datetime(["2026-09-09T13:30:00Z"])
    candlesticks = tmp_path / "ohlc.parquet"
    normalized = tmp_path / "normalized.parquet"
    pd.DataFrame(
        {"ts_event": timestamps, "open": [100.0], "high": [101.0], "low": [99.0], "close": [100.0]}
    ).to_parquet(candlesticks)
    pd.DataFrame({"ts_event": timestamps, "body": [0.0]}).to_parquet(normalized)

    # The annotation view must not display an invalid opening gap as a usable normalized
    # feature.
    with pytest.raises(ValueError, match="open_gap"):
        load_annotation_sessions(candlesticks, normalized, config=config)


def test_loader_and_typed_candle_use_same_configured_interval(tmp_path, config):
    """Verify loader and typed candle use same configured interval."""

    from dataclasses import replace
    from pricesanity.data.schemas import RawCandlestick

    config = replace(config, data=replace(config.data, target_interval="60s"))
    timestamp = pd.Timestamp("2026-09-09T13:30:00Z")
    ohlc_path, normalized_path = tmp_path / "ohlc.parquet", tmp_path / "normalized.parquet"
    pd.DataFrame(
        {
            "ts_event": [timestamp],
            "open": [100.0],
            "high": [101.0],
            "low": [99.0],
            "close": [100.0],
        }
    ).to_parquet(ohlc_path)
    pd.DataFrame(
        {"ts_event": [timestamp], "instrument": ["ES.v.0"], "open_gap": [0.0]}
    ).to_parquet(normalized_path)
    add_fixture_evidence(ohlc_path, normalized_path, config)
    loaded = load_annotation_sessions(ohlc_path, normalized_path, config=config)
    candle = RawCandlestick(
        timestamp, config.data.instrument, 100.0, 101.0, 99.0, 100.0, interval="1min"
    )

    assert loaded.iloc[0].candlestick_id == candle.candlestick_id


def test_loader_accepts_nanosecond_candles_with_iso_session_metadata(tmp_path, config):
    """Parquet and ISO timestamps may differ in resolution while naming identical instants."""

    timestamp = pd.Timestamp("2026-09-09T13:30:00Z")
    candles = pd.DataFrame(
        {
            "ts_event": pd.DatetimeIndex([timestamp]).as_unit("ns"),
            "open": [100.0],
            "high": [100.0],
            "low": [100.0],
            "close": [100.0],
        }
    )
    normalized = pd.DataFrame(
        {"ts_event": candles.ts_event, "instrument": "ES.v.0", "open_gap": 0.0}
    )
    ohlc_path, normalized_path = tmp_path / "ohlc.parquet", tmp_path / "normalized.parquet"
    candles.to_parquet(ohlc_path, index=False)
    normalized.to_parquet(normalized_path, index=False)
    add_fixture_evidence(ohlc_path, normalized_path, config)

    # Reconstructing the boundaries from ISO text must not make a full session
    # appear incomplete merely because pandas chooses a coarser storage unit.
    loaded = load_annotation_sessions(ohlc_path, normalized_path, config=config)

    assert len(loaded) == 1


@pytest.mark.parametrize(
    "damage",
    ["missing_evidence", "opening_gap", "intraday_gap", "first_candle", "last_candle",
     "degraded_reference", "future_reference", "changed_reference"],
)
def test_loader_rejects_invalid_session_evidence(tmp_path, config, damage):
    """Reject stale or damaged artifacts before any candle becomes an annotation target."""

    candles = pd.DataFrame(
        {
            "ts_event": pd.date_range("2026-09-09T13:30:00Z", periods=3, freq="5min"),
            "open": [101.0, 102.0, 103.0],
            "high": [101.0, 102.0, 103.0],
            "low": [101.0, 102.0, 103.0],
            "close": [101.0, 102.0, 103.0],
        }
    )
    normalized = pd.DataFrame(
        {
            "ts_event": candles.ts_event,
            "instrument": config.data.instrument,
            "open_gap": [0.01, 1.0 / 101.0, 1.0 / 102.0],
        }
    )
    ohlc_path, normalized_path = tmp_path / "ohlc.parquet", tmp_path / "normalized.parquet"
    candles.to_parquet(ohlc_path, index=False)
    normalized.to_parquet(normalized_path, index=False)
    add_fixture_evidence(ohlc_path, normalized_path, config)
    candles = pd.read_parquet(ohlc_path)
    normalized = pd.read_parquet(normalized_path)

    # Damage only the selected contract so each case exercises an independent
    # guard rather than failing merely because timestamps no longer align.
    if damage == "missing_evidence":
        candles.attrs.clear()
        normalized.attrs.clear()
    elif damage in ("opening_gap", "intraday_gap"):
        position = 0 if damage == "opening_gap" else 1
        normalized.loc[position, "open_gap"] += 0.1
    elif damage in ("first_candle", "last_candle"):
        selected = slice(1, None) if damage == "first_candle" else slice(None, -1)
        candles = candles.iloc[selected].reset_index(drop=True)
        normalized = normalized.iloc[selected].reset_index(drop=True)
    else:
        # Keep both artifacts' metadata equal to verify the reference itself,
        # not just whether the two copies of the evidence agree.
        for frame in (candles, normalized):
            record = frame.attrs[EVIDENCE_ATTRIBUTE]["sessions"][0]
            if damage == "degraded_reference":
                record["previous_condition"] = "degraded"
            elif damage == "future_reference":
                record["reference_timestamp"] = "2026-09-10T20:10:00Z"
            else:
                record["reference_close"] = 120.0

    candles.to_parquet(ohlc_path, index=False)
    normalized.to_parquet(normalized_path, index=False)

    # The user must reprepare invalid data rather than annotate a plausible but
    # incorrect chart or a session whose reference is only assumed.
    with pytest.raises(ValueError, match="evidence|Opening gaps|not full|preceding session"):
        load_annotation_sessions(ohlc_path, normalized_path, config=config)
