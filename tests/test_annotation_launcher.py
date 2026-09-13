"""Tests for loading prepared sessions into the annotation GUI."""

import pandas as pd
import pytest

from pricesanity.config import load_config
from pricesanity.gui.annotation_launcher import (
    build_argument_parser,
    load_annotation_sessions,
    parse_corpus_date_bounds,
)


@pytest.fixture
def config():
    """Load the project configuration used by prepared candlesticks."""

    return load_config("configs/default.yaml")


def test_load_annotation_sessions_creates_stable_identifiers(
    tmp_path,
    config,
) -> None:
    """One-session interim data is sorted and identified without a date flag."""

    candlestick_path = tmp_path / "ohlc.parquet"
    normalized_path = tmp_path / "normalized.parquet"

    # Write candles out of order to prove the GUI loader restores chronology
    # before creating identifiers and navigation positions.
    pd.DataFrame(
        {
            "ts_event": pd.to_datetime(["2026-09-09T13:35:00Z", "2026-09-09T13:30:00Z"]),
            "open": [101.0, 100.0],
            "high": [102.0, 101.0],
            "low": [100.0, 99.0],
            "close": [101.5, 100.5],
        }
    ).to_parquet(candlestick_path, index=False)

    # Store aligned opening gaps separately, matching the real pipeline output.
    pd.DataFrame(
        {
            "ts_event": pd.to_datetime(["2026-09-09T13:30:00Z", "2026-09-09T13:35:00Z"]),
            "open_gap": [0.001, -0.002],
        }
    ).to_parquet(normalized_path, index=False)

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
    assert candlestick_data["open_gap"].tolist() == [0.001, -0.002]


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
            "open_gap": [0.001, 0.002],
        }
    ).to_parquet(normalized_path, index=False)

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
    pd.DataFrame({"ts_event": [timestamp], "open_gap": [0.0]}).to_parquet(normalized_path)
    loaded = load_annotation_sessions(ohlc_path, normalized_path, config=config)
    candle = RawCandlestick(
        timestamp, config.data.instrument, 100.0, 101.0, 99.0, 100.0, interval="1min"
    )

    assert loaded.iloc[0].candlestick_id == candle.candlestick_id
