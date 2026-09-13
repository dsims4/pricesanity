"""CSV read boundaries must not change candle geometry or session eligibility."""

from dataclasses import replace

import pandas as pd
import pytest

from pricesanity.config import load_config
from pricesanity.data.pipeline import (
    prepare_candlestick_session_tables,
    prepare_csv_session_tables,
)
from pricesanity.data.databento_ingest import iter_ohlc_csv


def corpus(case):
    """Build four sessions with controlled gaps, early closes, or degraded evidence."""

    config = load_config("configs/default.yaml")
    config = replace(config, session=replace(config.session, end_time="09:45"))
    frames, statuses = [], []
    dates = ["2026-09-08", "2026-09-09", "2026-09-10", "2026-09-11"]

    # Generate multiple sessions so chunk boundaries can fall both inside candles and between
    # trading dates.
    for day_index, day in enumerate(dates):
        timestamps = pd.date_range(day + "T13:28:00Z", periods=19, freq="1min")
        prices = pd.Series(
            [
                100.0 + day_index + candlestick_index * 0.1
                for candlestick_index in range(len(timestamps))
            ]
        )
        frame = pd.DataFrame(
            {
                "ts_event": timestamps,
                "open": prices,
                "high": prices + 1,
                "low": prices - 1,
                "close": prices + 0.25,
                "volume": "unused",
            }
        )

        # Remove one source minute in the middle session to break completeness without dropping
        # the whole date.
        if case == "incomplete" and day_index == 1:
            frame = frame.loc[frame.ts_event != pd.Timestamp(day + "T13:44:00Z")]

        # Truncated corpus edges must not be mistaken for complete first or last sessions.
        if case == "partial_edges":
            # Start inside the first aggregation bucket so it cannot provide a trustworthy
            # closing reference.
            if day_index == 0:
                frame = frame.loc[frame.ts_event >= pd.Timestamp(day + "T13:32:00Z")]

            # End inside the last aggregation bucket so the trailing incomplete session is
            # rejected.
            if day_index == 3:
                frame = frame.loc[frame.ts_event < pd.Timestamp(day + "T13:43:00Z")]

        frames.append(frame)
        closing = "13:40" if case == "early" and day_index == 1 else "13:45"
        statuses.extend(
            [
                {
                    "ts_event": day + "T13:00:00Z",
                    "reason": "scheduled",
                    "trading_event": "none",
                    "is_trading": "Y",
                },
                {
                    "ts_event": day + "T" + closing + ":00Z",
                    "reason": "scheduled",
                    "trading_event": "none",
                    "is_trading": "N",
                },
            ]
        )

    raw = pd.concat(frames, ignore_index=True)
    conditions = pd.DataFrame({"date": dates, "condition": ["available"] * 4})

    # Keep prices intact while changing only quality evidence to isolate the degraded-day trust
    # rule.
    if case == "degraded":
        conditions.loc[1, "condition"] = "degraded"

    return raw, pd.DataFrame(statuses), conditions, config


@pytest.mark.parametrize("chunk_rows", [1, 2, 4, 7, 19, 100000])
@pytest.mark.parametrize("case", ["normal", "degraded", "incomplete", "early", "partial_edges"])
def test_chunked_equals_eager_across_candles_and_sessions(tmp_path, chunk_rows, case):
    """Verify chunked equals eager across candles and sessions."""

    raw, status, conditions, config = corpus(case)
    path = tmp_path / "candles.csv"
    raw.to_csv(path, index=False)

    # Both paths receive CSV-decoded floats, avoiding an unrelated text-roundtrip difference.
    decoded = pd.concat(list(iter_ohlc_csv(path, chunk_rows=100000)), ignore_index=True)
    expected = prepare_candlestick_session_tables(decoded, status, conditions, config=config)
    actual = prepare_csv_session_tables(
        path, status, conditions, config=config, csv_chunk_rows=chunk_rows
    )
    pd.testing.assert_frame_equal(actual.ohlc, expected.ohlc)
    pd.testing.assert_frame_equal(actual.normalized, expected.normalized)

    assert actual.ohlc.ts_event.is_unique
    assert "volume" not in actual.ohlc and "volume" not in actual.normalized

    # The damaged day breaks the chain; the next day restores the reference for the fourth day.
    if case in ("degraded", "incomplete"):
        assert set(actual.ohlc.ts_event.dt.date.astype(str)) == {"2026-09-11"}

    # Four complete sessions yield three eligible sessions because the first is reference-only.
    if case == "normal":
        assert len(actual.ohlc) == 9


def test_reader_rejects_duplicate_at_read_boundary(tmp_path):
    """Verify reader rejects duplicate at read boundary."""

    path = tmp_path / "candles.csv"
    path.write_text(
        "ts_event,open,high,low,close\n2026-09-08T13:30:00Z,100,100,100,100\n" * 1
        + "2026-09-08T13:30:00Z,100,100,100,100\n"
    )

    # Force equal timestamps into different reads to test the ordering check that per-chunk
    # validation misses.
    with pytest.raises(ValueError, match="across chunks"):
        list(iter_ohlc_csv(path, chunk_rows=1))


def test_invalid_chunk_rows_rejected(tmp_path):
    """Verify invalid chunk rows rejected."""

    # Reject a zero-sized read before trying to open the deliberately absent source file.
    with pytest.raises(ValueError, match="positive"):
        list(iter_ohlc_csv(tmp_path / "absent.csv", chunk_rows=0))


def test_chunked_build_uses_staged_outputs_end_to_end(tmp_path):
    """Verify chunked build uses staged outputs end to end."""

    from pricesanity.data.build_dataset import build_dataset_from_exports
    from dataclasses import asdict
    import json
    import yaml

    # Preserve the early-close scenario through actual export decoding, including
    # the configuration that defines its expected candle interval.
    raw, status, conditions, config = corpus("early")

    # Write realistic source artifacts so the builder exercises its loader and
    # staging path instead of receiving already prepared in-memory tables.
    raw_path, status_path = tmp_path / "raw.csv", tmp_path / "status.csv"
    raw.to_csv(raw_path, index=False)
    status.to_csv(status_path, index=False)
    condition_path, config_path = tmp_path / "condition.json", tmp_path / "config.yaml"
    condition_path.write_text(json.dumps(conditions.to_dict("records")))
    config_path.write_text(yaml.safe_dump(asdict(config)))

    # Two-row reads deliberately split five-minute buckets, requiring the builder
    # to carry their unfinished source minutes across reads.
    output, interim = tmp_path / "normalized.parquet", tmp_path / "ohlc.parquet"
    build_dataset_from_exports(
        raw_path,
        status_path,
        condition_path,
        output,
        config_path=config_path,
        interim_output_path=interim,
        csv_chunk_rows=2,
    )

    # A single large read provides an independent read-boundary comparison for
    # both artifacts after their Parquet round trip.
    expected = prepare_csv_session_tables(
        raw_path, status, conditions, config=config, csv_chunk_rows=100000
    )
    pd.testing.assert_frame_equal(pd.read_parquet(output), expected.normalized)
    pd.testing.assert_frame_equal(pd.read_parquet(interim), expected.ohlc)

    # A second build must honor overwrite protection for both aligned Parquet outputs.
    with pytest.raises(FileExistsError):
        build_dataset_from_exports(
            raw_path,
            status_path,
            condition_path,
            output,
            config_path=config_path,
            interim_output_path=interim,
            csv_chunk_rows=2,
        )
