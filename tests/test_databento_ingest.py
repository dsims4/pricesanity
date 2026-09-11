from pathlib import Path

import pytest

from pricesanity.data.databento_ingest import (
    load_dataset_conditions_json,
    load_ohlc_csv,
    load_status_csv,
)


def test_load_ohlc_csv_loads_ohlc_and_converts_to_utc(
    tmp_path: Path,
) -> None:
    # Create a vendor-style CSV with naive New York times and unused volume.
    csv_path = tmp_path / "sample.csv"
    csv_path.write_text(
        "ts_event,open,high,low,close,volume\n"
        "2026-09-09 09:30:00,6500,6502,6498,6501,100\n"
        "2026-09-09 09:31:00,6501,6503,6499,6502,120\n",
        encoding="utf-8",
    )

    # Load the file while supplying the timezone omitted by the CSV.
    candlestick_data = load_ohlc_csv(
        csv_path,
        source_timezone="America/New_York",
    )

    # Verify the loader kept only the required columns and both rows.
    assert list(candlestick_data.columns) == [
        "ts_event",
        "open",
        "high",
        "low",
        "close",
    ]
    assert len(candlestick_data) == 2

    # Verify timestamps became UTC and numeric prices retained their values.
    assert candlestick_data.loc[0, "ts_event"].isoformat() == (
        "2026-09-09T13:30:00+00:00"
    )
    assert candlestick_data.loc[0, "open"] == 6500

    # Confirm that volume does not enter this price-only pipeline.
    assert "volume" not in candlestick_data.columns


def test_load_ohlc_csv_rejects_duplicate_timestamps(
    tmp_path: Path,
) -> None:
    # Create two candlesticks with the same timestamp.
    csv_path = tmp_path / "duplicate.csv"
    csv_path.write_text(
        "ts_event,open,high,low,close\n"
        "2026-09-09 09:30:00,6500,6502,6498,6501\n"
        "2026-09-09 09:30:00,6501,6503,6499,6502\n",
        encoding="utf-8",
    )

    # Verify duplicate timestamps stop ingestion.
    with pytest.raises(ValueError, match="duplicate timestamps"):
        load_ohlc_csv(
            csv_path,
            source_timezone="America/New_York",
        )


def test_load_ohlc_csv_rejects_missing_required_column(
    tmp_path: Path,
) -> None:
    # Create a CSV without the required close column.
    csv_path = tmp_path / "missing_close.csv"
    csv_path.write_text(
        "ts_event,open,high,low\n"
        "2026-09-09 09:30:00,6500,6502,6498\n",
        encoding="utf-8",
    )

    # Verify the error identifies the missing column.
    with pytest.raises(ValueError, match="close"):
        load_ohlc_csv(
            csv_path,
            source_timezone="America/New_York",
        )


def test_load_ohlc_csv_rejects_invalid_timestamp(
    tmp_path: Path,
) -> None:
    # Create a CSV containing timestamp text that "pandas" cannot parse.
    csv_path = tmp_path / "invalid_timestamp.csv"
    csv_path.write_text(
        "ts_event,open,high,low,close\n"
        "not-a-timestamp,6500,6502,6498,6501\n",
        encoding="utf-8",
    )

    # Verify an invalid timestamp stops ingestion.
    with pytest.raises(ValueError, match="invalid timestamps"):
        load_ohlc_csv(
            csv_path,
            source_timezone="America/New_York",
        )


def test_load_ohlc_csv_rejects_nonchronological_rows(
    tmp_path: Path,
) -> None:
    # Create valid candlesticks arranged in reverse chronological order.
    csv_path = tmp_path / "nonchronological.csv"
    csv_path.write_text(
        "ts_event,open,high,low,close\n"
        "2026-09-09 09:31:00,6501,6503,6499,6502\n"
        "2026-09-09 09:30:00,6500,6502,6498,6501\n",
        encoding="utf-8",
    )

    # Verify unordered source rows stop causal ingestion.
    with pytest.raises(ValueError, match="chronological order"):
        load_ohlc_csv(
            csv_path,
            source_timezone="America/New_York",
        )


def test_load_ohlc_csv_rejects_invalid_price(
    tmp_path: Path,
) -> None:
    # Create a CSV containing a non-numeric high price.
    csv_path = tmp_path / "invalid_price.csv"
    csv_path.write_text(
        "ts_event,open,high,low,close\n"
        "2026-09-09 09:30:00,6500,not-a-price,6498,6501\n",
        encoding="utf-8",
    )

    # Verify an invalid OHLC value stops ingestion.
    with pytest.raises(ValueError, match="invalid OHLC"):
        load_ohlc_csv(
            csv_path,
            source_timezone="America/New_York",
        )


def test_load_status_csv_keeps_transition_fields(tmp_path: Path) -> None:
    # Create a vendor-style status export with one field the pipeline does not
    # need after downloading the data.
    csv_path = tmp_path / "status.csv"
    csv_path.write_text(
        "ts_event,reason,trading_event,is_trading,symbol\n"
        "2026-09-09T13:30:00Z,scheduled,none,Y,ES.c.0\n"
        "2026-09-09T20:15:00Z,scheduled,none,N,ES.c.0\n",
        encoding="utf-8",
    )

    # Load only the evidence later used to identify scheduled RTH transitions.
    status_data = load_status_csv(csv_path)

    # Confirm unused vendor metadata was excluded without changing raw states.
    assert list(status_data.columns) == [
        "ts_event",
        "reason",
        "trading_event",
        "is_trading",
    ]
    assert status_data.loc[0, "is_trading"] == "Y"
    assert "symbol" not in status_data.columns


def test_load_status_csv_rejects_missing_required_field(
    tmp_path: Path,
) -> None:
    # Omit the reason because the remaining values cannot prove that a status
    # transition was scheduled.
    csv_path = tmp_path / "status_missing_reason.csv"
    csv_path.write_text(
        "ts_event,trading_event,is_trading\n"
        "2026-09-09T13:30:00Z,none,Y\n",
        encoding="utf-8",
    )

    # Verify the loader identifies the missing transition context.
    with pytest.raises(ValueError, match="reason"):
        load_status_csv(csv_path)


def test_load_dataset_conditions_json_keeps_daily_quality_fields(
    tmp_path: Path,
) -> None:
    # Reproduce Databento's list of daily condition records, including delivery
    # metadata that the model pipeline does not need.
    json_path = tmp_path / "condition.json"
    json_path.write_text(
        '[{"date":"2026-09-08","condition":"available",'
        '"last_modified_date":"2026-09-09"},'
        '{"date":"2026-09-09","condition":"degraded",'
        '"last_modified_date":"2026-09-10"}]',
        encoding="utf-8",
    )

    # Load the two values needed to attach quality to each session schedule.
    condition_data = load_dataset_conditions_json(json_path)

    # Confirm delivery metadata was excluded while both quality decisions were
    # preserved for downstream validation.
    assert list(condition_data.columns) == ["date", "condition"]
    assert condition_data.to_dict("records") == [
        {"date": "2026-09-08", "condition": "available"},
        {"date": "2026-09-09", "condition": "degraded"},
    ]


def test_load_dataset_conditions_json_rejects_invalid_structure(
    tmp_path: Path,
) -> None:
    # Use a single object instead of Databento's expected list of daily records.
    json_path = tmp_path / "invalid_condition.json"
    json_path.write_text(
        '{"date":"2026-09-09","condition":"available"}',
        encoding="utf-8",
    )

    # Verify ambiguous top-level metadata cannot enter schedule construction.
    with pytest.raises(ValueError, match="list of records"):
        load_dataset_conditions_json(json_path)


def test_load_dataset_conditions_json_rejects_missing_field(
    tmp_path: Path,
) -> None:
    # Omit condition so the date has no trustworthy quality decision.
    json_path = tmp_path / "condition_missing_quality.json"
    json_path.write_text(
        '[{"date":"2026-09-09"}]',
        encoding="utf-8",
    )

    # Verify the loader identifies the missing daily condition.
    with pytest.raises(ValueError, match="condition"):
        load_dataset_conditions_json(json_path)
