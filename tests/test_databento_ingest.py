from pathlib import Path

import pytest

from pricesanity.data.databento_ingest import load_ohlc_csv


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
