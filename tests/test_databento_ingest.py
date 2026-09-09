from pathlib import Path

import pytest

from pricesanity.data.databento_ingest import load_ohlc_csv


def test_load_ohlc_csv_loads_ohlc_and_converts_to_utc(
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "sample.csv"
    csv_path.write_text(
        "ts_event,open,high,low,close,volume\n"
        "2026-09-09 09:30:00,6500,6502,6498,6501,100\n"
        "2026-09-09 09:31:00,6501,6503,6499,6502,120\n",
        encoding="utf-8",
    )

    frame = load_ohlc_csv(
        csv_path,
        source_timezone="America/New_York",
    )

    assert list(frame.columns) == [
        "ts_event",
        "open",
        "high",
        "low",
        "close",
    ]
    assert len(frame) == 2
    assert frame.loc[0, "ts_event"].isoformat() == (
        "2026-09-09T13:30:00+00:00"
    )
    assert frame.loc[0, "open"] == 6500
    assert "volume" not in frame.columns


def test_load_ohlc_csv_rejects_duplicate_timestamps(
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "duplicate.csv"
    csv_path.write_text(
        "ts_event,open,high,low,close\n"
        "2026-09-09 09:30:00,6500,6502,6498,6501\n"
        "2026-09-09 09:30:00,6501,6503,6499,6502\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate timestamps"):
        load_ohlc_csv(
            csv_path,
            source_timezone="America/New_York",
        )


def test_load_ohlc_csv_rejects_missing_required_column(
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "missing_close.csv"
    csv_path.write_text(
        "ts_event,open,high,low\n"
        "2026-09-09 09:30:00,6500,6502,6498\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="close"):
        load_ohlc_csv(
            csv_path,
            source_timezone="America/New_York",
        )


def test_load_ohlc_csv_rejects_invalid_timestamp(
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "invalid_timestamp.csv"
    csv_path.write_text(
        "ts_event,open,high,low,close\n"
        "not-a-timestamp,6500,6502,6498,6501\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="invalid timestamps"):
        load_ohlc_csv(
            csv_path,
            source_timezone="America/New_York",
        )


def test_load_ohlc_csv_rejects_nonchronological_rows(
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "nonchronological.csv"
    csv_path.write_text(
        "ts_event,open,high,low,close\n"
        "2026-09-09 09:31:00,6501,6503,6499,6502\n"
        "2026-09-09 09:30:00,6500,6502,6498,6501\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="chronological order"):
        load_ohlc_csv(
            csv_path,
            source_timezone="America/New_York",
        )


def test_load_ohlc_csv_rejects_invalid_price(
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "invalid_price.csv"
    csv_path.write_text(
        "ts_event,open,high,low,close\n"
        "2026-09-09 09:30:00,6500,not-a-price,6498,6501\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="invalid OHLC"):
        load_ohlc_csv(
            csv_path,
            source_timezone="America/New_York",
        )
