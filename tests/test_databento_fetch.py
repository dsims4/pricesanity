import argparse
import json
from datetime import date
from pathlib import Path

import pytest

from pricesanity.data import databento_fetch
from pricesanity.data.databento_fetch import (
    DatabentoFetchRequest,
    build_download_argument_parser,
    build_estimate_argument_parser,
    build_raw_data_paths,
    create_raw_data_directory,
    download_raw_data,
    estimate_fetch_cost,
    parse_request_date,
)


class FakeDataStore:
    """Represent downloaded records without contacting Databento."""

    def __init__(self, csv_contents: str) -> None:
        """Store controlled CSV contents and received save arguments."""
        # The contents identify which schema was saved, while the arguments
        # prove the production code used immutable raw-file settings.
        self.csv_contents = csv_contents
        self.received_save_arguments: dict[str, object] = {}

    def to_csv(
        self,
        path: str | Path,
        *,
        pretty_px: bool = True,
        pretty_ts: bool = True,
        map_symbols: bool = True,
        mode: str = "w",
    ) -> None:
        """Record save settings and create the requested fake CSV."""
        # Keep every option so the test can verify readable values, symbol
        # mapping, and exclusive file creation.
        self.received_save_arguments = {
            "path": Path(path),
            "pretty_px": pretty_px,
            "pretty_ts": pretty_ts,
            "map_symbols": map_symbols,
            "mode": mode,
        }

        # Use the requested mode so the fake preserves the real store's refusal
        # to replace an existing file.
        with Path(path).open(mode=mode, encoding="utf-8") as csv_file:
            csv_file.write(self.csv_contents)


class FakeTimeseriesClient:
    """Return controlled data stores for Databento schema requests."""

    def __init__(self) -> None:
        """Create fake candlestick and status downloads."""
        # Separate stores and recorded requests allow each schema call and file
        # destination to be checked independently.
        self.received_requests: list[dict[str, object]] = []
        self.candlestick_store = FakeDataStore("candlesticks\n")
        self.status_store = FakeDataStore("status\n")

    def get_range(
        self,
        *,
        dataset: str,
        start: date,
        end: date,
        symbols: str,
        schema: str,
        stype_in: str,
    ) -> FakeDataStore:
        """Record one request and return its schema-specific data."""
        # Preserve the complete request so estimation and download parameters
        # can be compared without involving Databento.
        self.received_requests.append(
            {
                "dataset": dataset,
                "start": start,
                "end": end,
                "symbols": symbols,
                "schema": schema,
                "stype_in": stype_in,
            }
        )

        # Match each supported schema with the fake contents intended for its
        # raw output file.
        if schema == "ohlcv-1m":
            return self.candlestick_store
        return self.status_store


class FakeMetadataClient:
    """Record cost calls and return controlled estimates for one test."""

    def __init__(self) -> None:
        """Create an empty record of received Databento request parameters."""
        # Each call is retained so the test can prove that estimation used the
        # same request identity and date range for both required schemas.
        self.received_requests: list[dict[str, object]] = []

        # Condition calls are recorded separately because their ending date is
        # inclusive instead of the time-series API's exclusive ending date.
        self.received_condition_requests: list[dict[str, object]] = []

    def get_cost(self, **request_parameters: object) -> float:
        """Return a schema-specific cost without contacting Databento."""
        # Preserve a separate copy because later changes to a supplied mapping
        # should not alter the evidence collected by this fake client.
        self.received_requests.append(dict(request_parameters))

        # Different values make it possible to verify both component estimates
        # and their combined total rather than accidentally counting one twice.
        if request_parameters["schema"] == "ohlcv-1m":
            return 1.25
        return 0.10

    def get_dataset_condition(
        self,
        *,
        dataset: str,
        start_date: date,
        end_date: date,
    ) -> list[dict[str, str | None]]:
        """Return controlled daily condition metadata."""
        # Record the adjusted date range so its different ending convention can
        # be asserted directly by the download test.
        self.received_condition_requests.append(
            {
                "dataset": dataset,
                "start_date": start_date,
                "end_date": end_date,
            }
        )

        # One representative record is sufficient to verify JSON persistence.
        return [
            {
                "date": start_date.isoformat(),
                "condition": "available",
                "last_modified_date": None,
            }
        ]


class FakeHistoricalClient:
    """Provide the metadata attribute expected by the cost estimator."""

    def __init__(self) -> None:
        """Attach the controlled metadata client used by the test."""
        # Match the official client's nested metadata interface without adding
        # a network dependency to this unit test.
        self.metadata = FakeMetadataClient()
        self.timeseries = FakeTimeseriesClient()


def test_estimate_fetch_cost_uses_volume_continuous_contract() -> None:
    # Use two adjacent dates because Databento interprets the end as exclusive.
    request = DatabentoFetchRequest(
        start_date=date(2026, 9, 8),
        end_date=date(2026, 9, 9),
    )
    client = FakeHistoricalClient()

    # Estimate both required schemas without invoking a data-download method.
    estimate = estimate_fetch_cost(client, request)

    # Confirm the request follows the previous-day-volume continuous contract
    # selected for the model rather than the calendar-ranked contract.
    assert client.metadata.received_requests == [
        {
            "dataset": "GLBX.MDP3",
            "start": date(2026, 9, 8),
            "end": date(2026, 9, 9),
            "symbols": "ES.v.0",
            "stype_in": "continuous",
            "schema": "ohlcv-1m",
        },
        {
            "dataset": "GLBX.MDP3",
            "start": date(2026, 9, 8),
            "end": date(2026, 9, 9),
            "symbols": "ES.v.0",
            "stype_in": "continuous",
            "schema": "status",
        },
    ]

    # Confirm the component estimates remain inspectable and sum correctly.
    assert estimate.candlestick_cost_usd == 1.25
    assert estimate.status_cost_usd == 0.10
    assert estimate.total_cost_usd == pytest.approx(1.35)


def test_fetch_request_rejects_empty_date_range() -> None:
    # Use the same start and exclusive end to describe a request with no time.
    with pytest.raises(ValueError, match="end date must follow"):
        DatabentoFetchRequest(
            start_date=date(2026, 9, 8),
            end_date=date(2026, 9, 8),
        )


def test_parse_request_date_rejects_invalid_date() -> None:
    # Verify invalid calendar dates fail before reaching Databento.
    with pytest.raises(
        argparse.ArgumentTypeError,
        match="YYYY-MM-DD",
    ):
        parse_request_date("2026-02-30")


def test_estimate_argument_parser_converts_dates() -> None:
    # Parse the same text a user would provide through the terminal.
    parsed_arguments = build_estimate_argument_parser().parse_args(
        [
            "--start",
            "2026-09-08",
            "--end",
            "2026-09-10",
        ]
    )

    # Confirm the argument parsing produced date objects rather than raw strings.
    assert parsed_arguments.start == date(2026, 9, 8)
    assert parsed_arguments.end == date(2026, 9, 10)


def test_main_displays_cost_estimate(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # Replace authentication with a fake client so the command cannot
    # contact Databento or download market data during this test.
    fake_client = FakeHistoricalClient()
    monkeypatch.setattr(
        databento_fetch,
        "create_historical_client",
        lambda: fake_client,
    )

    # Run the command with the same arguments supplied through a terminal.
    exit_status = databento_fetch.main(
        [
            "--start",
            "2026-09-08",
            "--end",
            "2026-09-10",
        ]
    )

    # Capture printed output so the command's user-facing estimate can
    # be tested.
    terminal_output = capsys.readouterr().out

    # Confirm the command succeeded and displayed every relevant cost.
    assert exit_status == 0
    assert "Symbol: ES.v.0" in terminal_output
    assert (
        "Dates: 2026-09-08 (inclusive) through "
        "2026-09-10 (exclusive)"
    ) in terminal_output
    assert "Candlesticks: $1.250000" in terminal_output
    assert "Status: $0.100000" in terminal_output
    assert "Total: $1.350000" in terminal_output


def test_build_raw_data_paths_keeps_request_files_together(
    tmp_path: Path,
) -> None:
    # Define one request whose identity should appear in its raw directory.
    request = DatabentoFetchRequest(
        start_date=date(2026, 9, 8),
        end_date=date(2026, 9, 9),
    )

    # Calculate temporary paths without writing licensed CME data.
    raw_paths = build_raw_data_paths(tmp_path, request)

    # Confirm every file belongs to the same request-specific directory.
    expected_directory = (
        tmp_path / "ES-v-0_2026-09-08_2026-09-09"
    )
    assert raw_paths.candlestick_csv_path == (
        expected_directory / "candlesticks.csv"
    )
    assert raw_paths.status_csv_path == expected_directory / "status.csv"
    assert raw_paths.condition_json_path == (
        expected_directory / "condition.json"
    )

    # Path construction must not create anything on disk.
    assert not expected_directory.exists()


def test_create_raw_data_directory_refuses_existing_request(
    tmp_path: Path,
) -> None:
    # Build three raw paths that share one request-specific directory.
    request = DatabentoFetchRequest(
        start_date=date(2026, 9, 8),
        end_date=date(2026, 9, 9),
    )
    raw_paths = build_raw_data_paths(tmp_path, request)

    # Create the directory once and confirm no data files were created.
    created_directory = create_raw_data_directory(raw_paths)
    assert created_directory.exists()
    assert created_directory.is_dir()
    assert not raw_paths.candlestick_csv_path.exists()
    assert not raw_paths.status_csv_path.exists()
    assert not raw_paths.condition_json_path.exists()

    # Refuse a second attempt because it could replace an existing download.
    with pytest.raises(FileExistsError):
        create_raw_data_directory(raw_paths)


def test_download_raw_data_saves_all_source_files(tmp_path: Path) -> None:
    # Use two requested dates so the exclusive and inclusive ending conventions
    # produce visibly different condition arguments.
    request = DatabentoFetchRequest(
        start_date=date(2026, 9, 8),
        end_date=date(2026, 9, 10),
    )
    raw_paths = build_raw_data_paths(tmp_path, request)
    client = FakeHistoricalClient()

    # Run the entire file workflow through fake network interfaces.
    returned_paths = download_raw_data(client, request, raw_paths)

    # Confirm both time-series calls use the same request except for schema.
    assert client.timeseries.received_requests == [
        {
            "dataset": "GLBX.MDP3",
            "start": date(2026, 9, 8),
            "end": date(2026, 9, 10),
            "symbols": "ES.v.0",
            "schema": "ohlcv-1m",
            "stype_in": "continuous",
        },
        {
            "dataset": "GLBX.MDP3",
            "start": date(2026, 9, 8),
            "end": date(2026, 9, 10),
            "symbols": "ES.v.0",
            "schema": "status",
            "stype_in": "continuous",
        },
    ]

    # Confirm the inclusive metadata endpoint stops one day before the
    # exclusive time-series end.
    assert client.metadata.received_condition_requests == [
        {
            "dataset": "GLBX.MDP3",
            "start_date": date(2026, 9, 8),
            "end_date": date(2026, 9, 9),
        }
    ]

    # Confirm all raw artifacts were saved with their controlled contents.
    assert returned_paths is raw_paths
    assert raw_paths.candlestick_csv_path.read_text(
        encoding="utf-8"
    ) == "candlesticks\n"
    assert raw_paths.status_csv_path.read_text(
        encoding="utf-8"
    ) == "status\n"
    assert json.loads(
        raw_paths.condition_json_path.read_text(encoding="utf-8")
    ) == [
        {
            "date": "2026-09-08",
            "condition": "available",
            "last_modified_date": None,
        }
    ]

    # Confirm both CSVs used readable Databento formatting and exclusive file
    # creation rather than permitting replacement.
    assert client.timeseries.candlestick_store.received_save_arguments == {
        "path": raw_paths.candlestick_csv_path,
        "pretty_px": True,
        "pretty_ts": True,
        "map_symbols": True,
        "mode": "x",
    }
    assert client.timeseries.status_store.received_save_arguments == {
        "path": raw_paths.status_csv_path,
        "pretty_px": True,
        "pretty_ts": True,
        "map_symbols": True,
        "mode": "x",
    }


def test_download_argument_parser_requires_cost_ceiling() -> None:
    # Omit the approval amount so argument parsing must stop before a client or
    # request can be created.
    with pytest.raises(SystemExit):
        build_download_argument_parser().parse_args(
            [
                "--start",
                "2026-09-08",
                "--end",
                "2026-09-09",
            ]
        )


def test_download_main_stops_above_approved_cost(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # Use the fake client whose combined estimate is $1.35, then approve less
    # than that amount to exercise the command's spending boundary.
    fake_client = FakeHistoricalClient()
    monkeypatch.setattr(
        databento_fetch,
        "create_historical_client",
        lambda: fake_client,
    )

    # Confirm the command exits before calling either fake download endpoint.
    with pytest.raises(SystemExit):
        databento_fetch.download_main(
            [
                "--start",
                "2026-09-08",
                "--end",
                "2026-09-09",
                "--max-cost-usd",
                "1.00",
                "--raw-data-directory",
                str(tmp_path),
            ]
        )

    # The error should explain both prices without creating a raw directory.
    terminal_error = capsys.readouterr().err
    assert "estimate $1.350000" in terminal_error
    assert "approved maximum $1.000000" in terminal_error
    assert fake_client.timeseries.received_requests == []
    assert list(tmp_path.iterdir()) == []


def test_download_main_saves_below_approved_cost(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # Approve more than the fake estimate so the command can exercise the local
    # download workflow without contacting Databento.
    fake_client = FakeHistoricalClient()
    monkeypatch.setattr(
        databento_fetch,
        "create_historical_client",
        lambda: fake_client,
    )

    # Run the guarded command with an isolated raw-data parent.
    exit_status = databento_fetch.download_main(
        [
            "--start",
            "2026-09-08",
            "--end",
            "2026-09-09",
            "--max-cost-usd",
            "1.35",
            "--raw-data-directory",
            str(tmp_path),
        ]
    )

    # Confirm success, all three artifacts, and the reported cost ceiling.
    request_directory = tmp_path / "ES-v-0_2026-09-08_2026-09-09"
    terminal_output = capsys.readouterr().out
    assert exit_status == 0
    assert (request_directory / "candlesticks.csv").exists()
    assert (request_directory / "status.csv").exists()
    assert (request_directory / "condition.json").exists()
    assert "Estimated cost at download: $1.350000" in terminal_output
