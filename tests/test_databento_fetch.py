import argparse
from datetime import date

import pytest

from pricesanity.data import databento_fetch
from pricesanity.data.databento_fetch import (
    DatabentoFetchRequest,
    build_estimate_argument_parser,
    estimate_fetch_cost,
    parse_request_date,
)


class FakeMetadataClient:
    """Record cost calls and return controlled estimates for one test."""

    def __init__(self) -> None:
        """Create an empty record of received Databento request parameters."""
        # Each call is retained so the test can prove that estimation used the
        # same request identity and date range for both required schemas.
        self.received_requests: list[dict[str, object]] = []

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


class FakeHistoricalClient:
    """Provide the metadata attribute expected by the cost estimator."""

    def __init__(self) -> None:
        """Attach the controlled metadata client used by the test."""
        # Match the official client's nested metadata interface without adding
        # a network dependency to this unit test.
        self.metadata = FakeMetadataClient()


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
