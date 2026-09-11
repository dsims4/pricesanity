"""Define Databento requests and estimate their cost before downloading."""

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from typing import Protocol

import databento as db
from dotenv import load_dotenv


class DatabentoMetadataClient(Protocol):
    """Describe the Databento metadata operation used for cost estimates."""

    def get_cost(
        self,
        *,
        dataset: str,
        start: date,
        end: date,
        symbols: str,
        schema: str,
        stype_in: str,
    ) -> float:
        """Return Databento's estimated cost for one historical request."""
        ...


class DatabentoHistoricalClient(Protocol):
    """Describe the portion of a historical client needed for estimation."""

    @property
    def metadata(self) -> DatabentoMetadataClient:
        """Return the client's historical metadata interface."""
        ...


def create_historical_client() -> db.Historical:
    """Create an authenticated Databento historical client.

    Returns:
        Historical client authenticated from the environment.
    """
    # Load local environment variables so authentication does not require
    # storing a private API key in source code.
    load_dotenv()

    # Databento automatically reads DATABENTO_API_KEY from the environment.
    return db.Historical()


def parse_request_date(date_text: str) -> date:
    """Parse one command-line date.

    Args:
        date_text: Date written in YYYY-MM-DD format.

    Returns:
        The parsed calendar date.

    Raises:
        ArgumentTypeError: If the date is invalid.
    """
    try:
        # Convert the command-line text into the date type expected
        # by the request object and Databento client.
        return date.fromisoformat(date_text)
    except ValueError as error:
        # Replace Python's parsing error with a short command-line explanation.
        raise argparse.ArgumentTypeError(
            "Dates must use YYYY-MM-DD format."
        ) from error


@dataclass(frozen=True)
class DatabentoFetchRequest:
    """One date range shared by the required Databento data requests."""

    start_date: date
    end_date: date
    dataset: str = "GLBX.MDP3"
    symbol: str = "ES.v.0"
    symbol_type: str = "continuous"
    candlestick_schema: str = "ohlcv-1m"
    status_schema: str = "status"

    def __post_init__(self) -> None:
        """Validate request settings before they reach Databento.

        Raises:
            ValueError: If the date range or request names are invalid.
        """
        # Databento treats the ending value as exclusive, so it must follow the
        # starting date for the request to contain any historical information.
        if self.start_date >= self.end_date:
            raise ValueError("The request end date must follow its start date.")

        # Empty API identifiers would broaden or invalidate the request instead
        # of describing the intended ES continuous-contract data.
        named_request_values = (
            self.dataset,
            self.symbol,
            self.symbol_type,
            self.candlestick_schema,
            self.status_schema,
        )
        if not all(value.strip() for value in named_request_values):
            raise ValueError("Databento request names cannot be empty.")


@dataclass(frozen=True)
class DatabentoCostEstimate:
    """Separate and combined estimated costs for the required market data."""

    candlestick_cost_usd: float
    status_cost_usd: float

    @property
    def total_cost_usd(self) -> float:
        """Return the estimated total cost in US dollars.

        Returns:
            Combined candlestick and status request cost.
        """
        # Keep both component estimates visible while providing the number a
        # user needs when deciding whether to approve the complete download.
        return self.candlestick_cost_usd + self.status_cost_usd


def estimate_fetch_cost(
    client: DatabentoHistoricalClient,
    request: DatabentoFetchRequest,
) -> DatabentoCostEstimate:
    """Estimate required Databento request costs without downloading data.

    Args:
        client: Authenticated Databento historical client.
        request: Dataset, symbol, schemas, and exclusive date range.

    Returns:
        Separate candlestick and status costs with a combined total.
    """
    # These parameters must remain identical between estimation and downloading
    # so the displayed price describes the data that will later be requested.
    shared_request_parameters = {
        "dataset": request.dataset,
        "start": request.start_date,
        "end": request.end_date,
        "symbols": request.symbol,
        "stype_in": request.symbol_type,
    }

    # Estimate one-minute candlesticks separately because they make up the main
    # price dataset and usually account for most of the request cost.
    candlestick_cost_usd = client.metadata.get_cost(
        **shared_request_parameters,
        schema=request.candlestick_schema,
    )

    # Estimate status records separately because they provide session boundaries
    # but use a distinct Databento schema and may have a different price.
    status_cost_usd = client.metadata.get_cost(
        **shared_request_parameters,
        schema=request.status_schema,
    )

    # Preserve both values instead of returning only a total so an unexpected
    # status cost remains visible before the user approves a download.
    return DatabentoCostEstimate(
        candlestick_cost_usd=float(candlestick_cost_usd),
        status_cost_usd=float(status_cost_usd),
    )


def build_estimate_argument_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for cost estimation.

    Returns:
        Parser accepting the requested start and exclusive end dates.
    """
    # Keep the terminal interface separate from its execution so tests can
    # inspect arguments without authenticating or contacting Databento.
    argument_parser = argparse.ArgumentParser(
        description="Estimate the cost of the required Databento data."
    )

    # Convert date text during argument parsing so invalid dates cannot
    # reach request construction or the Databento API.
    argument_parser.add_argument(
        "--start",
        required=True,
        type=parse_request_date,
        help="First requested date in YYYY-MM-DD format.",
    )
    argument_parser.add_argument(
        "--end",
        required=True,
        type=parse_request_date,
        help="Exclusive ending date in YYYY-MM-DD format.",
    )

    # Return the completed parser for both terminal execution and unit tests.
    return argument_parser


def main(arguments: Sequence[str] | None = None) -> int:
    """Estimate Databento costs from command-line arguments.

    Args:
        arguments: Optional arguments used instead of terminal input.

    Returns:
        Zero after the estimates are displayed.
    """
    # Parse and validate both dates before authenticating with Databento.
    parsed_arguments = build_estimate_argument_parser().parse_args(arguments)

    # Combine the validated dates with the project's fixed ES request settings.
    request = DatabentoFetchRequest(
        start_date=parsed_arguments.start,
        end_date=parsed_arguments.end,
    )

    # Authenticate only after local argument validation succeeds.
    client = create_historical_client()

    # Request metadata estimates without downloading either time-series schema.
    estimate = estimate_fetch_cost(client, request)

    # Display the request identity so the user can verify what was estimated.
    print(f"Symbol: {request.symbol}")
    print(
        f"Dates: {request.start_date} (inclusive) through "
        f"{request.end_date} (exclusive)"
    )
    print(f"Candlesticks: ${estimate.candlestick_cost_usd:.6f}")
    print(f"Status: ${estimate.status_cost_usd:.6f}")
    print(f"Total: ${estimate.total_cost_usd:.6f}")

    # Return a successful shell status after every estimate is displayed.
    return 0


if __name__ == "__main__":
    # Convert the function's return value into the process exit status.
    raise SystemExit(main())