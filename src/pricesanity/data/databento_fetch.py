"""Define Databento requests and estimate their cost before downloading."""

import argparse
from encodings.punycode import T
import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
import re
from typing import Literal, Protocol

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


class DatabentoDataStore(Protocol):
    """Describe a downloaded Databento result that can be saved as CSV."""

    def to_csv(
        self,
        path: str | Path,
        *,
        pretty_px: bool = True,
        pretty_ts: bool = True,
        map_symbols: bool = True,
        mode: Literal["w", "x"] = "w",
    ) -> None:
        """Write downloaded records to a CSV file."""
        ...


class DatabentoTimeseriesClient(Protocol):
    """Describe the Databento operation used to download one schema."""

    def get_range(
        self,
        *,
        dataset: str,
        start: date,
        end: date,
        symbols: str,
        schema: str,
        stype_in: str,
    ) -> DatabentoDataStore:
        """Download one historical time-series range."""
        ...


class DatabentoConditionClient(Protocol):
    """Describe the metadata operation used to retrieve daily conditions."""

    def get_dataset_condition(
        self,
        *,
        dataset: str,
        start_date: date,
        end_date: date,
    ) -> list[dict[str, str | None]]:
        """Return Databento's condition for each requested date."""
        ...


class DatabentoDownloadClient(Protocol):
    """Describe the client interfaces required for downloading raw data."""

    @property
    def timeseries(self) -> DatabentoTimeseriesClient:
        """Return the historical time-series interface."""
        ...

    @property
    def metadata(self) -> DatabentoConditionClient:
        """Return the dataset-condition interface."""
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


@dataclass(frozen=True)
class DatabentoRawDataPaths:
    """Paths for one immutable Databento download."""

    candlestick_csv_path: Path
    status_csv_path: Path
    condition_json_path: Path


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


def build_download_argument_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for guarded data downloads.

    Returns:
        Parser accepting dates, a cost ceiling, and a raw-data directory.
    """
    # Keep downloading on a separate command so estimating cost can never begin
    # a paid time-series request by accident.
    argument_parser = argparse.ArgumentParser(
        description="Download immutable raw data from Databento."
    )

    # Apply the same date conversion used by estimation so both commands build
    # identical request ranges.
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

    # Require the user to approve a concrete dollar ceiling so an estimate that
    # has increased since review cannot proceed silently.
    argument_parser.add_argument(
        "--max-cost-usd",
        required=True,
        type=float,
        help="Highest approved combined cost estimate in US dollars.",
    )

    # Default licensed files to the ignored raw-data tree while allowing tests
    # and deliberate external storage to select another parent directory.
    argument_parser.add_argument(
        "--raw-data-directory",
        type=Path,
        default=Path("data/raw"),
        help="Ignored parent directory for raw files (default: data/raw).",
    )

    # Return the parser without authenticating or making any API request.
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


def download_main(arguments: Sequence[str] | None = None) -> int:
    """Download raw Databento data within an approved cost ceiling.

    Args:
        arguments: Optional arguments used instead of terminal input.

    Returns:
        Zero after every raw source file is saved successfully.
    """
    # Parse the approval and request details before authenticating or checking
    # current Databento prices.
    argument_parser = build_download_argument_parser()
    parsed_arguments = argument_parser.parse_args(arguments)

    # A negative ceiling cannot represent meaningful spending authorization.
    if parsed_arguments.max_cost_usd < 0:
        argument_parser.error("--max-cost-usd cannot be negative.")

    # Rebuild the same fixed ES request used by the standalone estimator.
    request = DatabentoFetchRequest(
        start_date=parsed_arguments.start,
        end_date=parsed_arguments.end,
    )

    # Use one authenticated client so the estimate and immediate download share
    # the same account entitlements and pricing plan.
    client = create_historical_client()

    # Re-estimate immediately before downloading because pricing or account
    # entitlements may have changed since the earlier review.
    estimate = estimate_fetch_cost(client, request)

    # Stop before any time-series call when the current estimate exceeds the
    # exact amount authorized on the command line.
    if (
        estimate.total_cost_usd
        > parsed_arguments.max_cost_usd
    ):
        argument_parser.error(
            f"Current estimate ${estimate.total_cost_usd:.6f} exceeds "
            f"approved maximum "
            f"${parsed_arguments.max_cost_usd:.6f}."
        )

    # Calculate unique destinations beneath the ignored raw-data tree before
    # the downloader reserves their shared directory.
    raw_data_paths = build_raw_data_paths(
        parsed_arguments.raw_data_directory,
        request,
    )

    # Make the two approved time-series requests and save their supporting
    # condition metadata as one immutable raw download.
    saved_raw_data_paths = download_raw_data(
        client,
        request,
        raw_data_paths,
    )

    # Report each artifact separately so it can be inspected or passed into the
    # preparation command without searching the raw directory.
    print(f"Candlesticks: {saved_raw_data_paths.candlestick_csv_path}")
    print(f"Status: {saved_raw_data_paths.status_csv_path}")
    print(f"Conditions: {saved_raw_data_paths.condition_json_path}")
    print(f"Estimated cost at download: ${estimate.total_cost_usd:.6f}.")

    # A zero status tells shells and later automation that all files were saved.
    return 0


def build_raw_data_paths(
    raw_data_directory: str | Path,
    request: DatabentoFetchRequest,
) -> DatabentoRawDataPaths:
    """Build raw output paths for one Databento request.

    Args:
        raw_data_directory: Parent directory for raw market data.
        request: Request used to identify the downloaded data.

    Returns:
        Paths for the candlestick, status, and condition files.
    """
    # Remove symbol punctuation so the directory has a simple name.
    safe_symbol = request.symbol.replace(".", "-")

    # Use the request identity to prevent separate date ranges from
    # sharing filenames or silently replacing one another.
    request_directory = Path(raw_data_directory) / (
        f"{safe_symbol}_{request.start_date}_{request.end_date}"
    )

    # Keep all source files together because they describe the same request.
    return DatabentoRawDataPaths(
        candlestick_csv_path=request_directory / "candlesticks.csv",
        status_csv_path=request_directory / "status.csv",
        condition_json_path=request_directory / "condition.json",
    )


def create_raw_data_directory(
    raw_data_paths: DatabentoRawDataPaths,
) -> Path:
    """Create one new directory for a Databento download.

    Args:
        raw_data_paths: Related raw-file paths for one request.

    Returns:
        The newly created request directory.

    Raises:
        FileExistsError: If the request directory already exists.
        ValueError: If the raw files do not share one directory.
    """
    # Collect each parent to confirm that all files belong to one protected
    # request directory before anything is created.
    raw_file_directories = {
        raw_data_paths.candlestick_csv_path.parent,
        raw_data_paths.status_csv_path.parent,
        raw_data_paths.condition_json_path.parent,
    }

    # Files in separate directories could be partially overwritten or
    # confused with artifacts from another request.
    if len(raw_file_directories) != 1:
        raise ValueError("Raw data files must share one directory.")

    # Remove and return the only value after confirming the set has
    # one member.
    request_directory = raw_file_directories.pop()

    # Refuse an existing directory so immutable CME data cannot be
    # replaced by rerunning the same request.
    request_directory.mkdir(parents=True, exist_ok=False)

    return request_directory


def download_raw_data(
    client: DatabentoDownloadClient,
    request: DatabentoFetchRequest,
    raw_data_paths: DatabentoRawDataPaths,
) -> DatabentoRawDataPaths:
    """Download and save one immutable set of Databento source files.

    Args:
        client: Authenticated Databento historical client.
        request: Dataset, symbol, schemas, and requested date range.
        raw_data_paths: Destinations for the three raw source files.

    Returns:
        The paths containing the downloaded source data.

    Raises:
        FileExistsError: If the request directory or a raw file already exists.
    """
    # Reserve a new request directory before making paid calls so an
    # existing raw download can never be silently replaced.
    create_raw_data_directory(raw_data_paths)

    # Both time-series downloads must use the same symbol and date
    # range that were shown by the cost estimator.
    shared_request_parameters = {
        "dataset": request.dataset,
        "start": request.start_date,
        "end": request.end_date,
        "symbols": request.symbol,
        "stype_in": request.symbol_type,
    }

    # Download the one-minute candles that will later be filtered, resampled,
    # and normalized for model input.
    candlestick_store = client.timeseries.get_range(
        **shared_request_parameters,
        schema=request.candlestick_schema,
    )

    # Preserve readable timestamps, prices, and contract mappings in the
    # raw artifact while refusing to replace an existing file.
    candlestick_store.to_csv(
        raw_data_paths.candlestick_csv_path,
        pretty_px=True,
        pretty_ts=True,
        map_symbols=True,
        mode="x",
    )

    # Download session-status records separately because their scheduled
    # state changes establish regular and shortened session boundaries.
    status_store = client.timeseries.get_range(
        **shared_request_parameters,
        schema=request.status_schema,
    )

    # Keep the complete raw status export for auditing while later
    # ingestion selects only the fields required by schedule construction.
    status_store.to_csv(
        raw_data_paths.status_csv_path,
        pretty_px=True,
        pretty_ts=True,
        map_symbols=True,
        mode="x",
    )

    # Databento's time-series end is exclusive, while the condition endpoint's
    # ending date is inclusive, so subtract one day to describe the same range.
    condition_end_date = request.end_date - timedelta(days=1)

    # Retrieve the daily quality evidence used to reject degraded, pending, or
    # missing sessions without making another time-series download.
    condition_records = client.metadata.get_dataset_condition(
        dataset=request.dataset,
        start_date=request.start_date,
        end_date=condition_end_date,
    )

    # Save the original metadata records as readable JSON without changing
    # their dates, conditions, or last-modified values.
    with raw_data_paths.condition_json_path.open(
        mode="x",
        encoding="utf-8",
    ) as condition_file:
        json.dump(condition_records, condition_file, indent=2)
        condition_file.write("\n")

    # Return the destinations so the caller can display or pass them
    # directly into the preparation command.
    return raw_data_paths


if __name__ == "__main__":
    # Convert the function's return value into the process exit status.
    raise SystemExit(main())
