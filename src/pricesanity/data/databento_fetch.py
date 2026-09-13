"""Define Databento requests and estimate their cost before downloading."""

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from math import isfinite
from pathlib import Path
from typing import Literal, Protocol

import databento as db
from dotenv import load_dotenv
from pricesanity.data.download_config import (
    DEFAULT_CANDLE_CHUNK_YEARS,
    DEFAULT_STATUS_CHUNK_MONTHS,
)


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
        """Return Databento's estimated cost for one historical request.

        Args:
            dataset: Databento dataset identifier.
            start: Inclusive beginning of the requested date range.
            end: Exclusive end of the requested date range.
            symbols: Instrument symbol requested from Databento.
            schema: Databento record schema to retrieve.
            stype_in: Symbology used to interpret the requested instrument.

        Returns:
            Estimated price of the historical request in US dollars.
        """

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
        """Write downloaded records to a CSV file.

        Args:
            path: File to read or publish.
            pretty_px: Whether CSV prices use readable decimal values.
            pretty_ts: Whether CSV timestamps use readable date-time strings.
            map_symbols: Whether CSV records include their resolved symbols.
            mode: File creation mode; exclusive creation protects existing files.
        """

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
        """Download one historical time-series range.

        Args:
            dataset: Databento dataset identifier.
            start: Inclusive beginning of the requested date range.
            end: Exclusive end of the requested date range.
            symbols: Instrument symbol requested from Databento.
            schema: Databento record schema to retrieve.
            stype_in: Symbology used to interpret the requested instrument.

        Returns:
            Completed SDK data store ready for local serialization.
        """

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
        """Return Databento's condition for each requested date.

        Args:
            dataset: Databento dataset identifier.
            start_date: Inclusive first date requested from the metadata endpoint.
            end_date: Inclusive final date requested from the metadata endpoint.

        Returns:
            Daily condition records covering the requested inclusive range.
        """

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

    # Parse the date at the CLI boundary so invalid input never reaches a vendor request.
    try:
        # Convert the command-line text into the date type expected
        # by the request object and Databento client.
        return date.fromisoformat(date_text)

    # Let argparse display malformed dates using its normal command-line error handling.
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

        # Empty vendor identifiers cannot define a reproducible historical request.
        if not all(value.strip() for value in named_request_values):
            raise ValueError("Databento request names cannot be empty.")


@dataclass(frozen=True)
class DatabentoCostEstimate:
    """Separate and combined estimated costs for the required market data."""

    candlestick_cost_usd: float
    status_cost_usd: float

    def __post_init__(self) -> None:
        """Reject unusable estimates before the spending comparison."""

        # Validate both paid components and their total before exposing an estimate for
        # approval.
        costs = (self.candlestick_cost_usd, self.status_cost_usd)

        # Negative, nonfinite, or overflowing estimates cannot support a safe approval decision.
        if not all(isfinite(cost) and cost >= 0 for cost in costs) or not isfinite(sum(costs)):
            raise ValueError("Cost estimates must be finite and nonnegative.")

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
        help="Maximum additional estimated cost for this invocation, including retries.",
    )

    # Default licensed files to the ignored raw-data tree while allowing tests
    # and deliberate external storage to select another parent directory.
    argument_parser.add_argument(
        "--raw-data-directory",
        type=Path,
        default=Path("data/raw"),
        help="Ignored parent directory for raw files (default: data/raw).",
    )

    argument_parser.add_argument(
        "--candlestick-chunk-years",
        type=int,
        default=DEFAULT_CANDLE_CHUNK_YEARS,
        help=f"Calendar years per OHLC chunk (default: {DEFAULT_CANDLE_CHUNK_YEARS}).",
    )
    status_chunking = argument_parser.add_mutually_exclusive_group()
    status_chunking.add_argument(
        "--status-chunk-years",
        type=int,
        help="Calendar years per status chunk; overrides the quarterly default.",
    )
    status_chunking.add_argument(
        "--status-chunk-months",
        type=int,
        choices=(1, 2, 3, 4, 6, 12),
        help=(
            f"Calendar months per status chunk (default: {DEFAULT_STATUS_CHUNK_MONTHS}, "
            "quarterly). Keep the same setting on resume."
        ),
    )
    argument_parser.add_argument(
        "--retries",
        type=int,
        default=2,
        help="Transient retries per chunk; each paid attempt counts toward the ceiling.",
    )
    argument_parser.add_argument(
        "--estimate-only",
        action="store_true",
        help=(
            "Estimate without paid downloads. "
            "Repair mode checkpoints adoption under the writer lock."
        ),
    )

    # Existing unmanaged files need explicit adoption, including when an
    # estimate checkpoints reusable ranges before any paid requests begin.
    argument_parser.add_argument(
        "--repair",
        action="store_true",
        help="Adopt an interrupted unmanaged corpus or resume its repair manifest.",
    )

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
    """Run a managed download or explicit repair, retaining resumable checkpoints.

    Args:
        arguments: Command-line arguments, or None to read the process arguments.

    Returns:
        Zero on success, one on failure, or 130 after interruption.
    """

    from pricesanity.data.managed_download import ManagedDownload, DownloadBusyError, sanitized

    parser = build_download_argument_parser()

    # Adoption must be explicitly selected; existing files alone cannot authorize repair.
    args = parser.parse_args(arguments)
    repair = args.repair

    # Reject unusable cost ceilings before constructing any download state.
    if not isfinite(args.max_cost_usd) or args.max_cost_usd < 0:
        parser.error("--max-cost-usd must be finite and nonnegative.")

    # Positive chunks and bounded retries are required for the request to make progress.
    if (
        args.candlestick_chunk_years < 1
        or args.retries < 0
        or (args.status_chunk_years is not None and args.status_chunk_years < 1)
    ):
        parser.error("Chunk years must be positive; retries must be nonnegative.")

    plan = None

    # Keep planning and execution under one CLI error boundary so failures return a useful exit
    # code.
    try:
        request = DatabentoFetchRequest(args.start, args.end)
        paths = build_raw_data_paths(args.raw_data_directory, request)
        plan = ManagedDownload(
            request,
            paths.candlestick_csv_path.parent,
            candlestick_chunk_years=args.candlestick_chunk_years,
            status_chunk_years=args.status_chunk_years,
            status_chunk_months=args.status_chunk_months,
            repair=repair,
        )
        client = create_historical_client()

        # Estimate mode must avoid paid downloads while preserving its documented repair
        # checkpoints.
        if args.estimate_only:
            # Repair estimates adopt reusable ranges, so they need the same writer lock as
            # downloads.
            if repair:
                # Save range-count probes and adopted chunks once so the later
                # repair reuses them instead of repeating metadata work.
                with plan.operation():
                    plan.prepare_repair(client)
                    remaining = plan.estimate(client)
                    plan.summary(args.max_cost_usd, remaining)
            else:
                remaining = plan.estimate(client)
                plan.summary(args.max_cost_usd, remaining)

            # Make the repair estimate mutation explicit in the terminal confirmation.
            if repair:
                print(
                    "Estimate only: no time-series requests made; repair "
                    "probes and adopted ranges were checkpointed."
                )
            else:
                print(
                    "Estimate only: no time-series requests or local "
                    "checkpoints written."
                )
        else:
            plan.run(client, max_cost_usd=args.max_cost_usd, retries=args.retries)

        return 0

    # Handle interruption alongside failures so both retain checkpoints and return a nonzero
    # status.
    except (Exception, KeyboardInterrupt) as error:
        # A losing writer must exit before any managed diagnostic or checkpoint can be written.
        if isinstance(error, DownloadBusyError):
            import sys

            print(f"Download busy: {error}. Managed state unchanged.", file=sys.stderr)

            return 1

        context = plan.context if plan is not None else "local validation"
        log = str(plan.log_path) if plan is not None and plan._error_logged else None

        # Use the separate preflight log when this process did not record a managed failure.
        if log is None:
            # Preflight cannot write into an unmanaged/conflicting corpus. Keep
            # its traceback in the ignored raw parent instead.
            import traceback

            # Fallback diagnostics must not replace the original failure when the filesystem is
            # also unavailable.
            try:
                args.raw_data_directory.mkdir(parents=True, exist_ok=True)
                failure_path = args.raw_data_directory / "download-errors.log"

                # Append outside the managed corpus because preflight failure may occur before
                # writer ownership.
                with failure_path.open("a", encoding="utf-8") as stream:
                    from datetime import datetime, timezone

                    stream.write(datetime.now(timezone.utc).isoformat() + " " + context + "\n")
                    stream.write(sanitized(traceback.format_exc()) + "\n")

                log = str(failure_path)

            # The terminal still reports the original problem when even the fallback log cannot
            # be written.
            except OSError:
                log = "unavailable (could not write failure log)"

        message = sanitized(str(error) or type(error).__name__).splitlines()[0][:400]
        resumable = plan is not None and plan.manifest_path.exists()
        import sys

        print(f"Failed {context}: {message}. Resumable: {resumable}. Log: {log}", file=sys.stderr)

        return 130 if isinstance(error, KeyboardInterrupt) else 1


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
    *,
    max_cost_usd: float,
    candlestick_chunk_years: int = DEFAULT_CANDLE_CHUNK_YEARS,
    status_chunk_years: int | None = None,
    status_chunk_months: int | None = None,
) -> DatabentoRawDataPaths:
    """Download or resume a managed request within an additional-cost ceiling.

    Args:
        client: Databento client providing metadata and historical requests.
        request: Dataset, instrument, schemas, and inclusive/exclusive request boundaries.
        raw_data_paths: Destination paths for the raw files belonging to one request.
        max_cost_usd: Approved maximum additional estimated request cost, including retries.
        candlestick_chunk_years: Calendar years per candlestick chunk.
        status_chunk_years: Explicit yearly status partition, mutually exclusive with months.
        status_chunk_months: Explicit monthly status partition; omitted settings use quarters.

    Returns:
        The original destination paths after successful completion.

    Raises:
        ValueError: For conflicting paths, invalid files or insufficient approval.
    """

    from pricesanity.data.managed_download import ManagedDownload

    directory = raw_data_paths.candlestick_csv_path.parent

    # All output paths must belong to one corpus so checkpoints cannot mix unrelated artifacts.
    if (
        raw_data_paths.status_csv_path != directory / "status.csv"
        or raw_data_paths.condition_json_path != directory / "condition.json"
        or raw_data_paths.candlestick_csv_path.name != "candlesticks.csv"
    ):
        raise ValueError("Managed downloads require standard filenames in one directory")

    plan = ManagedDownload(
        request,
        directory,
        candlestick_chunk_years=candlestick_chunk_years,
        status_chunk_years=status_chunk_years,
        status_chunk_months=status_chunk_months,
    )
    plan.run(client, max_cost_usd=max_cost_usd)

    return raw_data_paths


# Only direct execution should run the standalone estimator.
if __name__ == "__main__":
    raise SystemExit(main())
