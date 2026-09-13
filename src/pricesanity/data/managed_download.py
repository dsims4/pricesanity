"""Resumable synchronous downloads with explicit additional-cost protection."""

from dataclasses import asdict
from contextlib import contextmanager
from datetime import date, datetime, timedelta, timezone
import json
import math
import os
from pathlib import Path
import re
import shutil
import time
import traceback
import warnings
from typing import Any

from databento.common.error import BentoError, BentoHttpError
from requests.exceptions import ConnectionError as RequestConnectionError
from requests.exceptions import SSLError, Timeout as RequestTimeout

from pricesanity.data.databento_fetch import DatabentoFetchRequest
from pricesanity.data.download_config import (
    DEFAULT_CANDLE_CHUNK_YEARS,
    CONDITION_CHUNK_YEARS,
    status_chunk_config,
)
from pricesanity.data.download_progress import DownloadProgress
from pricesanity.data.download_files import (
    assemble_files,
    atomic_json,
    file_stats,
    split_csv_into_ranges,
    validate_conditions,
    validate_csv,
)

FORMAT_VERSION = 2
COMPONENTS = ("candlesticks", "status", "conditions")
FINAL_NAMES = {
    "candlesticks": "candlesticks.csv",
    "status": "status.csv",
    "conditions": "condition.json",
}
MAX_RETRIES = 2
BACKOFF_SECONDS = 1.0


class CostLimitError(ValueError):
    """An additional estimated request would exceed this invocation's ceiling."""


class DownloadBusyError(RuntimeError):
    """Another writer owns the request directory's OS lock."""


def yearly_ranges(start: date, end: date, years: int = 1) -> list[tuple[date, date]]:
    """Partition [start, end) at calendar-year boundaries, without gaps.

    Args:
        start: Inclusive beginning of the requested date range.
        end: Exclusive end of the requested date range.
        years: Number of calendar years in each chunk.

    Returns:
        Ordered contiguous date ranges covering the complete request.

    Raises:
        ValueError: If the date range is empty or the chunk size is not positive.
    """

    # Reject empty ranges and invalid chunk sizes before any files or requests are created.
    if years < 1 or start >= end:
        raise ValueError("Chunk years must be positive and start must precede end")

    result = []
    cursor = start

    # Advance to calendar-year boundaries so resumes request exactly the original yearly
    # partitions.
    while cursor < end:
        boundary = date(min(cursor.year + years, 9999), 1, 1)
        stop = min(boundary, end)

        # The maximum supported year cannot produce another January boundary.
        if stop <= cursor:
            stop = end

        result.append((cursor, stop))
        cursor = stop

    return result


def monthly_ranges(start: date, end: date, months: int) -> list[tuple[date, date]]:
    """Partition [start, end) at calendar-month boundaries, including partial edges.

    Args:
        start: Inclusive beginning of the requested date range.
        end: Exclusive end of the requested date range.
        months: Number of calendar months in each chunk.

    Returns:
        Ordered contiguous date ranges aligned to calendar boundaries.

    Raises:
        ValueError: If the range is empty or the month size cannot partition calendar years.
    """

    # Use month sizes that align consistently with calendar-year boundaries.
    if months not in (1, 2, 3, 4, 6, 12) or start >= end:
        raise ValueError("Chunk months must divide 12; start must precede end")

    result = []
    cursor = start

    # Advance by calendar months so quarterly boundaries remain stable across varying month
    # lengths.
    while cursor < end:
        boundary_month = ((cursor.month - 1) // months + 1) * months
        year, month = divmod(cursor.year * 12 + boundary_month, 12)
        stop = end if year > 9999 else min(date(year, month + 1, 1), end)
        result.append((cursor, stop))
        cursor = stop

    return result


def transient_error(error: BaseException) -> bool:
    """Retry only transport failures and explicitly temporary HTTP responses.

    Args:
        error: Exception raised by the current operation.

    Returns:
        True only when the failure is safe to retry as a transient interruption.
    """

    # Only explicitly temporary HTTP responses justify another network attempt.
    if isinstance(error, BentoHttpError):
        return error.http_status in (408, 429, 500, 502, 503, 504)

    # Certificate failures require a configuration fix rather than repeated requests.
    if isinstance(error, SSLError):
        return False

    # Transport interruptions can succeed when the same request is attempted again.
    if isinstance(error, (TimeoutError, ConnectionError, RequestConnectionError, RequestTimeout)):
        return True

    # BentoError wraps streaming interruptions without a dedicated subtype.
    if isinstance(error, BentoError):
        return any(
            text in str(error).lower()
            for text in (
                "response ended prematurely",
                "connection reset",
                "connection aborted",
                "connection timed out",
                "connection timeout",
                "server disconnected",
                "read timed out",
                "connection broken",
                "incomplete read",
            )
        )

    return False


def sanitized(text: str) -> str:
    """Redact credentials without serializing client objects or the environment.

    Args:
        text: Diagnostic text that may contain credentials.

    Returns:
        Diagnostic text with recognized credentials redacted.
    """

    # Remove the configured credential even when an SDK error embeds it outside a recognizable
    # header.
    api_key = os.environ.get("DATABENTO_API_KEY")

    # Remove the configured secret even when it appears outside a recognizable header.
    if api_key:
        text = text.replace(api_key, "[REDACTED]")

    text = re.sub(
        r"(?i)(authorization\s*[:=]\s*)(?:basic|bearer)?\s*[^\r\n]+", r"\1[REDACTED]", text
    )
    text = re.sub(
        r"(?i)((?:api[_-]?key|token|password)\s*['\"]?\s*[:=]\s*['\"]?)[^\s,'\"}]+",
        r"\1[REDACTED]",
        text,
    )
    text = re.sub(r"(?i)\b(?:Basic|Bearer)\s+[A-Za-z0-9_+/.=-]+", "[REDACTED]", text)
    text = re.sub(r"db-[A-Za-z0-9_-]{16,}", "[REDACTED]", text)

    return text


def now() -> str:
    """Return a UTC timestamp for checkpoint and diagnostic ordering.

    Returns:
        UTC timestamp formatted as an ISO 8601 string.
    """

    # Store checkpoint times in UTC so resume comparisons do not depend on local timezone
    # settings.
    return datetime.now(timezone.utc).isoformat()


def identity(request: DatabentoFetchRequest) -> dict:
    """Serialize the request into comparable manifest fields.

    Args:
        request: Dataset, instrument, schemas, and inclusive/exclusive request boundaries.

    Returns:
        Request fields with dates represented as ISO 8601 strings.
    """

    # Convert the request to serializable identity fields before saving or comparing a manifest.
    result = asdict(request)
    result.update(start_date=request.start_date.isoformat(), end_date=request.end_date.isoformat())

    return result


def request_parameters(
    request: DatabentoFetchRequest, component: str, start: str, end: str
) -> dict:
    """Build vendor parameters with exclusive range-ending semantics.

    Args:
        request: Dataset, instrument, schemas, and inclusive/exclusive request boundaries.
        component: Candlesticks, status, or conditions component being processed.
        start: Inclusive beginning of the requested date range.
        end: Exclusive end of the requested date range.

    Returns:
        Vendor parameters preserving the requested schema and date boundaries.
    """

    return {
        "dataset": request.dataset,
        "symbols": request.symbol,
        "stype_in": request.symbol_type,
        "start": date.fromisoformat(start),
        "end": date.fromisoformat(end),
        "schema": (
            request.candlestick_schema if component == "candlesticks" else request.status_schema
        ),
    }


class ManagedDownload:
    """One request plan; constructing it validates local state without network calls.

    Args:
        request: Exact dataset, symbol, schemas and exclusive date range.
        directory: Request directory, not the parent raw-data directory.
        repair: Permit adoption of valid ranges from an interrupted OHLC CSV.

    Raises:
        ValueError: For conflicting manifests, unknown files or invalid imported data.
    """

    def __init__(
        self,
        request: DatabentoFetchRequest,
        directory: Path,
        *,
        candlestick_chunk_years: int = DEFAULT_CANDLE_CHUNK_YEARS,
        status_chunk_years: int | None = None,
        status_chunk_months: int | None = None,
        repair: bool = False,
    ):
        """Validate the request and initialize its resumable local state.

        Args:
            request: Dataset, instrument, schemas, and inclusive/exclusive request boundaries.
            directory: Directory containing this request and its checkpoints.
            candlestick_chunk_years: Calendar years per candlestick chunk.
            status_chunk_years: Explicit yearly status partition, mutually exclusive with
                months.
            status_chunk_months: Explicit monthly status partition; omitted settings use
                quarters.
            repair: Whether to adopt validated ranges from an interrupted download.
        """

        # Keep the intended request separate from mutable completion state so resume can verify
        # their identity.
        self.request = request
        self.directory = Path(directory)
        self.manifest_path = self.directory / "manifest.json"
        self.log_path = self.directory / "download.log"
        self.repair = repair
        self.context = "planning"
        self.costs: dict[tuple[str, str], float] = {}

        # Construction performs planning only; writer ownership begins explicitly inside an
        # operation.
        self._locked = False
        self._error_logged = False
        self.progress = None
        self.retry = 0

        # Remember the planning snapshot so a later lock acquisition can detect another writer's
        # changes.
        self._loaded_from_disk = self.manifest_path.exists()

        # Resolve status units once so planning, saved configuration, and resume use the same
        # default.
        status_chunk_years, status_chunk_months = status_chunk_config(
            status_chunk_years, status_chunk_months
        )
        chunk_configuration = {
            "candlestick_chunk_years": candlestick_chunk_years,
            "status_chunk_years": status_chunk_years,
        }

        # Record explicit month settings so resume cannot silently change the partition.
        if status_chunk_months is not None:
            chunk_configuration["status_chunk_months"] = status_chunk_months

        components = {}

        # Plan each component separately because status and conditions need different request
        # partitions.
        for component in COMPONENTS:
            years = candlestick_chunk_years if component == "candlesticks" else status_chunk_years

            # Smaller spans limit work lost when a status stream is interrupted;
            # response row count alone does not predict whether a request finishes.
            ranges = (
                monthly_ranges(request.start_date, request.end_date, status_chunk_months)
                if component == "status" and status_chunk_months is not None
                else yearly_ranges(request.start_date, request.end_date, years)
            )

            # Condition metadata has its own coarse partition, independent of status request size.
            if component == "conditions":
                ranges = yearly_ranges(request.start_date, request.end_date, CONDITION_CHUNK_YEARS)

            # Initialize explicit range checkpoints; no planned chunk is trusted before its file
            # is validated.
            chunks = [
                {
                    "start": range_start.isoformat(),
                    "end": range_end.isoformat(),
                    "state": "pending",
                }
                for range_start, range_end in ranges
            ]
            components[component] = {
                "state": "pending",
                "chunks": chunks,
                "last_successful_exclusive_boundary": None,
            }

        # Build a candidate manifest in memory; existing unmanaged files do not cause automatic
        # publication.
        self.manifest = {
            "format_version": FORMAT_VERSION,
            "request": identity(request),
            "chunk_config": chunk_configuration,
            "original_full_estimate_usd": None,
            "created_at": now(),
            "updated_at": now(),
            "last_error": None,
            "condition_chunk_years": CONDITION_CHUNK_YEARS,
            "components": components,
        }

        # Reuse a checkpoint only after its identity and range definitions agree with this request.
        if self.manifest_path.exists():
            saved_manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))

            # Check identity and settings before accepting any saved chunks as belonging to this
            # request.
            for field in ("format_version", "request", "chunk_config"):
                # A different identity or configuration could associate these files with the wrong
                # request.
                if saved_manifest.get(field) != self.manifest[field]:
                    raise ValueError(
                        f"Conflicting manifest {field}; use the matching request/settings"
                    )

            # A missing or unexpected component would make completion and assembly ambiguous.
            if set(saved_manifest.get("components", {})) != set(COMPONENTS):
                raise ValueError("Invalid manifest components")

            # Compare the full partition for every component so a manifest cannot omit or
            # overlap ranges.
            for component in COMPONENTS:
                component_state = saved_manifest["components"][component]
                expected_chunks = components[component]["chunks"]
                saved_chunk_ranges = [
                    (chunk["start"], chunk["end"]) for chunk in component_state["chunks"]
                ]

                # Preserve the already-validated condition partition of an existing checkpoint.
                if component == "conditions" and "condition_chunk_years" not in saved_manifest:
                    # Already checkpointed requests retain their original, verified
                    # partition. New requests use the independent annual policy.
                    expected_chunks = components["status"]["chunks"]

                # Exact range matching prevents gaps, overlap, and accidental reinterpretation on
                # resume.
                if saved_chunk_ranges != [
                    (chunk["start"], chunk["end"]) for chunk in expected_chunks
                ]:
                    raise ValueError("Conflicting manifest chunk ranges")

            # Reject an unsupported conditions policy rather than reinterpret stored checkpoints.
            if (
                saved_manifest.get("condition_chunk_years", CONDITION_CHUNK_YEARS)
                != CONDITION_CHUNK_YEARS
            ):
                raise ValueError("Conflicting conditions chunk policy")

            self.manifest = saved_manifest

            # An ordinary download must not be reinterpreted as source adoption.
            if repair and "repair" not in saved_manifest:
                raise ValueError("This manifest belongs to a normal download, not a repair")

            # Require explicit repair mode to preserve the original adoption workflow.
            if not repair and "repair" in saved_manifest:
                raise ValueError(
                    "This manifest belongs to a repair; resume it with "
                    "pricesanity-download --repair"
                )

        # Existing files need an explicit adoption path before they can become managed evidence.
        elif self.directory.exists():
            # Normal downloads cannot infer whether unmanaged files are complete or related.
            if not repair:
                raise ValueError(
                    "Unmanaged directory; use pricesanity-download --repair "
                    "for the known partial corpus"
                )

            allowed_paths = {
                "candlesticks.csv",
                "status.csv",
                "condition.json",
                ".DS_Store",
                ".download.lock",
            }

            # Reject unfamiliar source files rather than silently adopt unrelated market data.
            if set(file_path.name for file_path in self.directory.iterdir()) - allowed_paths:
                raise ValueError("Unmanaged repair directory contains conflicting files")

            candle_source = self.directory / "candlesticks.csv"

            # Repair requires a concrete source file whose records can be validated.
            if not candle_source.exists():
                raise ValueError("Repair requires an existing candlesticks.csv")

            # Existing records must be structurally trustworthy before any
            # complete ranges are adopted from the interrupted response.
            sources = {
                "candlesticks": {
                    "name": candle_source.name,
                    "stats": validate_csv(
                        candle_source,
                        "candlesticks",
                        request.start_date,
                        request.end_date,
                    ),
                }
            }
            status_source = self.directory / "status.csv"

            # An existing status file can contribute validated ranges alongside the candle source.
            if status_source.exists():
                sources["status"] = {
                    "name": status_source.name,
                    "stats": validate_csv(
                        status_source,
                        "status",
                        request.start_date,
                        request.end_date,
                    ),
                }

            self.manifest["repair"] = {
                "sources": sources,
                "adoption_states": {
                    component: "pending" for component in sources
                },
            }
            self.manifest["original_full_estimate_note"] = (
                "Unknown: repairing a manifest-less interrupted download"
            )

        # There is nothing to adopt when the requested repair directory does not exist.
        elif repair:
            raise ValueError("Repair requires an existing candlesticks.csv")

        self._check_files()
        self.validate_completed()

    def chunk_path(self, component: str, chunk: dict) -> Path:
        """Locate the completed file for one manifest chunk.

        Args:
            component: Candlesticks, status, or conditions component being processed.
            chunk: Manifest entry describing one requested date range.

        Returns:
            Path reserved for the completed chunk.
        """

        # Conditions are JSON metadata while candle and status chunks retain the vendor CSV
        # schema.
        suffix = ".json" if component == "conditions" else ".csv"

        return (
            self.directory / "chunks" / component / (chunk["start"] + "_" + chunk["end"] + suffix)
        )

    def _check_files(self) -> None:
        """Reject files that cannot safely belong to this managed request.

        Raises:
            ValueError: If linked or unrecognized files could conflict with managed output.
        """

        # A linked directory could redirect managed writes outside the intended corpus.
        if self.directory.is_symlink():
            raise ValueError("Managed directory cannot be a symlink")

        # A new request has no existing files that could conflict with the plan.
        if not self.directory.exists():
            return

        allowed_paths = {self.manifest_path, self.log_path, self.directory / ".download.lock"}

        # Permit each final artifact and its staging path because interrupted assembly may leave
        # either.
        for name in FINAL_NAMES.values():
            path = self.directory / name
            allowed_paths.update((path, self.partial_path(path)))

        # Derive allowed chunk paths from the manifest rather than trusting arbitrary files in
        # the directory.
        for component, component_state in self.manifest["components"].items():
            # Allow only the completed and temporary filenames associated with each planned
            # range.
            for chunk in component_state["chunks"]:
                path = self.chunk_path(component, chunk)
                allowed_paths.update((path, self.partial_path(path)))

        # Inspect nested paths too because conflicting files can live below the component
        # directories.
        for path in self.directory.rglob("*"):
            # Linked files could replace or expose content outside the validated managed paths.
            if path.is_symlink():
                raise ValueError("Symlinks are not supported in managed downloads")

            # Ignore only known system and checkpoint files; unfamiliar files may belong to another
            # request.
            if (
                path.is_file()
                and path not in allowed_paths
                and path.name != ".DS_Store"
                and not path.name.startswith(".checkpoint-")
            ):
                raise ValueError("Managed directory contains an unrecognized file")

    @staticmethod
    def partial_path(path: Path) -> Path:
        """Locate a temporary file that cannot be mistaken for completed data.

        Args:
            path: File to read or publish.

        Returns:
            Path reserved for uncommitted file contents.
        """

        # Separate unfinished filenames from committed chunks so resume cannot confuse partial
        # data with completion.
        return path.with_name(path.stem + ".partial" + path.suffix)

    def validate_completed(self) -> None:
        """Check hashes and recover a renamed chunk whose checkpoint was interrupted.

        Raises:
            ValueError: If a repaired final file is missing or differs from its saved checksum.
        """

        # Reconstruct component completion from verified files rather than trusting saved state
        # alone.
        for component, component_state in self.manifest["components"].items():
            # After repair cleanup, the final checksum is the remaining source of integrity
            # evidence.
            if component_state.get("chunks_cleaned"):
                final_stats = component_state.get("final_stats")
                final_path = self.directory / FINAL_NAMES[component]

                # Cleaned repairs cannot recover from chunks, so reject missing or changed final
                # artifacts.
                if (
                    not final_stats
                    or not final_path.exists()
                    or file_stats(final_path)
                    != {key: final_stats[key] for key in ("bytes", "sha256")}
                ):
                    raise ValueError(
                        f"Completed repaired {component} file changed or is missing"
                    )

                # The verified final file now supplies integrity evidence for chunks removed by
                # repair cleanup.
                for chunk in component_state["chunks"]:
                    chunk["state"] = "complete"

                self.refresh(component)
                continue

            # Recheck every chunk independently so later valid ranges survive a gap earlier in
            # the request.
            for chunk in component_state["chunks"]:
                path = self.chunk_path(component, chunk)
                is_valid_chunk = False

                # A missing completed filename remains pending regardless of a previous checkpoint.
                if path.exists():
                    # A recorded checksum detects edits to a previously validated chunk.
                    if chunk.get("stats"):
                        is_valid_chunk = file_stats(path) == {
                            key: chunk["stats"][key] for key in ("bytes", "sha256")
                        }

                    # A validated renamed file can recover completion after an interrupted
                    # checkpoint write.
                    elif "expected_rows" in chunk and component != "conditions":
                        # Recover a renamed chunk only if its content and expected row count
                        # still validate.
                        try:
                            chunk["stats"] = validate_csv(
                                path,
                                component,
                                date.fromisoformat(chunk["start"]),
                                date.fromisoformat(chunk["end"]),
                                chunk["expected_rows"],
                            )
                            is_valid_chunk = True

                        # Leave an invalid recovery candidate pending so a later download can
                        # replace the entire range.
                        except ValueError:
                            pass

                # The current file evidence determines reuse even if an earlier manifest claimed
                # completion.
                chunk["state"] = "complete" if is_valid_chunk else "pending"

            final_stats = component_state.get("final_stats")
            final_path = self.directory / FINAL_NAMES[component]

            # A final artifact cannot stay trusted when a required chunk or its own checksum is
            # invalid.
            if any(chunk["state"] != "complete" for chunk in component_state["chunks"]) or (
                final_stats
                and (
                    not final_path.exists()
                    or file_stats(final_path)
                    != {
                        statistic_name: final_stats[statistic_name]
                        for statistic_name in ("bytes", "sha256")
                    }
                )
            ):
                component_state.pop("final_stats", None)

            self.refresh(component)

    def refresh(self, component: str) -> None:
        """Recalculate component readiness from verified chunk and final-file state.

        Args:
            component: Candlesticks, status, or conditions component being processed.
        """

        # Select this component's checkpoint without mixing candle, status, or condition
        # completion evidence.
        component_state = self.manifest["components"][component]
        chunks = component_state["chunks"]
        boundary = None

        # Measure the contiguous boundary separately from completion counts so later chunks
        # cannot hide a gap.
        for chunk in chunks:
            # The contiguous boundary stops at the first gap, even if later chunks are complete.
            if chunk["state"] != "complete":
                break

            boundary = chunk["end"]

        component_state["last_successful_exclusive_boundary"] = boundary

        # Chunk completion alone is insufficient; the assembled final file must also have been
        # published.
        all_chunks_complete = all(chunk["state"] == "complete" for chunk in chunks)
        final = self.directory / FINAL_NAMES[component]
        is_final_published = bool(component_state.get("final_stats")) and final.exists()
        component_state["state"] = (
            "complete"
            if all_chunks_complete and is_final_published
            else "partial" if any(chunk["state"] != "pending" for chunk in chunks) else "pending"
        )

    def save(self) -> None:
        """Publish one checkpoint without exposing partially written JSON."""

        # Change the checkpoint revision before publication so a competing stale plan can detect
        # this write.
        self.manifest["updated_at"] = now()
        atomic_json(self.manifest_path, self.manifest)
        self._loaded_from_disk = True

    def missing(self) -> list[tuple[str, dict]]:
        """Return only ranges that still require a valid completed chunk.

        Returns:
            Component names and chunk entries that are not complete.
        """

        return [
            (component, chunk)
            for component, component_state in self.manifest["components"].items()
            for chunk in component_state["chunks"]
            if chunk["state"] != "complete"
        ]

    def prepare_repair(self, client: Any) -> None:
        """Checkpoint repair adoption while holding the shared writer lock.

        Args:
            client: Databento client providing metadata and historical requests.
        """

        # Normal downloads have no existing source ranges to adopt.
        if not self.repair:
            return

        # Direct repair callers need the same lock as CLI downloads because adoption writes
        # checkpoints.
        with self.operation():
            self._prepare_repair_locked(client)

    def _prepare_repair_locked(self, client: Any) -> None:
        """Adopt complete yearly ranges from the interrupted original OHLC CSV.

        Expected counts are saved immediately after their first metadata query.
        A resumed repair therefore probes only ranges whose count was never
        checkpointed. Ranges with fewer or extra records are downloaded again
        as a whole because absence alone cannot locate the missing records.

        Args:
            client: Databento client providing metadata and historical requests.

        Raises:
            ValueError: If source integrity or expected record-count evidence is invalid.
        """

        # Only explicit repair checkpoints carry source-adoption evidence.
        repair = self.manifest.get("repair")

        # Only a repair manifest contains the source evidence needed for adoption.
        if repair is None:
            return

        self.directory.mkdir(parents=True, exist_ok=True)
        self.save()

        # Adopt each available source independently so missing status data does not discard
        # reusable candles.
        for component, source_record in repair["sources"].items():
            # Completed adoption is checkpointed so resume does not split the same source again.
            if repair["adoption_states"].get(component) == "complete":
                continue

            source = self.directory / source_record["name"]
            expected_source = {key: source_record["stats"][key] for key in ("bytes", "sha256")}

            # Do not adopt from a source that changed after its initial validation.
            if not source.exists() or file_stats(source) != expected_source:
                raise ValueError(
                    f"The interrupted {component} source changed during repair"
                )

            # Select this component's checkpoint without mixing candle, status, or condition
            # completion evidence.
            component_state = self.manifest["components"][component]

            # Establish completeness for each source range before treating its copied rows as
            # reusable.
            for chunk in component_state["chunks"]:
                # Reuse saved completeness probes instead of repeating metadata
                # calls during resume.
                if "expected_rows" in chunk:
                    continue

                self.context = (
                    f"{component} {chunk['start']} to {chunk['end']} " "completeness probe"
                )
                chunk["expected_rows"] = int(
                    client.metadata.get_record_count(
                        **request_parameters(
                            self.request,
                            component,
                            chunk["start"],
                            chunk["end"],
                        )
                    )
                )

                # A negative count is invalid evidence and cannot establish source completeness.
                if chunk["expected_rows"] < 0:
                    raise ValueError(
                        f"Expected {component} record count cannot be negative"
                    )

                self.save()

            self.context = f"{component} local range adoption"
            destinations = [
                (
                    date.fromisoformat(chunk["start"]),
                    date.fromisoformat(chunk["end"]),
                    self.chunk_path(component, chunk),
                )
                for chunk in component_state["chunks"]
            ]
            split_csv_into_ranges(source, destinations, component=component)
            adopted_rows = 0

            # Compare each copied range with its saved vendor count before declaring adoption
            # successful.
            for chunk in component_state["chunks"]:
                path = self.chunk_path(component, chunk)

                # A failed range check should discard that adopted chunk while leaving the
                # original source intact.
                try:
                    chunk_statistics = validate_csv(
                        path,
                        component,
                        date.fromisoformat(chunk["start"]),
                        date.fromisoformat(chunk["end"]),
                        chunk["expected_rows"],
                    )

                # A count mismatch cannot locate missing rows, so request that whole range
                # again.
                except ValueError:
                    path.unlink(missing_ok=True)
                    chunk.pop("stats", None)
                    chunk["state"] = "pending"
                else:
                    # Save integrity evidence before marking the range complete so resume can
                    # verify what was committed.
                    chunk["stats"] = chunk_statistics
                    chunk["state"] = "complete"
                    adopted_rows += chunk_statistics["rows"]

            repair["adoption_states"][component] = "complete"

            # Record how much source data survived verification so repair progress reports
            # actual reused rows.
            source_record["adopted_rows"] = adopted_rows
            self.refresh(component)
            self.save()
            self.update_progress()

    def estimate(self, client: Any) -> float:
        """Estimate only missing paid requests; metadata/conditions are not time-series purchases.

        Args:
            client: Databento client providing metadata and historical requests.

        Returns:
            Total estimated cost of the remaining paid requests.
        """

        # The active operation already owns warning capture and diagnostic persistence.
        if self._locked:
            return self._estimate(client)

        # Temporarily capture estimate warnings without changing the managed request directory.
        with warnings.catch_warnings():
            warnings.simplefilter("always")
            warnings.showwarning = self.log_readonly_warning

            return self._estimate(client)

    def log_readonly_warning(
        self, message, category, filename, lineno, file=None, line=None
    ) -> None:
        """Retain estimate warnings without mutating the managed corpus.

        Args:
            message: Complete warning or error text to retain in the diagnostic log.
            category: Warning class or diagnostic category identifying the source.
            filename: Source filename supplied by the Python warning machinery.
            lineno: Source line number supplied by the Python warning machinery.
            file: Optional warning stream accepted for Python callback compatibility.
            line: Optional source text accepted for Python callback compatibility.
        """

        # Estimates can run beside a writer. Keep diagnostics outside its corpus
        # and never checkpoint the in-memory warning count from this read-only path.
        diagnostic_counts = self.manifest.setdefault("diagnostics", {"warnings": 0, "errors": 0})
        diagnostic_counts["warnings"] += 1
        self.directory.parent.mkdir(parents=True, exist_ok=True)
        diagnostic_entry = dict(
            time=now(),
            kind="warning",
            category=category.__name__,
            context=self.context,
            message=str(message),
            filename=filename,
            lineno=lineno,
        )

        # Read-only diagnostics belong beside the corpus because this operation does not own its
        # writer lock.
        with (self.directory.parent / "download-errors.log").open("a", encoding="utf-8") as stream:
            stream.write(sanitized(json.dumps(diagnostic_entry)) + "\n")

    def _estimate(self, client: Any) -> float:
        """Calculate the remaining cost from requests that are not complete.

        Args:
            client: Databento client providing metadata and historical requests.

        Returns:
            Total estimated cost of the remaining paid requests.

        Raises:
            ValueError: If individual or combined cost estimates are negative or nonfinite.
        """

        # Recalculate missing-request estimates so a resume does not reuse an outdated
        # invocation's costs.
        self.costs.clear()

        # Estimate only incomplete paid ranges so valid checkpoints do not consume the resume
        # budget.
        for component, chunk in self.missing():
            self.context = f"{component} {chunk['start']} to {chunk['end']} estimate"
            value = (
                0.0
                if component == "conditions"
                else float(
                    client.metadata.get_cost(
                        **request_parameters(self.request, component, chunk["start"], chunk["end"])
                    )
                )
            )

            # An invalid vendor estimate cannot safely authorize a paid request.
            if not math.isfinite(value) or value < 0:
                raise ValueError("Cost estimates must be finite and nonnegative")

            self.costs[(component, chunk["start"])] = value

        remaining_cost = sum(self.costs.values())

        # Reject overflow in the combined estimate before comparing it with the approved ceiling.
        if not math.isfinite(remaining_cost):
            raise ValueError("Combined cost estimate is not finite")

        # Retain the original estimate across resumes; repair cannot reconstruct the original
        # purchase.
        if self.manifest["original_full_estimate_usd"] is None and not self.repair:
            self.manifest["original_full_estimate_usd"] = remaining_cost

        self.manifest["remaining_estimate_usd"] = remaining_cost

        return remaining_cost

    def summary(self, ceiling: float, remaining: float) -> None:
        """Show the request identity and approved remaining-cost comparison.

        Args:
            ceiling: Approved additional estimated cost for this invocation.
            remaining: Estimated cost of the requests that still require downloading.
        """

        # Display the original estimate separately from this invocation's remaining additional
        # cost.
        original = self.manifest["original_full_estimate_usd"]
        full_estimate_text = (
            f"${original:.6f}" if original is not None else "unknown (repairing existing OHLC)"
        )
        count = sum(
            len(component_state["chunks"])
            for component_state in self.manifest["components"].values()
        )
        print(
            f"{self.request.symbol} | {self.request.dataset} | "
            f"{self.request.candlestick_schema}, {self.request.status_schema}"
        )
        print(f"{self.request.start_date} inclusive to {self.request.end_date} exclusive")
        print(
            f"Original full estimate: {full_estimate_text}; "
            f"remaining estimate: ${remaining:.6f}; "
            f"approved additional maximum: ${ceiling:.6f}"
        )
        print(f"Output: {self.directory}; chunks: {count} total, {len(self.missing())} remaining")
        diagnostic_counts = self.manifest.get("diagnostics", {})

        # Only display diagnostic totals when this plan has recorded diagnostic history.
        if diagnostic_counts:
            print(
                f"Warnings: {diagnostic_counts['warnings']} | "
                f"Errors: {diagnostic_counts['errors']}"
            )

    @contextmanager
    def operation(self):
        """Own all managed mutations and final-error logging under one OS lock.

        Raises:
            DownloadBusyError: If another writer owns the request directory.
            ValueError: If the checkpoint changed before lock acquisition.
        """

        # Nested repair work must reuse the lock already held by its outer operation.
        if self._locked:
            yield

            return

        import fcntl

        self.directory.mkdir(parents=True, exist_ok=True)

        # Keep the lock file open for the whole operation; its existence alone does not confer
        # ownership.
        with (self.directory / ".download.lock").open("a") as lock:
            # Attempt ownership without waiting so a second CLI invocation can report contention
            # immediately.
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

            # A losing writer must exit before logging or checkpointing inside the managed
            # directory.
            except BlockingIOError:
                raise DownloadBusyError(
                    "Another download owns this request directory; try again when it finishes"
                ) from None

            # A contender may have finished between construction and acquisition.
            # Never overwrite its manifest with the plan read before the lock.
            if self.manifest_path.exists() != self._loaded_from_disk:
                raise ValueError("Download state changed; restart to revalidate it")

            # Recheck the checkpoint read before locking so another writer cannot be overwritten.
            if self._loaded_from_disk:
                saved_manifest = json.loads(self.manifest_path.read_text())

                # A newer checkpoint must be reloaded before this process can safely mutate it.
                if saved_manifest.get("updated_at") != self.manifest.get("updated_at"):
                    raise ValueError("Download state changed; restart to revalidate it")

            self._locked = True
            self._error_logged = False
            self.progress = None
            self.retry = 0
            started = time.monotonic()

            # Keep warning capture, failure logging, and final display cleanup within the lock
            # lifetime.
            try:
                # Restore the previous warning handler when the operation ends, including
                # failure and interruption.
                with warnings.catch_warnings():
                    # Persist each emitted warning, including repeated SDK warnings
                    # that Python would otherwise suppress at the same source line.
                    warnings.simplefilter("always")
                    warnings.showwarning = self.log_warning
                    yield

            # The outer operation owns the final failure entry so nested retry handling does not
            # log it twice.
            except (Exception, KeyboardInterrupt) as error:
                # Try to retain diagnostic evidence without letting a second disk error hide the
                # original exception.
                try:
                    self.log_failure(error)
                    self._error_logged = True

                # Failure to write diagnostics leaves the original error available for the CLI
                # fallback log.
                except OSError:
                    # Keep the original failure if disk trouble also prevents
                    # diagnostics. The CLI can try its separate preflight log.
                    pass

                raise
            finally:
                # Release the instance ownership flag even if writing the final terminal summary
                # fails.
                try:
                    # Only operations that created a display need to finalize its visible summary.
                    if self.progress is not None:
                        self.update_progress(final=True)
                        print(
                            f"Elapsed: {time.monotonic()-started:.1f}s | Output: {self.directory}",
                            flush=True,
                        )
                finally:
                    # A broken output pipe must not leave this instance believing
                    # it owns a lock that the file context is about to release.
                    self._locked = False

    def update_progress(self, *, final=False, event=False) -> None:
        """Render the latest verified state when a display is active.

        Args:
            final: Whether this update is the final operation summary.
            event: Whether the update must appear in redirected output.
        """

        # Construction and read-only planning do not require an active terminal display.
        if self.progress is not None:
            diagnostic_counts = self.manifest.get("diagnostics", {})
            self.progress.render(
                self.manifest,
                self.context,
                diagnostic_counts.get("warnings", 0),
                diagnostic_counts.get("errors", 0),
                retry=self.retry,
                final=final,
                event=event,
            )

    def log_warning(self, message, category, filename, lineno, file=None, line=None) -> None:
        """Persist a Python warning and update its cumulative counter.

        Args:
            message: Complete warning or error text to retain in the diagnostic log.
            category: Warning class or diagnostic category identifying the source.
            filename: Source filename supplied by the Python warning machinery.
            lineno: Source line number supplied by the Python warning machinery.
            file: Optional warning stream accepted for Python callback compatibility.
            line: Optional source text accepted for Python callback compatibility.
        """

        self._write_diagnostic(
            "warning", category.__name__, str(message), filename=filename, lineno=lineno
        )
        self.save()
        self.update_progress()

    def _write_diagnostic(
        self, kind: str, category: str, message: str, **diagnostic_details
    ) -> None:
        """Record a sanitized diagnostic while the writer lock is owned.

        Args:
            kind: Warning or error classification used for cumulative counters.
            category: Warning class or diagnostic category identifying the source.
            message: Complete warning or error text to retain in the diagnostic log.

        Raises:
            RuntimeError: If the current operation does not own the writer lock.
        """

        # Diagnostics mutate managed state and therefore require the same
        # ownership as chunk writes.
        if not self._locked:
            raise RuntimeError("Managed diagnostics require the writer lock")

        diagnostic_counts = self.manifest.setdefault("diagnostics", {"warnings": 0, "errors": 0})
        diagnostic_counts["warnings" if kind == "warning" else "errors"] += 1
        diagnostic_entry = dict(
            time=now(),
            kind=kind,
            context=self.context,
            category=category,
            message=message,
            request=self.manifest["request"],
            **diagnostic_details,
        )

        # Append full diagnostics while ownership is held so competing writers cannot interleave
        # entries.
        with self.log_path.open("a", encoding="utf-8") as stream:
            stream.write(sanitized(json.dumps(diagnostic_entry)) + "\n")

    def log_failure(self, error: BaseException) -> None:
        """Preserve exception context and checkpoint the most recent failure.

        Args:
            error: Exception raised by the current operation.
        """

        # Retain traceback context while stripping secrets before anything reaches a persistent
        # log.
        diagnostic_details = sanitized(traceback.format_exc())
        self._write_diagnostic(
            "error",
            type(error).__name__,
            str(error) or type(error).__name__,
            traceback=diagnostic_details,
            components=self.manifest["components"],
        )
        self.manifest["last_error"] = {
            "time": now(),
            "context": self.context,
            "message": sanitized(str(error) or type(error).__name__),
        }
        self.save()

    def run(
        self, client: Any, *, max_cost_usd: float, retries: int = MAX_RETRIES, sleep=time.sleep
    ) -> None:
        """Download missing chunks and atomically assemble the requested files.

        Each attempted paid request reserves its estimate, including retries.
        The ceiling therefore bounds estimated additional spending even when a
        failed response may have been billed. Actual billing remains vendor-owned.

        Args:
            client: Databento client providing metadata and historical requests.
            max_cost_usd: Approved maximum additional estimated request cost, including retries.
            retries: Maximum transient retries permitted after the initial attempt.
            sleep: Backoff function; tests can replace it without waiting.

        Raises:
            ValueError: If cost or retry settings are invalid.
            CostLimitError: If required paid requests exceed the approved additional estimate.
        """

        # Reject invalid budgets and retry limits before entering the mutation path.
        if not math.isfinite(max_cost_usd) or max_cost_usd < 0 or retries < 0:
            raise ValueError(
                "Cost ceiling must be finite/nonnegative; retries must be nonnegative"
            )

        # Hold writer ownership from validation through final publication and any repair
        # cleanup.
        with self.operation():
            self.validate_completed()
            self.prepare_repair(client)
            remaining_cost = self.estimate(client)
            self.summary(max_cost_usd, remaining_cost)
            self.progress = DownloadProgress()

            # The approved additional budget must cover every currently missing paid request.
            if remaining_cost > max_cost_usd:
                raise CostLimitError(
                    f"Remaining estimate ${remaining_cost:.6f} "
                    f"exceeds approved maximum ${max_cost_usd:.6f}"
                )

            self.save()

            # The remaining estimate covers each initial paid attempt; only started paid retries
            # add reservations.
            reserved_cost = remaining_cost

            # Process only missing ranges; completed chunks remain the durable resume points.
            for component, chunk in self.missing():
                # Select this component's checkpoint without mixing candle, status, or condition
                # completion evidence.
                component_state = self.manifest["components"][component]
                self.context = f"{component} {chunk['start']} to {chunk['end']}"
                path = self.chunk_path(component, chunk)
                path.parent.mkdir(parents=True, exist_ok=True)
                partial = self.partial_path(path)
                chunk_cost = self.costs[(component, chunk["start"])]
                chunk["state"] = "partial"
                self.refresh(component)
                self.save()

                # Track paid attempts separately from loop retries because a metadata failure is
                # not another data purchase.
                paid_attempts = 0

                # Bound attempts per chunk so a persistent failure cannot retry indefinitely or
                # escape the cost ceiling.
                for attempt in range(retries + 1):
                    self.retry = attempt
                    self.update_progress()

                    # Retry the whole chunk because an interrupted in-memory response has no
                    # durable partial records.
                    try:
                        partial.unlink(missing_ok=True)
                        vendor_parameters = request_parameters(
                            self.request, component, chunk["start"], chunk["end"]
                        )

                        # Conditions use the free metadata endpoint and its inclusive end-date
                        # convention.
                        if component == "conditions":
                            records = client.metadata.get_dataset_condition(
                                dataset=self.request.dataset,
                                start_date=vendor_parameters["start"],
                                end_date=vendor_parameters["end"] - timedelta(days=1),
                            )
                            records = sorted(records, key=lambda row: row["date"])
                            atomic_json(partial, records)
                            chunk_statistics = validate_conditions(
                                partial, vendor_parameters["start"], vendor_parameters["end"]
                            )
                        else:
                            # Repair already probed OHLC completeness once;
                            # reuse that checkpoint instead of probing again.
                            if "expected_rows" not in chunk:
                                chunk["expected_rows"] = int(
                                    client.metadata.get_record_count(**vendor_parameters)
                                )
                                self.save()

                            # The initial estimate already covers the first paid attempt for this
                            # chunk.
                            if paid_attempts:
                                # Reserve additional cost before another paid attempt can begin.
                                if reserved_cost + chunk_cost > max_cost_usd:
                                    raise CostLimitError(
                                        "Retry would exceed additional-cost ceiling; "
                                        "resume with an approved ceiling"
                                    )

                                reserved_cost += chunk_cost

                            paid_attempts += 1

                            # get_range completes in memory before CSV writing;
                            # durability starts when this whole chunk is saved.
                            store = client.timeseries.get_range(**vendor_parameters)
                            store.to_csv(
                                partial, pretty_px=True, pretty_ts=True, map_symbols=True, mode="x"
                            )

                            # Sync the completed CSV before checksum validation and atomic
                            # publication make it a checkpoint.
                            with partial.open("rb") as stream:
                                os.fsync(stream.fileno())

                            chunk_statistics = validate_csv(
                                partial,
                                component,
                                vendor_parameters["start"],
                                vendor_parameters["end"],
                                chunk["expected_rows"],
                            )

                        # Persist verified stats first to recover a crash between
                        # rename and the final complete-state checkpoint.
                        chunk["stats"] = chunk_statistics
                        self.save()
                        partial.replace(path)
                        chunk["state"] = "complete"
                        self.refresh(component)
                        self.save()
                        self.update_progress()
                        break

                    # Classify the failed attempt before deciding whether another bounded
                    # request is safe.
                    except Exception as error:
                        # Stop on permanent failures or exhausted retries so
                        # unsafe requests are not repeated.
                        if not transient_error(error) or attempt >= retries:
                            raise

                        # The outer operation owns the final failure. Only
                        # failures actually being retried are logged here.
                        self.log_failure(error)
                        self.retry = attempt + 1
                        self.update_progress(event=True)
                        sleep(BACKOFF_SECONDS * 2**attempt)

            # Assemble each component locally after its chunks finish, preserving them if final
            # publication fails.
            for component, component_state in self.manifest["components"].items():
                self.context = f"{component} local assembly"
                destination = self.directory / FINAL_NAMES[component]
                expected_stats = component_state.get("final_stats")

                # A matching final checksum makes local reassembly unnecessary.
                if (
                    expected_stats
                    and destination.exists()
                    and file_stats(destination)
                    == {
                        statistic_name: expected_stats[statistic_name]
                        for statistic_name in ("bytes", "sha256")
                    }
                ):
                    continue

                component_state["final_stats"] = assemble_files(
                    [self.chunk_path(component, chunk) for chunk in component_state["chunks"]],
                    destination,
                    component,
                    self.request.start_date,
                    self.request.end_date,
                    sum(chunk["stats"]["rows"] for chunk in component_state["chunks"]),
                )
                self.refresh(component)
                self.save()

            self.manifest["last_error"] = None

            # Only repair owns the cleanup policy for its now-redundant adopted chunks.
            if self.repair:
                # Once every merged final has a verified checksum, repair
                # chunks have served their recovery purpose. The manifest
                # retains their counts and the final-file checksums.
                for component_state in self.manifest["components"].values():
                    component_state["chunks_cleaned"] = True

                self.save()
                chunk_directory = self.directory / "chunks"

                # Cleanup is repeatable when a prior successful attempt already removed the chunk
                # directory.
                if chunk_directory.exists():
                    shutil.rmtree(chunk_directory)

            self.save()
