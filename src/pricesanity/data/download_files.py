"""Streaming validation and atomic local assembly of licensed download files."""

import csv
import hashlib
import json
import math
import os
import re
from datetime import date, datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile

OHLC_HEADER = {"ts_event", "open", "high", "low", "close"}
STATUS_HEADER = {"ts_event", "reason", "trading_event", "is_trading"}


def atomic_json(path: Path, value: dict | list) -> None:
    """Publish a flushed JSON checkpoint without exposing a half-written file.

    Args:
        path: File to read or publish.
        value: JSON-compatible checkpoint data to publish atomically.
    """

    # Track whether staging began so cleanup also works when temporary-file creation fails.
    temporary_path = None

    # Always remove an abandoned temporary checkpoint while preserving the published manifest.
    try:
        # Stage beside the destination so replacement stays on the same filesystem.
        with NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, prefix=".checkpoint-", delete=False
        ) as stream:
            temporary_path = Path(stream.name)
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())

        temporary_path.replace(path)
    finally:
        # Remove an abandoned checkpoint file while preserving any already-published destination.
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def file_stats(path: Path) -> dict:
    """Hash in bounded memory; checksums detect edited completed chunks.

    Args:
        path: File to read or publish.

    Returns:
        Byte count and SHA-256 checksum of the existing file.
    """

    # Use the same content hash for checkpoints, resume validation, and final artifacts.
    digest = hashlib.sha256()

    # Read raw bytes so the checksum describes the stored file, independent of text decoding.
    with path.open("rb") as stream:
        # Hash bounded blocks so integrity checks do not load a full historical corpus into
        # memory.
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)

    return {"bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def validate_csv(
    path: Path, component: str, start: date, end: date, expected_rows: int | None = None
) -> dict:
    """Stream-check schema, prices, order and boundaries without retaining records.

    Args:
        path: File to read or publish.
        component: Candlesticks, status, or conditions component being processed.
        start: Inclusive beginning of the requested date range.
        end: Exclusive end of the requested date range.
        expected_rows: Trusted record count used to reject incomplete files when provided.

    Returns:
        Validated row count, timestamp bounds, header, size, and checksum.

    Raises:
        ValueError: If records are malformed, duplicated, unordered or incomplete.
    """

    # Express both date boundaries in UTC so record checks match the historical request range.
    range_start_timestamp = datetime.combine(start, datetime.min.time(), timezone.utc)
    range_end_timestamp = datetime.combine(end, datetime.min.time(), timezone.utc)
    previous = None
    same_time_records: set[tuple[str, ...]] = set()
    first = last = None
    count = 0

    # Let the CSV parser handle quoting and line endings before validating individual fields.
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.reader(stream, strict=True)
        header = next(reader, [])
        required = OHLC_HEADER if component == "candlesticks" else STATUS_HEADER

        # Named columns must exist exactly once so positional field reads remain unambiguous.
        if not required.issubset(header) or len(header) != len(set(header)):
            raise ValueError(f"Invalid {component} CSV header")

        column_indices = {name: header.index(name) for name in required}

        # Status is ordered/requested on ts_recv when that field is present.
        # ts_event remains the event time consumed by the preparation pipeline.
        timestamp_column_index = (
            header.index("ts_recv")
            if component == "status" and "ts_recv" in header
            else column_indices["ts_event"]
        )

        # Check every record while retaining only the current timestamp group for duplicate
        # detection.
        for row in reader:
            count += 1

            # Reject truncated or shifted rows before using their timestamp and price fields.
            if len(row) != len(header):
                raise ValueError(f"Malformed {component} record {count}")

            # Attach record context to conversion failures so a damaged export can be located
            # and corrected.
            try:
                timestamp = datetime.fromisoformat(
                    row[timestamp_column_index].replace("Z", "+00:00")
                )
                event_timestamp = datetime.fromisoformat(
                    row[column_indices["ts_event"]].replace("Z", "+00:00")
                )

                # Absolute timestamps are required to compare records with UTC request boundaries.
                if timestamp.tzinfo is None or event_timestamp.tzinfo is None:
                    raise ValueError("Timestamps must include timezone")

                # Retain sub-microsecond order without allocating a pandas object
                # per row. Offsets and fractional precision may differ in valid CSVs.
                fractional_timestamp_match = re.search(r"[.,](\d+)", row[timestamp_column_index])
                fractional_timestamp_digits = (
                    fractional_timestamp_match.group(1) if fractional_timestamp_match else ""
                )

                # Reject unsupported precision rather than silently discard ordering information.
                if len(fractional_timestamp_digits) > 9:
                    raise ValueError("Timestamp precision exceeds nanoseconds")

                remaining_nanoseconds = int(fractional_timestamp_digits.ljust(9, "0")[6:9])
                order_key = (timestamp, remaining_nanoseconds)

                # Exclusive end semantics prevent the same record from entering adjacent chunks.
                if not range_start_timestamp <= timestamp < range_end_timestamp:
                    raise ValueError("Timestamp outside requested range")

                # Out-of-order records cannot safely be assembled by chronological chunk range.
                if previous is not None and order_key < previous:
                    raise ValueError("Timestamps are not chronological")

                # Duplicate tracking only needs the current timestamp group,
                # keeping memory bounded.
                if previous is None or order_key != previous:
                    same_time_records.clear()

                record = tuple(row)

                # Candles require one record per timestamp; distinct simultaneous status records
                # remain valid.
                if record in same_time_records or (
                    component == "candlesticks" and same_time_records
                ):
                    raise ValueError("Duplicate record or OHLC timestamp")

                same_time_records.add(record)

                # Only price records have OHLC geometry to validate.
                if component == "candlesticks":
                    open_price, high_price, low_price, close_price = (
                        float(row[column_indices[name]])
                        for name in ("open", "high", "low", "close")
                    )

                    # Nonfinite prices cannot produce usable normalized features.
                    if not all(
                        math.isfinite(price)
                        for price in (open_price, high_price, low_price, close_price)
                    ):
                        raise ValueError("Nonfinite OHLC price")

                    # The high and low must enclose the open and close to describe a valid
                    # candlestick.
                    if high_price < max(open_price, close_price) or low_price > min(
                        open_price, close_price
                    ):
                        raise ValueError("Invalid OHLC geometry")

            # Preserve the conversion cause while identifying which component and row failed
            # validation.
            except (ValueError, OverflowError) as error:
                raise ValueError(f"Invalid {component} record {count}: {error}") from error

            previous = order_key
            first = first or row[timestamp_column_index]
            last = row[timestamp_column_index]

    # A structurally valid file may still be truncated; compare its record count with trusted
    # evidence.
    if expected_rows is not None and count != expected_rows:
        raise ValueError(f"{component} row count {count} does not match expected {expected_rows}")

    return {
        "rows": count,
        "first_timestamp": first,
        "last_timestamp": last,
        "header": header,
        **file_stats(path),
    }


def validate_conditions(path: Path, start: date, end: date) -> dict:
    """Validate daily condition structure without treating absent dates as available.

    Args:
        path: File to read or publish.
        start: Inclusive beginning of the requested date range.
        end: Exclusive end of the requested date range.

    Returns:
        Validated record count, date bounds, size, and checksum.

    Raises:
        ValueError: If condition records are malformed, unordered, repeated, or outside the
            range.
    """

    # Decode the small metadata artifact before validating its date order and quality fields.
    records = json.loads(path.read_text(encoding="utf-8"))

    # The conditions endpoint must provide a sequence of daily records, not another JSON shape.
    if not isinstance(records, list):
        raise ValueError("Conditions must be a list of records")

    previous = None

    # Validate all quality dates because duplicate or misplaced metadata can change session
    # eligibility.
    for record in records:
        # Each daily record needs a usable date and quality condition before it can affect
        # eligibility.
        if not isinstance(record, dict) or not isinstance(record.get("condition"), str):
            raise ValueError("Condition records require date and condition")

        current = date.fromisoformat(record["date"])

        # Reject duplicate dates, disorder, and records outside the requested exclusive range.
        if not start <= current < end or (previous is not None and current <= previous):
            raise ValueError("Condition dates must be unique, chronological and within range")

        previous = current

    return {
        "rows": len(records),
        "first_timestamp": records[0]["date"] if records else None,
        "last_timestamp": records[-1]["date"] if records else None,
        **file_stats(path),
    }


def split_csv_into_ranges(
    source: Path,
    destinations: list[tuple[date, date, Path]],
    *,
    component: str,
) -> None:
    """Copy chronological source records into nonoverlapping date-range files.

    The source is never changed. Each destination is written atomically so an
    interrupted repair cannot present a half-copied range as reusable data.

    Args:
        source: Original CSV preserved during repair adoption.
        destinations: Ordered inclusive-start, exclusive-end ranges and their output paths.
        component: Candlesticks, status, or conditions component being processed.

    Raises:
        ValueError: If source records or destination boundaries cannot be partitioned safely.
    """

    # No destination ranges means there is no source data to partition.
    if not destinations:
        return

    # Check neighboring bounds before copying any records, preventing gaps or overlapping
    # adoption.
    for previous, current in zip(destinations, destinations[1:]):
        # Contiguous destinations ensure that each source record has exactly one possible range.
        if previous[1] != current[0]:
            raise ValueError("Repair ranges must be chronological and contiguous")

    temporary_paths = [
        path.with_name(path.stem + ".adopted.partial" + path.suffix) for _, _, path in destinations
    ]
    streams = []

    # Own all temporary outputs together so a failed split releases open files and incomplete
    # artifacts.
    try:
        # Create one temporary destination per range; the original source remains available for
        # retry.
        for (_, _, path), temporary_path in zip(
            destinations, temporary_paths, strict=True
        ):
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary_path.unlink(missing_ok=True)
            streams.append(temporary_path.open("x", encoding="utf-8", newline=""))

        # Scan the chronological source once instead of rereading it for every destination
        # range.
        with source.open(encoding="utf-8", newline="") as source_stream:
            reader = csv.reader(source_stream, strict=True)
            header = next(reader, [])
            required = OHLC_HEADER if component == "candlesticks" else STATUS_HEADER

            # Repair can split only files whose header identifies the expected record fields.
            if not required.issubset(header) or len(header) != len(set(header)):
                raise ValueError(f"Invalid {component} CSV header")

            clock_name = "ts_recv" if component == "status" and "ts_recv" in header else "ts_event"
            timestamp_column_index = header.index(clock_name)
            writers = [csv.writer(stream) for stream in streams]

            # Each chunk must remain independently readable, including a range containing no
            # data rows.
            for writer in writers:
                writer.writerow(header)

            range_index = 0

            # Route source rows in order so adoption preserves chronology without sorting the
            # full CSV.
            for row_number, row in enumerate(reader, start=1):
                # A damaged row must not be copied into a seemingly complete adopted range.
                if len(row) != len(header):
                    raise ValueError(
                        f"Malformed {component} repair record {row_number}"
                    )

                # Reject an unreadable timestamp before assigning the record to a date range.
                try:
                    timestamp = datetime.fromisoformat(
                        row[timestamp_column_index].replace("Z", "+00:00")
                    )

                # Include the source row number so a failed repair does not hide the malformed
                # input.
                except (ValueError, OverflowError) as error:
                    raise ValueError(
                        f"Invalid {component} repair timestamp at record "
                        f"{row_number}"
                    ) from error

                request_date = timestamp.astimezone(timezone.utc).date()

                # Advance across empty ranges until the current record belongs to the active
                # exclusive boundary.
                while (
                    range_index < len(destinations)
                    and request_date >= destinations[range_index][1]
                ):
                    range_index += 1

                # A record beyond the final destination cannot be silently dropped during repair.
                if range_index >= len(destinations):
                    raise ValueError("Repair record falls after the requested range")

                start, end, _ = destinations[range_index]

                # Reject uncovered dates instead of assigning them to the wrong destination.
                if not start <= request_date < end:
                    raise ValueError("Repair record falls outside the requested ranges")

                writers[range_index].writerow(row)

        # Flush and sync every output before publishing any adopted filename.
        for stream in streams:
            stream.flush()
            os.fsync(stream.fileno())
            stream.close()

        streams.clear()

        # Publish only finished temporary files so a partial copy cannot look like a reusable
        # chunk.
        for (_, _, destination), temporary_path in zip(
            destinations, temporary_paths, strict=True
        ):
            temporary_path.replace(destination)
    finally:
        # Close any streams still owned after an interrupted split, including partially
        # initialized ones.
        for stream in streams:
            stream.close()

        # Remove only temporary adoption files; preserve both source data and committed
        # destinations.
        for temporary_path in temporary_paths:
            temporary_path.unlink(missing_ok=True)


def assemble_files(
    paths: list[Path],
    destination: Path,
    component: str,
    start: date,
    end: date,
    expected_rows: int,
) -> dict:
    """Assemble locally; a failed assembly leaves the completed chunks intact.

    Args:
        paths: Completed chunk files in chronological order.
        destination: Final file published only after assembly passes validation.
        component: Candlesticks, status, or conditions component being processed.
        start: Inclusive beginning of the requested date range.
        end: Exclusive end of the requested date range.
        expected_rows: Trusted record count used to reject incomplete files when provided.

    Returns:
        Statistics describing the validated, atomically published final file.

    Raises:
        ValueError: If headers, record counts, timestamps, or condition dates fail validation.
    """

    # Keep assembly separate from the published file until the combined records pass validation.
    partial = destination.with_name(destination.stem + ".partial" + destination.suffix)

    # A failed assembly must clean its temporary output without removing the committed chunks.
    try:
        # Write a separate final candidate so validation failure cannot replace an existing
        # corpus file.
        with partial.open("w", encoding="utf-8", newline="") as output:
            # Condition arrays concatenate as JSON records; market records retain their CSV header.
            if component == "conditions":
                records = []

                # Concatenate ordered condition records before checking the combined date
                # sequence.
                for path in paths:
                    records.extend(json.loads(path.read_text(encoding="utf-8")))

                json.dump(records, output, indent=2)
                output.write("\n")
            else:
                writer = csv.writer(output)
                header = None

                # Visit chunks in manifest order so final chronology matches the validated range
                # partition.
                for path in paths:
                    # Read and compare each header separately so only one header is written to
                    # the final CSV.
                    with path.open(encoding="utf-8", newline="") as stream:
                        reader = csv.reader(stream, strict=True)
                        current_header = next(reader, [])

                        # Write the shared CSV header once before any assembled records.
                        if header is None:
                            header = current_header
                            writer.writerow(header)

                        # Different column orders would silently reinterpret subsequent chunk
                        # records.
                        elif header != current_header:
                            raise ValueError("Chunk CSV headers do not match")

                        writer.writerows(reader)

            output.flush()
            os.fsync(output.fileno())

        file_statistics = (
            validate_conditions(partial, start, end)
            if component == "conditions"
            else validate_csv(partial, component, start, end, expected_rows)
        )

        # Assembly must preserve the number of records committed by the completed chunks.
        if file_statistics["rows"] != expected_rows:
            raise ValueError("Assembled row count does not match committed chunks")

        partial.replace(destination)

        return file_statistics
    finally:
        partial.unlink(missing_ok=True)
