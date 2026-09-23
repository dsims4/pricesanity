"""Load a provably developmental annotation prefix without opening sealed target rows."""

from contextlib import closing
from pathlib import Path
import json
import sqlite3

import pandas as pd
import pyarrow.parquet as parquet

from pricesanity.annotation.schema import CandlestickAnnotation, MarketRegime
from pricesanity.benchmark.artifacts import canonical_sha256
from pricesanity.benchmark.protocol import BenchmarkConfig
from pricesanity.benchmark.snapshot import load_benchmark_snapshot
from pricesanity.config import AppConfig
from pricesanity.data.annotation_evidence import (
    EVIDENCE_ATTRIBUTE,
    validate_annotation_evidence,
)
from pricesanity.data.identifiers import build_candlestick_id
from pricesanity.training.dataset import build_complete_annotated_sessions


def load_development_sessions(
    normalized_path: Path,
    candlestick_path: Path,
    database_path: Path,
    *,
    app_config: AppConfig,
    benchmark_config: BenchmarkConfig,
    benchmark_snapshot: Path | None = None,
) -> tuple[tuple[pd.DataFrame, ...], dict]:
    """Filter market rows and SQL queries before labels outside development can be read.

    A complete prepared session catalog establishes the eventual chronological partition.
    A known official snapshot additionally constrains the allowed dates; its sealed file is
    never opened. Cropped sources require that explicit partition evidence.
    """

    timestamp_column = app_config.data.timestamp_column
    metadata = parquet.read_metadata(normalized_path).metadata or {}
    attributes = json.loads(metadata.get(b"PANDAS_ATTRS", b"{}"))
    evidence = attributes.get(EVIDENCE_ATTRIBUTE, {})
    records = evidence.get("sessions", [])
    if not records:
        raise ValueError("Prepared session validation metadata is required; rerun prepare.")
    opens = pd.DatetimeIndex(pd.to_datetime(
        [record["session_open"] for record in records], utc=True,
    ))
    dates = opens.tz_convert(app_config.data.session_timezone).date
    if not opens.is_monotonic_increasing or len(set(dates)) != len(dates):
        raise ValueError("Prepared session catalog must be uniquely chronological.")

    # The full catalog is metadata, not a read of future feature/target rows. Without it or
    # an official partition, an arbitrary 500-day late-history export is not proven development.
    allowed_dates = None
    partition_identities = []
    known_snapshots = (
        [benchmark_snapshot] if benchmark_snapshot is not None
        else sorted({path.parent for path in benchmark_config.output_root.glob(
            "**/benchmark_snapshot.json"
        )})
    )
    for directory in known_snapshots:
        partition = json.loads((directory / "benchmark_snapshot.json").read_text())
        if partition.get("format_version") != 2:
            raise ValueError(
                "Official partition evidence must physically separate the sealed holdout."
            )
        snapshot = load_benchmark_snapshot(directory)
        if snapshot.development_session_count != benchmark_config.development_session_count:
            raise ValueError("Official snapshot does not match the development partition.")
        if snapshot.session_count != benchmark_config.expected_session_count:
            raise ValueError("Official snapshot does not match the benchmark corpus count.")
        # Only the physically separate development file is allowed here, including when a
        # sealed_holdout.parquet sits beside it. Never fall back to a mixed legacy snapshot.
        development_dates = set(pd.to_datetime(pd.read_parquet(
            snapshot.data_path, columns=["session_date"],
        )["session_date"]).dt.date)
        allowed_dates = (
            development_dates if allowed_dates is None else allowed_dates & development_dates
        )
        partition_identities.append(snapshot.identity_sha256)

    if len(records) == benchmark_config.expected_session_count:
        prefix_dates = set(dates[:benchmark_config.development_session_count])
        allowed_dates = prefix_dates if allowed_dates is None else allowed_dates & prefix_dates
    elif allowed_dates is None:
        raise ValueError(
            "Cannot prove the final-holdout boundary from a cropped prepared catalog. "
            "Use the full prepared corpus or supply --benchmark-snapshot."
        )
    selected_records = [record for record, day in zip(records, dates) if day in allowed_dates]
    if not selected_records:
        raise ValueError("No prepared sessions belong to the proven development prefix.")
    last_close = pd.Timestamp(selected_records[-1]["session_close"])

    # Predicate pushdown keeps final feature and OHLC rows out of the returned tables.
    # The same evidence validator still checks every retained grid and preceding close.
    frames = []
    for path in (normalized_path, candlestick_path):
        frame = pd.read_parquet(path, filters=[(timestamp_column, "<", last_close)])
        frame_dates = frame[timestamp_column].dt.tz_convert(
            app_config.data.session_timezone
        ).dt.date
        frame = frame.loc[frame_dates.isin(allowed_dates)].reset_index(drop=True)
        original_evidence = frame.attrs.get(EVIDENCE_ATTRIBUTE, {})
        retained = [
            record for record in original_evidence.get("sessions", [])
            if pd.Timestamp(record["session_open"]).tz_convert(
                app_config.data.session_timezone
            ).date() in allowed_dates
        ]
        frame.attrs[EVIDENCE_ATTRIBUTE] = {**original_evidence, "sessions": retained}
        frames.append(frame)
    normalized, ohlc = frames
    if not normalized[timestamp_column].equals(ohlc[timestamp_column]):
        raise ValueError("Development OHLC and normalized timestamps must match exactly.")
    validate_annotation_evidence(ohlc, normalized, config=app_config)

    candle_ids = [build_candlestick_id(
        app_config.data.instrument, timestamp, app_config.data.target_interval,
    ) for timestamp in normalized[timestamp_column]]
    annotations = []
    # A read-only connection neither initializes schema nor reads unrelated annotation labels.
    # One transaction holds a consistent view even if annotation continues in another process.
    with closing(sqlite3.connect(database_path.resolve().as_uri() + "?mode=ro", uri=True)) as db:
        db.execute("BEGIN")
        for start in range(0, len(candle_ids), 400):
            selected_ids = candle_ids[start:start + 400]
            placeholders = ",".join("?" for _ in selected_ids)
            rows = db.execute(
                "SELECT candlestick_id, current_regime, anticipated_regime FROM annotations "
                f"WHERE candlestick_id IN ({placeholders}) ORDER BY candlestick_id",
                selected_ids,
            )
            annotations.extend(CandlestickAnnotation(
                candle_id, MarketRegime(current), MarketRegime(anticipated),
            ) for candle_id, current, anticipated in rows)
    sessions = build_complete_annotated_sessions(
        normalized, annotations, timestamp_column=timestamp_column,
        interval=app_config.data.target_interval,
        session_timezone=app_config.data.session_timezone,
    )
    sources = {
        "kind": "development_data_sufficiency",
        "normalized_path": str(normalized_path.resolve()),
        "candlestick_path": str(candlestick_path.resolve()),
        "annotation_database_path": str(database_path.resolve()),
        "partition_snapshot_identities": partition_identities,
        "last_allowed_session": str(max(allowed_dates)),
        "development_session_limit": benchmark_config.development_session_count,
        "prepared_catalog_sha256": canonical_sha256([str(day) for day in dates]),
        "annotation_rows_sha256": canonical_sha256(sorted([
            (value.candlestick_id, value.current_regime.value, value.anticipated_regime.value)
            for value in annotations
        ])),
        "eligibility_evidence_sha256": canonical_sha256(normalized.attrs),
        # Hash only the loaded development content; future rows are not scientific inputs.
        "normalized_development_sha256": canonical_sha256(
            pd.util.hash_pandas_object(normalized, index=False).tolist()
        ),
        "ohlc_development_sha256": canonical_sha256(
            pd.util.hash_pandas_object(ohlc, index=False).tolist()
        ),
    }
    return tuple(sessions), sources
