"""Freeze exact benchmark rows so later trials never reread a changing database."""

from collections.abc import Sequence
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json
import os

import numpy as np
import pandas as pd

from pricesanity.annotation.store import AnnotationStore
from pricesanity.benchmark.artifacts import canonical_sha256, file_sha256
from pricesanity.config import AppConfig
from pricesanity.data.annotation_evidence import validate_annotation_evidence, EVIDENCE_POLICY
from pricesanity.features import FEATURE_COLUMNS
from pricesanity.training.dataset import build_complete_annotated_sessions


SNAPSHOT_COLUMNS = (
    "session_index", "candle_position", "session_date", "candlestick_id", "timestamp",
    *FEATURE_COLUMNS, "current_target", "anticipated_target",
)


@dataclass(frozen=True)
class BenchmarkSnapshot:
    """Immutable rows and identities consumed by every run in one benchmark study."""

    directory: Path
    data_path: Path
    manifest_path: Path
    identity_sha256: str
    session_count: int
    row_count: int
    source_identities: dict[str, Any]
    development_session_count: int | None = None


def freeze_benchmark_snapshot(
    *,
    normalized_path: str | Path,
    annotation_database_path: str | Path,
    app_config: AppConfig,
    output_directory: str | Path,
    expected_session_count: int | None = None,
    candlestick_path: str | Path | None = None,
    development_session_count: int | None = None,
) -> BenchmarkSnapshot:
    """Join private source artifacts once and atomically publish benchmark-only rows."""

    normalized_path = Path(normalized_path)
    database_path = Path(annotation_database_path)
    normalized = pd.read_parquet(normalized_path)
    if candlestick_path is None:
        raise ValueError("Strict benchmark initialization requires the paired validated OHLC artifact.")
    ohlc = pd.read_parquet(candlestick_path)
    if not normalized[app_config.data.timestamp_column].equals(ohlc[app_config.data.timestamp_column]):
        raise ValueError("Normalized and OHLC snapshot inputs must have exactly aligned timestamps.")
    validate_annotation_evidence(ohlc, normalized, config=app_config)
    with closing(AnnotationStore(database_path)) as store:
        annotations = store.load_all()
    sessions = build_complete_annotated_sessions(
        normalized,
        annotations,
        timestamp_column=app_config.data.timestamp_column,
        interval=app_config.data.target_interval,
        session_timezone=app_config.data.session_timezone,
    )
    annotation_identity = canonical_sha256([
        {
            "candlestick_id": annotation.candlestick_id,
            "current_regime": annotation.current_regime.value,
            "anticipated_regime": annotation.anticipated_regime.value,
        }
        for annotation in sorted(annotations, key=lambda value: value.candlestick_id)
    ])
    source_identities = {
        "normalized_path": str(normalized_path.resolve()),
        "normalized_sha256": file_sha256(normalized_path),
        "annotation_database_path": str(database_path.resolve()),
        "annotation_rows_sha256": annotation_identity,
        "instrument": app_config.data.instrument,
        "target_interval": app_config.data.target_interval,
        "session_timezone": app_config.data.session_timezone,
        "eligibility_policy": EVIDENCE_POLICY,
    }
    if candlestick_path is not None:
        candlestick_path = Path(candlestick_path)
        if not candlestick_path.is_file():
            raise ValueError("Benchmark OHLC candlestick artifact does not exist.")
        source_identities.update({
            "candlestick_path": str(candlestick_path.resolve()),
            "candlestick_sha256": file_sha256(candlestick_path),
        })
    return freeze_benchmark_snapshot_from_sessions(
        sessions,
        output_directory=output_directory,
        timestamp_column=app_config.data.timestamp_column,
        expected_session_count=expected_session_count,
        source_identities=source_identities,
        development_session_count=development_session_count,
    )


def freeze_benchmark_snapshot_from_sessions(
    sessions: Sequence[pd.DataFrame],
    *,
    output_directory: str | Path,
    timestamp_column: str = "ts_event",
    expected_session_count: int | None = None,
    source_identities: dict[str, Any] | None = None,
    development_session_count: int | None = None,
) -> BenchmarkSnapshot:
    """Publish a validated synthetic or prejoined session collection for one study."""

    if expected_session_count is not None and len(sessions) != expected_session_count:
        raise ValueError(
            f"Snapshot requires {expected_session_count} sessions, but received {len(sessions)}."
        )
    rows = []
    previous_date = None
    for session_index, session in enumerate(sessions):
        if session.empty:
            raise ValueError("Benchmark snapshot sessions cannot be empty.")
        required = {
            "session_date", "candlestick_id", timestamp_column,
            *FEATURE_COLUMNS, "current_target", "anticipated_target",
        }
        missing = required.difference(session.columns)
        if missing:
            raise ValueError("Snapshot session is missing columns: " + ", ".join(sorted(missing)))
        unique_dates = session["session_date"].drop_duplicates()
        if len(unique_dates) != 1:
            raise ValueError("Each benchmark snapshot session needs one date.")
        session_date = pd.Timestamp(unique_dates.iloc[0]).date()
        if previous_date is not None and session_date <= previous_date:
            raise ValueError("Benchmark snapshot sessions must be uniquely chronological.")
        previous_date = session_date
        timestamps = pd.to_datetime(session[timestamp_column], utc=True, errors="raise")
        if timestamps.duplicated().any() or not timestamps.is_monotonic_increasing:
            raise ValueError("Snapshot candles must be uniquely chronological.")
        for candle_position, (_, row) in enumerate(session.iterrows()):
            rows.append({
                "session_index": session_index,
                "candle_position": candle_position,
                "session_date": session_date,
                "candlestick_id": str(row["candlestick_id"]),
                "timestamp": timestamps.iloc[candle_position],
                **{column: float(row[column]) for column in FEATURE_COLUMNS},
                "current_target": int(row["current_target"]),
                "anticipated_target": int(row["anticipated_target"]),
            })
    snapshot_data = validate_snapshot_data(pd.DataFrame(rows, columns=SNAPSHOT_COLUMNS))
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    isolated = development_session_count is not None
    if isolated and not 0 < development_session_count < len(sessions):
        raise ValueError("Development partition must leave nonempty development and holdout.")
    data_path = output_directory / ("development.parquet" if isolated else "benchmark_snapshot.parquet")
    manifest_path = output_directory / "benchmark_snapshot.json"
    holdout_path = output_directory / "sealed_holdout.parquet"
    if data_path.exists() or manifest_path.exists() or holdout_path.exists():
        raise FileExistsError("Benchmark snapshot already exists and is immutable.")
    temporary_data = output_directory / ".benchmark_snapshot.parquet.partial"
    development_data = (
        snapshot_data.loc[snapshot_data.session_index < development_session_count]
        if isolated else snapshot_data
    )
    development_data.to_parquet(temporary_data, index=False)
    validate_snapshot_data(pd.read_parquet(temporary_data))
    _publish(temporary_data, data_path)
    sources = dict(source_identities or {"kind": "synthetic"})
    partition = {}
    if isolated:
        temporary_holdout = output_directory / ".sealed_holdout.parquet.partial"
        holdout_data = snapshot_data.loc[snapshot_data.session_index >= development_session_count]
        holdout_data.to_parquet(temporary_holdout, index=False)
        _publish(temporary_holdout, holdout_path)
        partition = {
            "session_count": len(sessions),
            "row_count": len(snapshot_data),
            "development_session_count": development_session_count,
            "development_row_count": len(development_data),
            "holdout_file": holdout_path.name,
            "holdout_sha256": file_sha256(holdout_path),
            "holdout_row_count": len(holdout_data),
        }
    identity = canonical_sha256({
        "format_version": 2 if isolated else 1,
        "data_sha256": file_sha256(data_path),
        "source_identities": sources,
        "columns": SNAPSHOT_COLUMNS,
        **partition,
    })
    manifest = {
        "format_version": 2 if isolated else 1,
        "state": "complete",
        "identity_sha256": identity,
        "data_file": data_path.name,
        "data_sha256": file_sha256(data_path),
        "session_count": len(sessions),
        "row_count": len(snapshot_data),
        "columns": list(SNAPSHOT_COLUMNS),
        "source_identities": sources,
        **partition,
    }
    _write_json_atomic(manifest_path, manifest)
    return load_benchmark_snapshot(output_directory)


def load_benchmark_snapshot(directory: str | Path) -> BenchmarkSnapshot:
    """Verify the immutable snapshot manifest and data before returning its identity."""

    directory = Path(directory)
    manifest_path = directory / "benchmark_snapshot.json"
    with manifest_path.open(encoding="utf-8") as input_file:
        manifest = json.load(input_file)
    if manifest.get("format_version") not in (1, 2) or manifest.get("state") != "complete":
        raise ValueError("Benchmark snapshot is not complete or supported.")
    data_path = directory / str(manifest.get("data_file", ""))
    if data_path.parent != directory or file_sha256(data_path) != manifest.get("data_sha256"):
        raise ValueError("Benchmark snapshot data checksum does not match its manifest.")
    expected_identity = canonical_sha256({
        "format_version": manifest["format_version"],
        "data_sha256": manifest["data_sha256"],
        "source_identities": manifest["source_identities"],
        "columns": tuple(manifest["columns"]),
        **({key: manifest[key] for key in (
            "session_count", "row_count",
            "development_session_count", "development_row_count", "holdout_file",
            "holdout_sha256", "holdout_row_count",
        )} if manifest["format_version"] == 2 else {}),
    })
    if expected_identity != manifest.get("identity_sha256"):
        raise ValueError("Benchmark snapshot identity is inconsistent.")
    return BenchmarkSnapshot(
        directory=directory,
        data_path=data_path,
        manifest_path=manifest_path,
        identity_sha256=expected_identity,
        session_count=int(manifest["session_count"]),
        row_count=int(manifest["row_count"]),
        source_identities=dict(manifest["source_identities"]),
        development_session_count=manifest.get("development_session_count"),
    )


def load_snapshot_sessions(snapshot: BenchmarkSnapshot) -> tuple[pd.DataFrame, ...]:
    """Load and validate frozen rows, then restore their chronological session grouping."""

    data = validate_snapshot_data(pd.read_parquet(snapshot.data_path))
    manifest = json.loads(snapshot.manifest_path.read_text())
    expected_rows = manifest.get("development_row_count", snapshot.row_count)
    expected_sessions = snapshot.development_session_count or snapshot.session_count
    if len(data) != expected_rows or data["session_index"].nunique() != expected_sessions:
        raise ValueError("Benchmark snapshot counts changed after publication.")
    return tuple(
        session.rename(columns={"timestamp": "ts_event"}).reset_index(drop=True)
        for _, session in data.groupby("session_index", sort=True)
    )


def load_sealed_holdout(
    snapshot: BenchmarkSnapshot, *, seal_path: Path, confirm_final_holdout: bool,
) -> tuple[pd.DataFrame, ...]:
    """Open holdout bytes only after the executor verifies the global development seal."""

    if not confirm_final_holdout or not seal_path.is_file():
        raise PermissionError("Final holdout is locked until global development freeze and confirmation.")
    seal = json.loads(seal_path.read_text())
    # A per-model winner also has state="frozen" and a snapshot hash. It is not a global
    # development seal, even when a caller passes it to this lower-level API by mistake.
    scope = seal.get("scope", {})
    models = scope.get("models", [])
    tracks = scope.get("tracks", [])
    if (seal_path.name != "development_frozen.json" or not models or not tracks
            or any(Path(name).name != name for name in (*models, *tracks))):
        raise PermissionError("Final holdout requires the global development seal, not a model winner.")
    expected = {f"{track}/{model}" for track in tracks for model in models}
    if set(seal.get("selected", {})) != expected:
        raise ValueError("Development seal does not bind every declared model and track.")
    for key in expected:
        selected_path = seal_path.parent / "selected" / f"{key}.json"
        selected = json.loads(selected_path.read_text())
        if (canonical_sha256(selected) != seal["selected"][key]["configuration_sha256"]
                or selected.get("state") != "frozen"
                or not selected.get("all_development_folds")
                or selected.get("fold_count", 0) < 2
                or selected.get("protocol_sha256") != seal.get("protocol_sha256")):
            raise ValueError("Development seal contains an incompatible or incomplete selection.")
    if seal.get("snapshot_sha256") != snapshot.identity_sha256 or seal.get("state") != "frozen":
        raise ValueError("Development seal belongs to another snapshot or is not frozen.")
    manifest = json.loads(snapshot.manifest_path.read_text())
    if manifest.get("format_version") != 2:
        raise ValueError("Final evaluation requires a physically isolated snapshot.")
    path = snapshot.directory / manifest["holdout_file"]
    if path.parent != snapshot.directory or file_sha256(path) != manifest["holdout_sha256"]:
        raise ValueError("Sealed holdout checksum does not match its immutable snapshot.")
    data = validate_snapshot_data(pd.read_parquet(path))
    if len(data) != manifest["holdout_row_count"]:
        raise ValueError("Sealed holdout row count changed.")
    if data["session_index"].drop_duplicates().tolist() != list(range(snapshot.development_session_count, snapshot.session_count)):
        raise ValueError("Sealed holdout session accounting changed.")
    return tuple(
        session.rename(columns={"timestamp": "ts_event"}).reset_index(drop=True)
        for _, session in data.groupby("session_index", sort=True)
    )


def validate_snapshot_data(data: pd.DataFrame) -> pd.DataFrame:
    """Require complete identities, finite features, labels, and chronological ordering."""

    missing = set(SNAPSHOT_COLUMNS).difference(data.columns)
    if missing or data.empty:
        raise ValueError("Benchmark snapshot data is missing required nonempty rows.")
    data = data.loc[:, SNAPSHOT_COLUMNS].copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True, errors="raise")
    if data["candlestick_id"].duplicated().any() or not data["timestamp"].is_monotonic_increasing:
        raise ValueError("Benchmark snapshot identities must be uniquely chronological.")
    if not np.isfinite(data.loc[:, FEATURE_COLUMNS].to_numpy(dtype=float)).all():
        raise ValueError("Benchmark snapshot features must be finite.")
    for target in ("current_target", "anticipated_target"):
        values = pd.to_numeric(data[target], errors="raise").to_numpy(dtype=np.int64)
        if not np.isin(values, (0, 1, 2)).all():
            raise ValueError("Benchmark snapshot contains an unknown regime target.")
        data[target] = values
    return data


def _publish(temporary: Path, final: Path) -> None:
    with temporary.open("rb") as input_file:
        os.fsync(input_file.fileno())
    os.replace(temporary, final)


def _write_json_atomic(path: Path, values: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.partial")
    with temporary.open("x", encoding="utf-8") as output_file:
        json.dump(values, output_file, indent=2, sort_keys=True)
        output_file.write("\n")
        output_file.flush()
        os.fsync(output_file.fileno())
    os.replace(temporary, path)
