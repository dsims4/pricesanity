"""Write and verify self-describing benchmark runs without silent replacement."""

from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
from typing import Any, Literal

import numpy as np
import pandas as pd

from pricesanity.benchmark.metrics import (
    BenchmarkMetrics,
    EfficiencyMetrics,
    evaluate_benchmark_predictions,
)


BENCHMARK_PREDICTION_COLUMNS = (
    "candlestick_id",
    "timestamp",
    "session_date",
    "session_index",
    "candle_position",
    "uncertainty_kind",
    "predicted_current_regime",
    "predicted_anticipated_regime",
    "current_probability_bull",
    "current_probability_bear",
    "current_probability_range",
    "anticipated_probability_bull",
    "anticipated_probability_bear",
    "anticipated_probability_range",
    "current_score_bull",
    "current_score_bear",
    "current_score_range",
    "anticipated_score_bull",
    "anticipated_score_bear",
    "anticipated_score_range",
    "human_current_regime",
    "human_anticipated_regime",
)


@dataclass(frozen=True)
class BenchmarkRunIdentity:
    """Fields that make two experiment runs meaningfully different."""

    track: str
    model_name: str
    run_name: str
    seed: int
    feature_representation: str
    window_length: int
    feature_columns: tuple[str, ...]
    training_session_range: tuple[int, int]
    validation_session_ranges: tuple[tuple[int, int], ...]
    test_session_range: tuple[int, int] | None
    model_configuration_sha256: str
    representation_sha256: str
    training_session_ids_sha256: str
    validation_session_ids_sha256: str
    test_session_ids_sha256: str
    annotation_snapshot_sha256: str
    normalized_dataset_sha256: str
    protocol_sha256: str
    search_space_sha256: str
    label_mapping_sha256: str

    def __post_init__(self) -> None:
        """Require every leakage-sensitive input to participate in run identity."""

        hash_fields = (
            self.model_configuration_sha256,
            self.representation_sha256,
            self.training_session_ids_sha256,
            self.validation_session_ids_sha256,
            self.test_session_ids_sha256,
            self.annotation_snapshot_sha256,
            self.normalized_dataset_sha256,
            self.protocol_sha256,
            self.search_space_sha256,
            self.label_mapping_sha256,
        )
        if any(len(value) != 64 for value in hash_fields):
            raise ValueError("Benchmark identity hashes must be complete SHA-256 values.")

    def to_dict(self) -> dict[str, Any]:
        """Use one stable mapping for equality and persisted identity."""

        return asdict(self)

    @property
    def experiment_sha256(self) -> str:
        """Hash the complete canonical identity for directory naming and comparison."""

        return canonical_sha256(self.to_dict())


def build_benchmark_run_identity(
    *,
    track: str,
    model_name: str,
    run_name: str,
    seed: int,
    model_configuration: dict[str, Any],
    representation: dict[str, Any],
    feature_columns: tuple[str, ...],
    window_length: int,
    training_session_ids: list[str] | tuple[str, ...],
    validation_session_ids: list[str] | tuple[str, ...],
    test_session_ids: list[str] | tuple[str, ...],
    annotation_snapshot_identity: Any,
    normalized_dataset_identity: Any,
    protocol_configuration: dict[str, Any],
    search_space: dict[str, Any],
    label_mapping: dict[str, Any],
    training_session_range: tuple[int, int],
    validation_session_ranges: tuple[tuple[int, int], ...],
    test_session_range: tuple[int, int] | None,
) -> BenchmarkRunIdentity:
    """Bind every scientific input to one canonical resumable experiment identity."""

    return BenchmarkRunIdentity(
        track=track,
        model_name=model_name,
        run_name=run_name,
        seed=seed,
        feature_representation=str(representation.get("name", "unnamed")),
        window_length=window_length,
        feature_columns=feature_columns,
        training_session_range=training_session_range,
        validation_session_ranges=validation_session_ranges,
        test_session_range=test_session_range,
        model_configuration_sha256=canonical_sha256(model_configuration),
        representation_sha256=canonical_sha256(representation),
        training_session_ids_sha256=canonical_sha256(training_session_ids),
        validation_session_ids_sha256=canonical_sha256(validation_session_ids),
        test_session_ids_sha256=canonical_sha256(test_session_ids),
        annotation_snapshot_sha256=canonical_sha256(annotation_snapshot_identity),
        normalized_dataset_sha256=canonical_sha256(normalized_dataset_identity),
        protocol_sha256=canonical_sha256(protocol_configuration),
        search_space_sha256=canonical_sha256(search_space),
        label_mapping_sha256=canonical_sha256(label_mapping),
    )


@dataclass(frozen=True)
class BenchmarkRunPaths:
    """Predictable paths inside one isolated experiment directory."""

    directory: Path
    identity: Path
    progress: Path
    metadata: Path
    model: Path
    predictions: Path
    metrics: Path
    lock: Path
    resume_environment: Path


def benchmark_run_paths(
    output_root: str | Path,
    identity: BenchmarkRunIdentity,
) -> BenchmarkRunPaths:
    """Place each model beneath its track without inventing a global run counter."""

    directory = (
        Path(output_root)
        / identity.track
        / identity.model_name
        / f"{identity.run_name}_{identity.experiment_sha256[:12]}"
    )
    return BenchmarkRunPaths(
        directory=directory,
        identity=directory / "run_identity.json",
        progress=directory / "run_progress.json",
        metadata=directory / "benchmark_metadata.json",
        model=directory / "model.bin",
        predictions=directory / "predictions.parquet",
        metrics=directory / "metrics.json",
        lock=directory / ".run.lock",
        resume_environment=directory / "resume_environment.json",
    )


def prepare_benchmark_run(
    output_root: str | Path,
    identity: BenchmarkRunIdentity,
    *,
    resume: bool = False,
    execution_environment: dict[str, Any] | None = None,
) -> tuple[BenchmarkRunPaths, Literal["new", "resume", "complete"]]:
    """Reserve a run or verify that an existing directory is the same experiment."""

    paths = benchmark_run_paths(output_root, identity)
    execution_environment = execution_environment or resume_environment_fingerprint()
    if not paths.directory.exists():
        paths.directory.mkdir(parents=True)
        _write_json_atomic(paths.identity, identity.to_dict(), exclusive=True)
        _write_json_atomic(
            paths.resume_environment, execution_environment, exclusive=True
        )
        return paths, "new"
    if not resume:
        raise FileExistsError(
            f"Benchmark run already exists and will not be overwritten: {paths.directory}"
        )
    stored_identity = _read_json(paths.identity, "benchmark run identity")
    if stored_identity != _json_compatible(identity.to_dict()):
        raise ValueError("Existing benchmark directory belongs to a different run identity.")
    if paths.metadata.is_file():
        load_benchmark_run(paths.directory)
        return paths, "complete"
    stored_environment = _read_json(
        paths.resume_environment, "benchmark resume environment"
    )
    if stored_environment != _json_compatible(execution_environment):
        raise ValueError(
            "Interrupted benchmark run was created in an incompatible execution environment."
        )
    return paths, "resume"


def save_benchmark_run(
    paths: BenchmarkRunPaths,
    identity: BenchmarkRunIdentity,
    *,
    model: Any,
    predictions: pd.DataFrame,
    metrics: BenchmarkMetrics,
    model_configuration: dict[str, Any],
    dataset_description: dict[str, Any],
) -> dict[str, Any]:
    """Publish model, predictions, metrics, and final checksums exactly once."""

    with _run_lock(paths.lock):
        return _save_benchmark_run_locked(
            paths,
            identity,
            model=model,
            predictions=predictions,
            metrics=metrics,
            model_configuration=model_configuration,
            dataset_description=dataset_description,
        )


def checkpoint_fitted_model(paths: BenchmarkRunPaths, model: Any, timing: dict) -> None:
    """Publish model and original fitting evidence before inference can fail."""

    with _run_lock(paths.lock):
        if paths.progress.exists() or paths.metadata.exists():
            raise ValueError("A fitted stage is already checkpointed.")
        _remove_stale_partial(paths.model)
        temporary = _temporary_path(paths.model)
        model.save(temporary)
        _validate_nonempty_file(temporary, "model")
        _publish_atomic(temporary, paths.model)
        _write_json_atomic(paths.progress, {
            "format_version": 1,
            "completed": {"model": file_sha256(paths.model), "model_state_sha256": model.state_fingerprint()},
            "training_evidence": timing,
        })


def _save_benchmark_run_locked(
    paths: BenchmarkRunPaths,
    identity: BenchmarkRunIdentity,
    *,
    model: Any,
    predictions: pd.DataFrame,
    metrics: BenchmarkMetrics,
    model_configuration: dict[str, Any],
    dataset_description: dict[str, Any],
) -> dict[str, Any]:
    """Publish one run while its advisory writer lock is held."""

    if paths.metadata.exists():
        raise FileExistsError("Completed benchmark runs cannot be silently overwritten.")
    stored_identity = _read_json(paths.identity, "benchmark run identity")
    if stored_identity != _json_compatible(identity.to_dict()):
        raise ValueError("Benchmark identity changed after its directory was reserved.")
    if canonical_sha256(model_configuration) != identity.model_configuration_sha256:
        raise ValueError("Model configuration does not match the reserved run identity.")
    validated_predictions = validate_benchmark_predictions(predictions)
    model_state_sha256 = model.state_fingerprint()

    progress = (
        _read_json(paths.progress, "benchmark progress")
        if paths.progress.is_file()
        else {"format_version": 1, "completed": {}}
    )
    completed = progress.get("completed")
    if progress.get("format_version") != 1 or not isinstance(completed, dict):
        raise ValueError("Benchmark progress checkpoint is malformed.")

    # Publish and checkpoint each component independently. On resume, only a component whose
    # recorded checksum still matches can be reused; an uncheckpointed file is never trusted.
    if not _component_is_complete(paths.model, completed.get("model")):
        _remove_stale_partial(paths.model)
        temporary_model = _temporary_path(paths.model)
        model.save(temporary_model)
        _validate_nonempty_file(temporary_model, "model")
        _publish_atomic(temporary_model, paths.model)
        completed["model"] = file_sha256(paths.model)
        completed["model_state_sha256"] = model_state_sha256
        progress["training_evidence"] = asdict(metrics.efficiency) if metrics.efficiency else None
        _write_json_atomic(paths.progress, progress)
    elif completed.get("model_state_sha256") != model_state_sha256:
        # A new stochastic instance is not equivalent merely because it has the same settings.
        # Downstream output must come from the exact checkpoint that survived the crash.
        raise ValueError("Resumed model instance does not match the checkpointed fitted model.")
    if not _component_is_complete(paths.predictions, completed.get("predictions")):
        _remove_stale_partial(paths.predictions)
        temporary_predictions = _temporary_path(paths.predictions)
        validated_predictions.to_parquet(temporary_predictions, index=False)
        validate_benchmark_predictions(pd.read_parquet(temporary_predictions))
        _publish_atomic(temporary_predictions, paths.predictions)
        completed["predictions"] = file_sha256(paths.predictions)
        completed["predictions_source_model_state_sha256"] = model_state_sha256
        progress["efficiency"] = asdict(metrics.efficiency) if metrics.efficiency else None
        _write_json_atomic(paths.progress, progress)
    elif completed.get("predictions_source_model_state_sha256") != model_state_sha256:
        raise ValueError("Checkpointed predictions were not produced by this fitted model.")
    # Scientific metrics are derived from the exact persisted prediction rows. The caller's
    # efficiency evidence is retained, but cannot substitute results from another model pass.
    persisted_predictions = validate_benchmark_predictions(
        pd.read_parquet(paths.predictions)
    )
    persisted_metrics = _metrics_from_prediction_frame(
        persisted_predictions,
        efficiency=(
            replace(
                metrics.efficiency,
                serialized_model_bytes=paths.model.stat().st_size,
            )
            if metrics.efficiency is not None
            else None
        ),
    )
    if not _component_is_complete(paths.metrics, completed.get("metrics")):
        _remove_stale_partial(paths.metrics)
        _write_json_atomic(paths.metrics, persisted_metrics.to_dict())
        completed["metrics"] = file_sha256(paths.metrics)
        completed["metrics_source_predictions_sha256"] = completed["predictions"]
        _write_json_atomic(paths.progress, progress)
    elif completed.get("metrics_source_predictions_sha256") != completed["predictions"]:
        raise ValueError("Checkpointed metrics do not belong to persisted predictions.")
    if not (
        completed.get("model_state_sha256")
        == completed.get("predictions_source_model_state_sha256")
        and completed.get("predictions")
        == completed.get("metrics_source_predictions_sha256")
    ):
        raise ValueError("Benchmark components do not form one coherent fitted-model chain.")
    return publish_checkpointed_run(paths, identity, model_configuration, dataset_description)


def publish_checkpointed_run(paths, identity, model_configuration, dataset_description):
    """Finish a metrics-complete run by verifying its chain and publishing metadata only."""
    if _read_json(paths.identity, "run identity") != _json_compatible(identity.to_dict()):
        raise ValueError("Checkpoint identity changed.")
    if canonical_sha256(model_configuration) != identity.model_configuration_sha256:
        raise ValueError("Checkpoint model configuration changed.")
    progress = _read_json(paths.progress, "run progress")
    completed = progress.get("completed", {})
    for name in ("model", "predictions", "metrics"):
        if not _component_is_complete(getattr(paths, name), completed.get(name)):
            raise ValueError("Checkpoint component is incomplete: " + name)
    if (completed.get("model_state_sha256") != completed.get("predictions_source_model_state_sha256")
            or completed.get("predictions") != completed.get("metrics_source_predictions_sha256")):
        raise ValueError("Checkpoint components are not a coherent chain.")
    metadata = {
        "format_version": 1,
        "state": "complete",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "identity": identity.to_dict(),
        "model_configuration": model_configuration,
        "dataset": dataset_description,
        "artifacts": {
            "model": paths.model.name,
            "model_sha256": file_sha256(paths.model),
            "predictions": paths.predictions.name,
            "predictions_sha256": file_sha256(paths.predictions),
            "metrics": paths.metrics.name,
            "metrics_sha256": file_sha256(paths.metrics),
        },
        "library_versions": _library_versions(),
        "git": source_identity(),
        "resume_environment": _read_json(
            paths.resume_environment, "benchmark resume environment"
        ),
    }
    # Metadata is the commit marker. Readers cannot mistake an interrupted collection of
    # components for a complete run because this file is published only after all validation.
    _write_json_atomic(paths.metadata, metadata, exclusive=True)
    return metadata


def load_benchmark_summary(
    run_directory: str | Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load verified metadata and metrics without reading the prediction Parquet."""

    run_directory = Path(run_directory)
    metadata = _validated_metadata(run_directory)
    artifacts = metadata["artifacts"]
    metrics_path = run_directory / artifacts["metrics"]
    if file_sha256(metrics_path) != artifacts["metrics_sha256"]:
        raise ValueError("Benchmark artifact checksum mismatch: metrics")
    return metadata, _read_json(metrics_path, "benchmark metrics")


def load_benchmark_run(
    run_directory: str | Path,
) -> tuple[dict[str, Any], pd.DataFrame, dict[str, Any]]:
    """Verify a complete run before any GUI or report displays its results."""

    run_directory = Path(run_directory)
    metadata = _validated_metadata(run_directory)
    artifacts = metadata["artifacts"]
    for artifact_name in ("model", "predictions", "metrics"):
        relative_path = Path(str(artifacts.get(artifact_name, "")))
        if relative_path.is_absolute() or relative_path.name != str(relative_path):
            raise ValueError("Benchmark artifact paths must remain inside their run.")
        artifact_path = run_directory / relative_path
        if file_sha256(artifact_path) != artifacts.get(f"{artifact_name}_sha256"):
            raise ValueError(f"Benchmark artifact checksum mismatch: {artifact_name}")

    try:
        predictions = pd.read_parquet(run_directory / artifacts["predictions"])
    except OSError as error:
        raise ValueError("Could not read benchmark predictions.") from error
    predictions = validate_benchmark_predictions(predictions)
    metrics = _read_json(run_directory / artifacts["metrics"], "benchmark metrics")
    return metadata, predictions, metrics


def audit_benchmark_run(
    run_directory: str | Path,
) -> tuple[dict[str, Any], pd.DataFrame, dict[str, Any]]:
    """Fully verify checksums and recompute persisted scientific metrics."""

    metadata, predictions, metrics = load_benchmark_run(run_directory)
    recomputed = _metrics_from_prediction_frame(predictions).to_dict()
    saved_scientific_metrics = dict(metrics)
    saved_scientific_metrics.pop("efficiency", None)
    recomputed.pop("efficiency", None)
    # Additive diagnostics do not make an older completed result unreadable. Audit every
    # field it actually recorded; new runs always include both transition anchors.
    for added in ("anticipated_transitions", "anticipated_transition_neighborhoods"):
        if added not in saved_scientific_metrics:
            recomputed.pop(added, None)
    if canonical_sha256(saved_scientific_metrics) != canonical_sha256(recomputed):
        raise ValueError("Saved benchmark metrics do not match recomputed predictions.")
    return metadata, predictions, metrics


def _metrics_from_prediction_frame(
    predictions: pd.DataFrame,
    *,
    efficiency: EfficiencyMetrics | None = None,
) -> BenchmarkMetrics:
    """Recompute scientific results from the persisted source-of-truth rows."""

    label_indices = {"bull": 0, "bear": 1, "range": 2}
    decoded = {}
    for name in (
        "human_current", "predicted_current",
        "human_anticipated", "predicted_anticipated",
    ):
        decoded[name] = predictions[f"{name}_regime"].map(label_indices).to_numpy()
    uses_probabilities = predictions["uncertainty_kind"].iloc[0] != (
        "uncalibrated_decision_score"
    )
    current_probabilities = None
    anticipated_probabilities = None
    if uses_probabilities:
        current_probabilities = predictions[[
            "current_probability_bull", "current_probability_bear",
            "current_probability_range",
        ]].to_numpy()
        anticipated_probabilities = predictions[[
            "anticipated_probability_bull", "anticipated_probability_bear",
            "anticipated_probability_range",
        ]].to_numpy()
    return evaluate_benchmark_predictions(
        human_current=decoded["human_current"],
        predicted_current=decoded["predicted_current"],
        human_anticipated=decoded["human_anticipated"],
        predicted_anticipated=decoded["predicted_anticipated"],
        current_probabilities=current_probabilities,
        anticipated_probabilities=anticipated_probabilities,
        session_indices=predictions["session_index"].to_numpy(),
        efficiency=efficiency,
    )


def _validated_metadata(run_directory: Path) -> dict[str, Any]:
    """Validate the small run manifest without loading heavyweight artifacts."""

    metadata = _read_json(
        run_directory / "benchmark_metadata.json", "benchmark metadata"
    )
    if metadata.get("format_version") != 1 or metadata.get("state") != "complete":
        raise ValueError("Benchmark run is not a supported completed artifact.")
    identity = _read_json(run_directory / "run_identity.json", "benchmark identity")
    if metadata.get("identity") != identity:
        raise ValueError("Benchmark metadata does not match its reserved identity.")
    artifacts = metadata.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("Benchmark metadata is missing artifact identities.")
    for artifact_name in ("model", "predictions", "metrics"):
        relative_path = Path(str(artifacts.get(artifact_name, "")))
        if relative_path.is_absolute() or relative_path.name != str(relative_path):
            raise ValueError("Benchmark artifact paths must remain inside their run.")
    return metadata


def validate_benchmark_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    """Require exact identities, labels, probabilities, and chronological ordering."""

    missing_columns = set(BENCHMARK_PREDICTION_COLUMNS).difference(predictions.columns)
    if missing_columns:
        raise ValueError(
            "Benchmark predictions are missing columns: "
            + ", ".join(sorted(missing_columns))
        )
    predictions = predictions.loc[:, BENCHMARK_PREDICTION_COLUMNS].copy()
    if predictions.empty:
        raise ValueError("Benchmark predictions cannot be empty.")
    predictions["timestamp"] = pd.to_datetime(
        predictions["timestamp"], utc=True, errors="raise"
    )
    predictions["session_date"] = pd.to_datetime(
        predictions["session_date"], errors="raise"
    ).dt.date
    if (
        predictions["candlestick_id"].astype(str).str.strip().eq("").any()
        or predictions["candlestick_id"].duplicated().any()
        or predictions["timestamp"].duplicated().any()
        or not predictions["timestamp"].is_monotonic_increasing
    ):
        raise ValueError("Benchmark predictions must be uniquely chronological.")

    valid_regimes = {"bull", "bear", "range"}
    for regime_column in (
        "predicted_current_regime",
        "predicted_anticipated_regime",
        "human_current_regime",
        "human_anticipated_regime",
    ):
        if not set(predictions[regime_column]).issubset(valid_regimes):
            raise ValueError("Benchmark predictions contain an unknown regime label.")

    uncertainty_kinds = set(predictions["uncertainty_kind"])
    if len(uncertainty_kinds) != 1:
        raise ValueError("One benchmark run must use one uncertainty-value contract.")
    uncertainty_kind = next(iter(uncertainty_kinds))
    uses_probabilities = uncertainty_kind in {
        "probability_estimate",
        "softmax_probability_estimate",
        "deterministic_distribution",
    }
    uses_scores = uncertainty_kind == "uncalibrated_decision_score"
    if not uses_probabilities and not uses_scores:
        raise ValueError("Benchmark predictions name an unknown uncertainty contract.")

    for prefix in ("current", "anticipated"):
        probability_columns = [
            f"{prefix}_probability_bull",
            f"{prefix}_probability_bear",
            f"{prefix}_probability_range",
        ]
        probabilities = predictions[probability_columns].to_numpy(dtype=float)
        score_columns = [
            f"{prefix}_score_bull",
            f"{prefix}_score_bear",
            f"{prefix}_score_range",
        ]
        scores = predictions[score_columns].to_numpy(dtype=float)
        if uses_probabilities:
            if (
                not np.isfinite(probabilities).all()
                or (probabilities < 0.0).any()
                or (probabilities > 1.0).any()
                or not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6)
                or not np.isnan(scores).all()
            ):
                raise ValueError("Benchmark probability estimates are invalid.")
        elif not np.isnan(probabilities).all() or not np.isfinite(scores).all():
            raise ValueError("Benchmark decision scores are invalid.")

        # Native model classes are authoritative. SVC can legitimately disagree with the
        # ordering of auxiliary probability estimates, so integrity means preserving both
        # values honestly rather than forcing an argmax equality that the model never promised.
    return predictions


def file_sha256(path: str | Path) -> str:
    """Hash one completed artifact without loading it into memory."""

    with Path(path).open("rb") as artifact_file:
        return hashlib.file_digest(artifact_file, "sha256").hexdigest()


def canonical_sha256(value: Any) -> str:
    """Hash JSON-compatible experiment data with stable key ordering."""

    encoded = json.dumps(
        _json_compatible(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def resume_environment_fingerprint(
    *,
    device: str = "unknown",
) -> dict[str, Any]:
    """Describe execution compatibility separately from scientific identity."""

    return {**_library_versions(), "device": device, "source": source_identity()}


def source_identity() -> dict[str, Any]:
    """Bind partial execution to installed source bytes, including uncommitted edits."""

    source_root = Path(__file__).resolve().parents[1]
    files = {
        str(path.relative_to(source_root)): file_sha256(path)
        for path in sorted(source_root.rglob("*.py"))
    }
    # Scientific identity describes the question; this stricter identity describes the code
    # that produced an unfinished checkpoint. Dirty Git state alone cannot identify its bytes.
    return {**_git_identity(source_root), "source_sha256": canonical_sha256(files)}


def _write_json_atomic(
    path: Path,
    values: dict[str, Any],
    *,
    exclusive: bool = False,
) -> None:
    """Replace only the small progress checkpoint after durable component writes."""

    if exclusive and path.exists():
        raise FileExistsError(f"Artifact already exists: {path}")
    temporary_path = _temporary_path(path)
    with temporary_path.open("x", encoding="utf-8") as output_file:
        json.dump(_json_compatible(values), output_file, indent=2, sort_keys=True)
        output_file.write("\n")
        output_file.flush()
        os.fsync(output_file.fileno())
    _read_json(temporary_path, path.stem)
    if exclusive and path.exists():
        temporary_path.unlink()
        raise FileExistsError(f"Artifact already exists: {path}")
    _publish_atomic(temporary_path, path)


def _temporary_path(path: Path) -> Path:
    """Give each writer an isolated temporary name inside the destination directory."""

    return path.with_name(f".{path.name}.{os.getpid()}.partial")


def _publish_atomic(temporary_path: Path, final_path: Path) -> None:
    """Durably publish a validated file and then sync its directory entry."""

    with temporary_path.open("rb") as artifact_file:
        os.fsync(artifact_file.fileno())
    os.replace(temporary_path, final_path)
    directory_descriptor = os.open(final_path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_descriptor)
    finally:
        os.close(directory_descriptor)


def _remove_stale_partial(path: Path) -> None:
    """Discard only this process-neutral temporary artifact during locked resume."""

    for partial_path in path.parent.glob(f".{path.name}.*.partial"):
        partial_path.unlink()


def _validate_nonempty_file(path: Path, description: str) -> None:
    """Reject a serializer that returned without producing usable bytes."""

    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"Benchmark {description} serializer produced an empty file.")


@contextmanager
def _run_lock(lock_path: Path):
    """Allow one cooperating writer per run while other run directories remain parallel."""

    with lock_path.open("a+") as lock_file:
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another process is already writing this benchmark run.") from error
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _git_identity(run_directory: Path) -> dict[str, Any]:
    """Record revision state when the output directory belongs to a Git checkout."""

    import subprocess

    try:
        root = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=run_directory,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, check=True,
            capture_output=True, text=True,
        ).stdout.strip()
        dirty = bool(subprocess.run(
            ["git", "status", "--porcelain"], cwd=root, check=True,
            capture_output=True, text=True,
        ).stdout.strip())
        return {"revision": revision, "dirty": dirty}
    except (OSError, subprocess.SubprocessError):
        return {"revision": None, "dirty": None}


def _component_is_complete(path: Path, recorded_checksum: object) -> bool:
    """Reuse one interrupted-run component only when its checkpoint still verifies."""

    if recorded_checksum is None:
        return False
    if not isinstance(recorded_checksum, str) or not path.is_file():
        raise ValueError(f"Checkpointed benchmark component is missing: {path.name}")
    if file_sha256(path) != recorded_checksum:
        raise ValueError(f"Checkpointed benchmark component changed: {path.name}")
    return True


def _read_json(path: Path, description: str) -> dict[str, Any]:
    """Read a named mapping with an error identifying the broken artifact."""

    try:
        with path.open(encoding="utf-8") as input_file:
            values = json.load(input_file)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not read {description}: {path}") from error
    if not isinstance(values, dict):
        raise ValueError(f"{description.title()} must be a named mapping.")
    return values


def _json_compatible(value: Any) -> Any:
    """Normalize tuples and paths before identity comparison and serialization."""

    if isinstance(value, dict):
        return {str(key): _json_compatible(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_compatible(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _library_versions() -> dict[str, str]:
    """Record core versions and optional libraries actually installed at run time."""

    versions = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }
    try:
        import sklearn
        versions["scikit_learn"] = sklearn.__version__
    except ImportError:
        pass
    try:
        import torch
        versions["torch"] = torch.__version__
    except ImportError:
        pass
    return versions
