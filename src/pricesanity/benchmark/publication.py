"""Publish deterministic, allowlisted benchmark summaries for source control."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
import csv
import json
import math
from pathlib import Path, PureWindowsPath
import statistics
import tempfile
from typing import Any

from pricesanity.benchmark.artifacts import canonical_sha256, file_sha256, load_benchmark_summary
from pricesanity.benchmark.registry import MODEL_CONCEPTUAL_PARAMETERS


PUBLIC_RESULTS_FORMAT_VERSION = 1
_REQUIRED_FILES = {"README.md", "manifest.json", "summary.json", "metrics.csv"}
_OPTIONAL_FILES = {"model_comparison.csv", "learning_curve.csv", "tuning_summary.json"}
_ALLOWED_FILES = _REQUIRED_FILES | _OPTIONAL_FILES
_REGIMES = ("bull", "bear", "range")
_TOLERANCES = ("exact", "within_one_candle", "within_two_candles")

# Exact names intentionally avoid substring matching: public prose may legitimately say what was
# excluded, while structured raw observations and row-level identifiers are never permitted.
PROHIBITED_PUBLIC_FIELDS = frozenset({
    "open", "high", "low", "close", "volume", "open_gap", "body",
    "high_from_close", "low_from_close", "normalized_features", "features",
    "feature_matrix", "feature_matrices", "prepared_windows", "windows", "tensors",
    "candlestick_id", "candlestick_ids", "timestamp", "timestamps", "session_ids",
    "development_session_ids", "test_session_ids", "predictions",
    "human_current_regime", "human_anticipated_regime", "annotation_database_path",
    "normalized_path", "candlestick_path", "snapshot_path", "run_directory",
})

_HARDWARE_FIELDS = frozenset({
    "platform", "machine", "processor", "logical_cpu_count", "physical_cpu_count",
    "cpu_worker_count", "data_loader_worker_count", "ram_bytes", "device",
    "pytorch_version", "pytorch_cuda_version", "pytorch_hip_version",
    "accelerator_name", "backend",
})
_LIBRARY_FIELDS = frozenset({
    "python", "python_version", "platform", "numpy", "pandas", "scikit_learn",
    "sklearn", "torch",
})
_SOURCE_FIELDS = frozenset({"revision", "dirty", "source_sha256"})
_PROTOCOL_FIELDS = (
    "protocol_version", "total_session_count", "development_session_count",
    "test_session_count", "test_fraction", "rounding_rule", "corpus_status",
    "chronological_validation_folds", "learning_curve_session_counts",
    "learning_curve_evaluation_range",
)
_EFFICIENCY_FIELDS = (
    "training_seconds", "inference_seconds", "serialized_model_bytes", "parameter_count",
    "training_sample_count", "inference_sample_count", "training_samples_per_second",
    "inference_samples_per_second", "device", "cpu_worker_count", "timing_repetitions",
    "data_loader_worker_count",
)


def publish_benchmark_results(run: str | Path, output: str | Path) -> Path:
    """Export one completed run or one sealed completed study without private artifacts."""

    source = Path(run)
    output = Path(output)
    if not source.is_dir():
        raise ValueError(f"Benchmark run or study does not exist: {source}")

    public = _build_public_result(source)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".pricesanity-public-", dir=output.parent) as temporary:
        staging = Path(temporary) / "bundle"
        staging.mkdir()
        _write_json(staging / "summary.json", public["summary"])
        _write_csv(staging / "metrics.csv", public["metric_rows"])
        if len(public["summary"]["models"]) > 1:
            _write_csv(staging / "model_comparison.csv", public["comparison_rows"])
        if public["learning_rows"]:
            _write_csv(staging / "learning_curve.csv", public["learning_rows"])
        if public["tuning"]:
            _write_json(staging / "tuning_summary.json", {
                "format_version": PUBLIC_RESULTS_FORMAT_VERSION,
                "selections": public["tuning"],
            })

        identity = canonical_sha256({
            "summary": public["summary"],
            "metrics": public["metric_rows"],
            "comparison": public["comparison_rows"],
            "learning_curve": public["learning_rows"],
            "tuning": public["tuning"],
        })
        manifest = _manifest(public, identity)
        (staging / "README.md").write_text(_readme(manifest, public["summary"]), encoding="utf-8")
        manifest["artifacts"] = [
            {"name": path.name, "sha256": file_sha256(path)}
            for path in sorted(staging.iterdir(), key=lambda item: item.name)
        ]
        _write_json(staging / "manifest.json", manifest)
        validate_public_results(staging)

        if output.exists():
            if not output.is_dir() or _directory_bytes(output) != _directory_bytes(staging):
                raise FileExistsError(
                    f"Public result output exists with different content: {output}"
                )
            return output
        staging.rename(output)
    return output


def validate_public_results(directory: str | Path) -> dict[str, Any]:
    """Verify the public bundle schema, checksums, and privacy boundary."""

    directory = Path(directory)
    if not directory.is_dir():
        raise ValueError("Public benchmark result must be a directory.")
    entries = list(directory.iterdir())
    if any(path.is_symlink() or not path.is_file() for path in entries):
        raise ValueError("Public benchmark result may contain only direct regular files.")
    names = {path.name for path in entries}
    if not _REQUIRED_FILES.issubset(names) or not names.issubset(_ALLOWED_FILES):
        raise ValueError("Public benchmark result contains a missing or unsupported artifact.")

    structured: dict[str, Any] = {}
    for name in sorted(names):
        path = directory / name
        if path.suffix == ".json":
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                raise ValueError(f"Malformed public JSON artifact: {name}") from error
            _validate_public_value(value)
            structured[name] = value
        elif path.suffix == ".csv":
            with path.open(newline="", encoding="utf-8") as stream:
                reader = csv.DictReader(stream)
                if reader.fieldnames is None or len(reader.fieldnames) != len(set(reader.fieldnames)):
                    raise ValueError(f"Malformed public CSV header: {name}")
                for field in reader.fieldnames:
                    if field in PROHIBITED_PUBLIC_FIELDS:
                        raise ValueError(f"Prohibited public field: {field}")
                for row in reader:
                    _validate_public_value(row)

    manifest = structured.get("manifest.json")
    if not isinstance(manifest, dict) or manifest.get("format_version") != PUBLIC_RESULTS_FORMAT_VERSION:
        raise ValueError("Unsupported public benchmark manifest.")
    listed = manifest.get("artifacts")
    if not isinstance(listed, list):
        raise ValueError("Public manifest artifact list is malformed.")
    expected = names - {"manifest.json"}
    observed: set[str] = set()
    for artifact in listed:
        if not isinstance(artifact, dict) or set(artifact) != {"name", "sha256"}:
            raise ValueError("Public manifest artifact entry is malformed.")
        name = artifact["name"]
        if name not in expected or name in observed:
            raise ValueError("Public manifest artifact set is inconsistent.")
        if file_sha256(directory / name) != artifact["sha256"]:
            raise ValueError(f"Public artifact checksum mismatch: {name}")
        observed.add(name)
    if observed != expected:
        raise ValueError("Public manifest does not cover every public artifact.")
    return manifest


def _build_public_result(source: Path) -> dict[str, Any]:
    is_run = (source / "benchmark_metadata.json").is_file()
    is_study = (source / "study_scope.json").is_file()
    if is_run == is_study:
        raise ValueError("Source must be exactly one completed run or one benchmark study.")

    if is_run:
        runs = [_load_run(source)]
        kind = "completed_run"
        scope = None
        learning_runs: list[dict[str, Any]] = []
        tuning: list[dict[str, Any]] = []
    else:
        kind = "completed_study"
        scope = _read_mapping(source / "study_scope.json", "study scope")
        seal = _read_mapping(source / "development_frozen.json", "development freeze")
        if seal.get("state") != "frozen" or seal.get("scope") != scope:
            raise ValueError("Study development freeze does not match its immutable scope.")
        runs = [
            _load_run(path.parent)
            for path in sorted((source / "runs").rglob("benchmark_metadata.json"))
            if str(_read_mapping(path, "benchmark metadata").get("identity", {}).get("run_name", ""))
            .startswith("final_seed_")
        ]
        if not runs:
            raise ValueError("Study contains no completed final benchmark runs.")
        _validate_complete_study(runs, scope)
        learning_runs = [
            _load_run(path.parent)
            for path in sorted((source / "learning_curves").rglob("benchmark_metadata.json"))
        ]
        tuning = _load_tuning(source / "selected")
        expected_selections = {
            (str(track), str(model))
            for track in scope["tracks"] for model in scope["models"]
        }
        if {(str(item["track"]), str(item["model_name"])) for item in tuning} != expected_selections:
            raise ValueError("Public study export requires every frozen model selection.")

    _require_consistent_protocol(runs)
    metric_rows = [_metric_row(run) for run in sorted(runs, key=_run_sort_key)]
    models = _aggregate_models(runs)
    protocol = _protocol(runs)
    population = _population(runs)
    if scope is not None and scope.get("snapshot_sha256") != population.get("snapshot_sha256"):
        raise ValueError(
            "Completed final results do not match the study's frozen population."
        )
    environments = _environments(runs)
    source_identity = _source_identity(runs)
    summary = {
        "format_version": PUBLIC_RESULTS_FORMAT_VERSION,
        "result_kind": kind,
        "status": "complete",
        "protocol": protocol,
        "population": population,
        "source": source_identity,
        "models": models,
        "execution_environments": environments,
    }
    learning_rows = [_learning_row(run) for run in sorted(learning_runs, key=_run_sort_key)]
    comparison_rows = [_comparison_row(model) for model in models]
    return {
        "summary": summary,
        "metric_rows": metric_rows,
        "comparison_rows": comparison_rows,
        "learning_rows": learning_rows,
        "tuning": tuning,
        "scope": scope,
    }


def _load_run(directory: Path) -> dict[str, Any]:
    metadata, metrics = load_benchmark_summary(directory)
    identity = metadata.get("identity")
    if not isinstance(identity, dict):
        raise ValueError("Completed benchmark run lacks a valid identity.")
    return {
        "identity": identity,
        "metadata": metadata,
        "metrics": _select_metrics(metrics),
        "configuration": _select_configuration(identity, metadata.get("model_configuration")),
        "experiment_sha256": canonical_sha256(identity),
    }


def _select_configuration(identity: Mapping[str, Any], value: Any) -> dict[str, Any]:
    model_name = str(identity.get("model_name", ""))
    if not isinstance(value, dict):
        raise ValueError("Benchmark model configuration must be a mapping.")
    parameters = value.get("parameters", value)
    if not isinstance(parameters, dict):
        raise ValueError("Benchmark conceptual parameters must be a mapping.")
    allowed = set(MODEL_CONCEPTUAL_PARAMETERS.get(model_name, frozenset()))
    if model_name in MODEL_CONCEPTUAL_PARAMETERS:
        allowed.add("context_length")
    # Some artifacts wrap conceptual parameters with public identity keys. Ignore wrappers but
    # never publish a future estimator-specific field without explicitly adding it here.
    selected = {key: parameters[key] for key in sorted(allowed) if key in parameters}
    unknown = set(parameters) - allowed
    if model_name in MODEL_CONCEPTUAL_PARAMETERS and unknown:
        raise ValueError(
            f"Public export refuses unallowlisted {model_name} parameter(s): "
            + ", ".join(sorted(unknown))
        )
    _validate_public_value(selected)
    return {"model_name": model_name, "parameters": selected}


def _select_metrics(metrics: Any) -> dict[str, Any]:
    if not isinstance(metrics, dict):
        raise ValueError("Benchmark metrics must be a mapping.")
    selected = {
        "current": _classification(metrics.get("current")),
        "anticipated": _classification(metrics.get("anticipated")),
        "transitions": _transitions(metrics.get("transitions")),
        "transition_neighborhoods": _neighborhoods(metrics.get("transition_neighborhoods")),
        "mean_head_macro_f1": _number(metrics.get("mean_head_macro_f1")),
    }
    if metrics.get("anticipated_transitions") is not None:
        selected["anticipated_transitions"] = _transitions(metrics["anticipated_transitions"])
    if metrics.get("anticipated_transition_neighborhoods") is not None:
        selected["anticipated_transition_neighborhoods"] = _neighborhoods(
            metrics["anticipated_transition_neighborhoods"]
        )
    for name in ("current_probability", "anticipated_probability"):
        if metrics.get(name) is not None:
            value = metrics[name]
            selected[name] = {
                key: _integer(value[key]) if key == "support" else _number(value[key])
                for key in ("multiclass_log_loss", "multiclass_brier_score", "support")
            }
    if metrics.get("efficiency") is not None:
        value = metrics["efficiency"]
        selected["efficiency"] = {
            key: value.get(key) for key in _EFFICIENCY_FIELDS if key in value
        }
        selected["efficiency"]["hardware_fingerprint"] = _allowlist(
            value.get("hardware_fingerprint"), _HARDWARE_FIELDS
        )
        _validate_public_value(selected["efficiency"])
    return selected


def _classification(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("Classification metrics are missing.")
    per_class = value.get("per_class")
    matrix = value.get("confusion_matrix")
    if not isinstance(per_class, dict) or set(per_class) != set(_REGIMES):
        raise ValueError("Classification metrics require the fixed regime vocabulary.")
    if not isinstance(matrix, list) or len(matrix) != 3 or any(
        not isinstance(row, list) or len(row) != 3 for row in matrix
    ):
        raise ValueError("Classification confusion matrix must be three by three.")
    return {
        "accuracy": _number(value.get("accuracy")),
        "macro_f1": _number(value.get("macro_f1")),
        "support": _integer(value.get("support")),
        "per_class": {
            name: {
                "precision": _number(per_class[name].get("precision")),
                "recall": _number(per_class[name].get("recall")),
                "f1": _number(per_class[name].get("f1")),
                "support": _integer(per_class[name].get("support")),
            }
            for name in _REGIMES
        },
        "confusion_matrix": [[_integer(item) for item in row] for row in matrix],
    }


def _transitions(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("Transition metrics are missing.")
    result = {}
    for name in _TOLERANCES:
        item = value.get(name)
        if not isinstance(item, dict):
            raise ValueError(f"Transition metrics lack {name}.")
        result[name] = {
            key: (_number(item.get(key)) if key in {"precision", "recall"} else _integer(item.get(key)))
            for key in (
                "tolerance_candles", "human_transition_count", "predicted_transition_count",
                "matched_transition_count", "precision", "recall",
            )
        }
    return result


def _neighborhoods(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("Transition-neighborhood metrics are missing.")
    names = (
        "exact_current", "exact_anticipated", "within_one_current",
        "within_one_anticipated", "within_two_current", "within_two_anticipated",
    )
    return {
        name: None if value.get(name) is None else _classification(value[name])
        for name in names
    }


def _validate_complete_study(runs: Sequence[dict[str, Any]], scope: Mapping[str, Any]) -> None:
    models = scope.get("models")
    tracks = scope.get("tracks")
    if not isinstance(models, list) or not isinstance(tracks, list):
        raise ValueError("Study scope model and track declarations are malformed.")
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for run in runs:
        identity = run["identity"]
        grouped[(str(identity["track"]), str(identity["model_name"]))].append(run)
    expected = {(str(track), str(model)) for track in tracks for model in models}
    if set(grouped) != expected:
        raise ValueError("Completed final results do not cover the declared study scope.")
    for group in grouped.values():
        declared_sets = {
            tuple(sorted(int(seed) for seed in run["metadata"].get("dataset", {}).get(
                "declared_final_seeds", []
            )))
            for run in group
        }
        if len(declared_sets) != 1 or not next(iter(declared_sets)):
            raise ValueError("Final runs disagree about their declared seed set.")
        observed = tuple(sorted(int(run["identity"]["seed"]) for run in group))
        if observed != next(iter(declared_sets)) or len(observed) != len(set(observed)):
            raise ValueError("Completed final results do not match the declared seed set.")


def _require_consistent_protocol(runs: Sequence[dict[str, Any]]) -> None:
    first = _protocol([runs[0]])
    for run in runs[1:]:
        if _protocol([run]) != first:
            raise ValueError("Completed runs disagree about the benchmark protocol.")


def _protocol(runs: Sequence[dict[str, Any]]) -> dict[str, Any]:
    dataset = runs[0]["metadata"].get("dataset", {})
    protocol = {key: dataset[key] for key in _PROTOCOL_FIELDS if key in dataset}
    protocol["chronological_split_rule"] = (
        "first N-ceil(test_fraction*N) development; "
        "last ceil(test_fraction*N) test"
        if protocol.get("protocol_version") == "generalized_chronological_90_10_v1"
        else "recorded chronological development followed by final test"
    )
    return protocol


def _population(runs: Sequence[dict[str, Any]]) -> dict[str, Any]:
    identities = [run["identity"] for run in runs]
    datasets = [run["metadata"].get("dataset", {}) for run in runs]
    return {
        "development_population_sha256": _one(
            identity.get("training_session_ids_sha256") for identity in identities
        ),
        "test_population_sha256": _one(
            identity.get("test_session_ids_sha256") for identity in identities
        ),
        "annotation_snapshot_sha256": _one(
            identity.get("annotation_snapshot_sha256") for identity in identities
        ),
        "normalized_dataset_sha256": _one(
            identity.get("normalized_dataset_sha256") for identity in identities
        ),
        "snapshot_sha256": _one(dataset.get("snapshot_sha256") for dataset in datasets),
    }


def _source_identity(runs: Sequence[dict[str, Any]]) -> dict[str, Any]:
    values = [_allowlist(run["metadata"].get("git"), _SOURCE_FIELDS) for run in runs]
    return _one(values)


def _environments(runs: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    values = {}
    for run in runs:
        value = {
            "library_versions": _allowlist(run["metadata"].get("library_versions"), _LIBRARY_FIELDS),
            "hardware": run["metrics"].get("efficiency", {}).get("hardware_fingerprint", {}),
        }
        digest = canonical_sha256(value)
        values[digest] = {"environment_sha256": digest, **value}
    return [values[key] for key in sorted(values)]


def _aggregate_models(runs: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for run in runs:
        identity = run["identity"]
        key = (
            str(identity["track"]), str(identity["model_name"]),
            str(identity["model_configuration_sha256"]), str(identity["representation_sha256"]),
            str(identity["test_session_ids_sha256"]),
        )
        grouped[key].append(run)
    return [
        _aggregate_group(grouped[key])
        for key in sorted(grouped)
    ]


def _aggregate_group(runs: Sequence[dict[str, Any]]) -> dict[str, Any]:
    runs = sorted(runs, key=_run_sort_key)
    identity = runs[0]["identity"]
    configurations = [run["configuration"] for run in runs]
    configuration = _one(configurations)
    result = {
        "track": identity["track"],
        "model_name": identity["model_name"],
        "seed_count": len(runs),
        "seeds": [int(run["identity"]["seed"]) for run in runs],
        "model_configuration_sha256": identity["model_configuration_sha256"],
        "representation_sha256": identity["representation_sha256"],
        "test_population_sha256": identity["test_session_ids_sha256"],
        "benchmark_run_identity_sha256": sorted(run["experiment_sha256"] for run in runs),
        "configuration": configuration,
        "current": _aggregate_classifications([run["metrics"]["current"] for run in runs]),
        "anticipated": _aggregate_classifications([
            run["metrics"]["anticipated"] for run in runs
        ]),
        "mean_head_macro_f1": _stats([
            run["metrics"]["mean_head_macro_f1"] for run in runs
        ]),
        "current_transitions": _aggregate_transitions([
            run["metrics"]["transitions"] for run in runs
        ]),
    }
    if all("anticipated_transitions" in run["metrics"] for run in runs):
        result["anticipated_transitions"] = _aggregate_transitions([
            run["metrics"]["anticipated_transitions"] for run in runs
        ])
    for name in ("current_probability", "anticipated_probability"):
        present = [name in run["metrics"] for run in runs]
        if any(present) and not all(present):
            raise ValueError(f"Runs disagree about availability of {name}.")
        if all(present):
            result[name] = {
                key: _stats([run["metrics"][name][key] for run in runs])
                for key in ("multiclass_log_loss", "multiclass_brier_score")
            }
            result[name]["support"] = _one(run["metrics"][name]["support"] for run in runs)
    result["efficiency"] = _aggregate_efficiency(runs)
    return result


def _aggregate_classifications(values: Sequence[dict[str, Any]]) -> dict[str, Any]:
    result = {
        "accuracy": _stats([value["accuracy"] for value in values]),
        "macro_f1": _stats([value["macro_f1"] for value in values]),
        "support": _one(value["support"] for value in values),
        "per_class": {},
        "confusion_matrix_mean": [],
    }
    for name in _REGIMES:
        result["per_class"][name] = {
            key: _stats([value["per_class"][name][key] for value in values])
            for key in ("precision", "recall", "f1")
        }
        result["per_class"][name]["support"] = _one(
            value["per_class"][name]["support"] for value in values
        )
    for row in range(3):
        result["confusion_matrix_mean"].append([
            statistics.fmean(value["confusion_matrix"][row][column] for value in values)
            for column in range(3)
        ])
    return result


def _aggregate_transitions(values: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return {
        name: {
            "tolerance_candles": _one(value[name]["tolerance_candles"] for value in values),
            "human_transition_count": _one(
                value[name]["human_transition_count"] for value in values
            ),
            **{
                key: _stats([value[name][key] for value in values])
                for key in (
                    "predicted_transition_count", "matched_transition_count", "precision", "recall"
                )
            },
        }
        for name in _TOLERANCES
    }


def _aggregate_efficiency(runs: Sequence[dict[str, Any]]) -> dict[str, Any] | None:
    if not all("efficiency" in run["metrics"] for run in runs):
        return None
    values = [run["metrics"]["efficiency"] for run in runs]
    hardware = [value.get("hardware_fingerprint", {}) for value in values]
    compatible = len({canonical_sha256(value) for value in hardware}) == 1
    result: dict[str, Any] = {
        "hardware_compatible": compatible,
        "hardware_environment_sha256": canonical_sha256(hardware[0]) if compatible else None,
    }
    for name in _EFFICIENCY_FIELDS:
        items = [value.get(name) for value in values]
        if all(isinstance(item, (int, float)) and not isinstance(item, bool) for item in items):
            result[name] = _stats(items) if compatible else None
        elif len({json.dumps(item, sort_keys=True) for item in items}) == 1:
            result[name] = items[0]
        else:
            result[name] = None
    return result


def _metric_row(run: dict[str, Any]) -> dict[str, Any]:
    identity = run["identity"]
    metrics = run["metrics"]
    row: dict[str, Any] = {
        "track": identity["track"], "model_name": identity["model_name"],
        "seed": identity["seed"], "benchmark_run_identity_sha256": run["experiment_sha256"],
        "model_configuration_sha256": identity["model_configuration_sha256"],
        "representation_sha256": identity["representation_sha256"],
        "test_population_sha256": identity["test_session_ids_sha256"],
        "current_accuracy": metrics["current"]["accuracy"],
        "current_macro_f1": metrics["current"]["macro_f1"],
        "anticipated_accuracy": metrics["anticipated"]["accuracy"],
        "anticipated_macro_f1": metrics["anticipated"]["macro_f1"],
        "mean_head_macro_f1": metrics["mean_head_macro_f1"],
    }
    for prefix, name in (("current_transition", "transitions"),
                         ("anticipated_transition", "anticipated_transitions")):
        if name in metrics:
            for tolerance in _TOLERANCES:
                for key in ("precision", "recall", "matched_transition_count"):
                    row[f"{prefix}_{tolerance}_{key}"] = metrics[name][tolerance][key]
    efficiency = metrics.get("efficiency")
    if efficiency:
        row["hardware_environment_sha256"] = canonical_sha256(
            efficiency.get("hardware_fingerprint", {})
        )
        for key in ("training_seconds", "inference_seconds", "inference_samples_per_second",
                    "serialized_model_bytes", "parameter_count", "device"):
            row[key] = efficiency.get(key)
    return row


def _comparison_row(model: dict[str, Any]) -> dict[str, Any]:
    return {
        "track": model["track"], "model_name": model["model_name"],
        "seed_count": model["seed_count"],
        "current_macro_f1_mean": model["current"]["macro_f1"]["mean"],
        "current_macro_f1_std": model["current"]["macro_f1"]["std"],
        "anticipated_macro_f1_mean": model["anticipated"]["macro_f1"]["mean"],
        "anticipated_macro_f1_std": model["anticipated"]["macro_f1"]["std"],
        "mean_head_macro_f1_mean": model["mean_head_macro_f1"]["mean"],
        "mean_head_macro_f1_std": model["mean_head_macro_f1"]["std"],
        "model_configuration_sha256": model["model_configuration_sha256"],
        "test_population_sha256": model["test_population_sha256"],
    }


def _learning_row(run: dict[str, Any]) -> dict[str, Any]:
    row = _metric_row(run)
    dataset = run["metadata"].get("dataset", {})
    identity = run["identity"]
    row["training_session_count"] = (
        int(identity["training_session_range"][1]) - int(identity["training_session_range"][0])
    )
    if "development_session_count" in dataset:
        row["development_session_count"] = dataset["development_session_count"]
    return row


def _load_tuning(directory: Path) -> list[dict[str, Any]]:
    records = []
    if not directory.is_dir():
        return records
    for path in sorted(directory.glob("*/*.json")):
        value = _read_mapping(path, "selected model")
        if value.get("state") != "frozen":
            continue
        model = str(value.get("model_name", ""))
        parameters = value.get("parameters", {})
        allowed = set(MODEL_CONCEPTUAL_PARAMETERS.get(model, frozenset()))
        if model in MODEL_CONCEPTUAL_PARAMETERS:
            allowed.add("context_length")
        if not isinstance(parameters, dict) or set(parameters) - allowed:
            raise ValueError(f"Selected {model} parameters are not publicly allowlisted.")
        records.append({
            "model_name": model,
            "track": value.get("track"),
            "parameters": {key: parameters[key] for key in sorted(parameters)},
            "tuning_budget": value.get("tuning_budget"),
            "tuning_seed": value.get("tuning_seed"),
            "fold_count": value.get("fold_count"),
            "fold_indices": value.get("fold_indices"),
            "fold_mean_head_macro_f1": value.get("fold_mean_head_macro_f1"),
            "mean_validation_macro_f1": value.get("mean_validation_macro_f1"),
            "protocol_sha256": value.get("protocol_sha256"),
            "search_space_sha256": value.get("search_space_sha256"),
            "representation_sha256": value.get("representation_sha256"),
            "parameters_sha256": value.get("parameters_sha256"),
        })
    return records


def _manifest(public: Mapping[str, Any], identity: str) -> dict[str, Any]:
    summary = public["summary"]
    return {
        "format_version": PUBLIC_RESULTS_FORMAT_VERSION,
        "public_result_identity_sha256": identity,
        "result_kind": summary["result_kind"],
        "status": summary["status"],
        "protocol": summary["protocol"],
        "population": summary["population"],
        "source": summary["source"],
        "models": [
            {"track": item["track"], "model_name": item["model_name"],
             "seed_count": item["seed_count"]}
            for item in summary["models"]
        ],
        "privacy_boundary": {
            "publication_policy": "explicit_allowlist_only",
            "contains_aggregate_metrics": True,
            "contains_model_hyperparameters": True,
            "contains_population_hashes": True,
            "excluded_artifacts": [
                "trained_models", "prediction_rows", "source_market_data", "normalized_features",
                "annotation_database", "session_identifiers", "candlestick_identifiers",
            ],
        },
        "reproducibility": {
            "public": "protocol, population hashes, source identity, settings, aggregate metrics",
            "private": "licensed source data, annotation database, exact IDs, predictions, models",
        },
        "artifacts": [],
    }


def _readme(manifest: Mapping[str, Any], summary: Mapping[str, Any]) -> str:
    protocol = manifest["protocol"]
    source_revision = (
        manifest["source"].get("revision")
        or manifest["source"].get("source_sha256", "unspecified")
    )
    environment = json.dumps(
        summary["execution_environments"],
        sort_keys=True,
        separators=(",", ":"),
    )
    models = ", ".join(
        f"{item['track']}/{item['model_name']} ({item['seed_count']} seed(s))"
        for item in manifest["models"]
    )
    return (
        "# Price Sanity public benchmark result\n\n"
        f"Public result identity: `{manifest['public_result_identity_sha256']}`\n\n"
        f"Protocol: `{protocol.get('protocol_version', 'unspecified')}`. "
        f"This result evaluates {protocol.get('test_session_count', 'an unspecified number of')} "
        "chronologically last sessions after development-only model selection.\n\n"
        f"Corpus status: `{protocol.get('corpus_status', 'unspecified')}`. Counts: "
        f"{protocol.get('total_session_count', 'unspecified')} total / "
        f"{protocol.get('development_session_count', 'unspecified')} development / "
        f"{protocol.get('test_session_count', 'unspecified')} test under the chronological "
        "90/10 design with the test count rounded up.\n\n"
        f"Source revision: `{source_revision}`.\n\n"
        f"Execution environment evidence: `{environment}`.\n\n"
        f"Included configurations: {models}. Primary comparison is the mean of current-regime "
        "and anticipated-regime macro F1; both heads and exact / ±1 / ±2 transition diagnostics "
        "remain separately reported.\n\n"
        "This directory is a deterministic allowlisted export intended for source control. It "
        "contains aggregate scientific results, safe configuration/provenance fields, and hashes. "
        "It does not contain licensed market data, annotations, exact session/candlestick IDs, "
        "prediction rows, feature matrices, database files, or trained models. Full reproduction "
        "therefore requires the separately held private inputs named by the hashes.\n"
    )


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    if not fields:
        raise ValueError(f"Cannot publish empty table: {path.name}")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in fields})


def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    return value


def _validate_public_value(value: Any) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if key in PROHIBITED_PUBLIC_FIELDS:
                raise ValueError(f"Prohibited public field: {key}")
            _validate_public_value(item)
    elif isinstance(value, list):
        for item in value:
            _validate_public_value(item)
    elif (
        isinstance(value, str)
        and (Path(value).is_absolute() or PureWindowsPath(value).is_absolute())
    ):
        raise ValueError("Absolute paths are prohibited in public structured artifacts.")


def _allowlist(value: Any, fields: Iterable[str]) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    result = {key: value[key] for key in sorted(set(fields)) if key in value}
    _validate_public_value(result)
    return result


def _read_mapping(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not read {label}: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label.capitalize()} must be a JSON object.")
    return value


def _directory_bytes(directory: Path) -> dict[str, bytes]:
    return {
        str(path.relative_to(directory)): path.read_bytes()
        for path in sorted(directory.rglob("*")) if path.is_file()
    }


def _run_sort_key(run: Mapping[str, Any]) -> tuple[str, str, str, int]:
    identity = run["identity"]
    return (
        str(identity.get("track", "")), str(identity.get("model_name", "")),
        str(identity.get("run_name", "")), int(identity.get("seed", 0)),
    )


def _one(values: Iterable[Any]) -> Any:
    values = list(values)
    if not values:
        raise ValueError("Expected at least one public value.")
    canonical = {json.dumps(value, sort_keys=True, separators=(",", ":")) for value in values}
    if len(canonical) != 1:
        raise ValueError("Completed runs disagree about public provenance.")
    return values[0]


def _stats(values: Sequence[int | float]) -> dict[str, float]:
    numeric = [_number(value) for value in values]
    return {"mean": statistics.fmean(numeric), "std": statistics.pstdev(numeric)}


def _number(value: Any) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
        raise ValueError("Public metric values must be finite numbers.")
    return float(value)


def _integer(value: Any) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError("Public count values must be integers.")
    return value
