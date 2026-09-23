"""Focused tests for deterministic, privacy-preserving benchmark publication."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd
import pytest

from pricesanity.benchmark.artifacts import (
    BenchmarkRunIdentity,
    canonical_sha256,
    file_sha256,
    prepare_benchmark_run,
    save_benchmark_run,
)
from pricesanity.benchmark.cli import main
from pricesanity.benchmark.metrics import evaluate_benchmark_predictions
from pricesanity.benchmark.publication import (
    PROHIBITED_PUBLIC_FIELDS,
    publish_benchmark_results,
    validate_public_results,
)
from pricesanity.models.baselines import MajorityClassBaseline


_HASH = "a" * 64
_PROTOCOL = {
    "protocol_version": "generalized_chronological_90_10_v1",
    "total_session_count": 550,
    "development_session_count": 495,
    "test_session_count": 55,
    "test_fraction": 0.1,
    "rounding_rule": "ceil(test_fraction * N)",
    "corpus_status": "interim",
    "chronological_validation_folds": [
        {"training": [0, 385], "validation": [385, 440]},
        {"training": [0, 440], "validation": [440, 495]},
    ],
    "learning_curve_session_counts": [100, 200, 300, 400],
    "learning_curve_evaluation_range": [440, 495],
}


def _predictions() -> pd.DataFrame:
    labels = ["bull", "bear", "range", "bull", "bear", "range"]
    predictions = ["bull", "bear", "bull", "bull", "range", "range"]
    probabilities = {
        "bull": [1.0, 0.0, 1.0, 1.0, 0.0, 0.0],
        "bear": [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        "range": [0.0, 0.0, 0.0, 0.0, 1.0, 1.0],
    }
    return pd.DataFrame({
        "candlestick_id": [f"PRIVATE-CANDLE-{index}" for index in range(6)],
        "timestamp": pd.date_range("2026-01-02T14:30:00Z", periods=6, freq="5min"),
        "session_date": [pd.Timestamp("2026-01-02").date()] * 3
        + [pd.Timestamp("2026-01-03").date()] * 3,
        "session_index": [495] * 3 + [496] * 3,
        "candle_position": list(range(3)) * 2,
        "uncertainty_kind": ["deterministic_distribution"] * 6,
        "predicted_current_regime": predictions,
        "predicted_anticipated_regime": predictions[1:] + predictions[:1],
        **{
            f"current_probability_{name}": values
            for name, values in probabilities.items()
        },
        **{
            f"anticipated_probability_{name}": values[1:] + values[:1]
            for name, values in probabilities.items()
        },
        **{
            f"{head}_score_{name}": [np.nan] * 6
            for head in ("current", "anticipated")
            for name in ("bull", "bear", "range")
        },
        "human_current_regime": labels,
        "human_anticipated_regime": labels[1:] + labels[:1],
    })


def _create_run(
    root: Path,
    *,
    model_name: str,
    run_name: str = "final_seed_7",
    output_branch: str = "runs",
    seed: int = 7,
) -> Path:
    configuration = (
        {"model_name": "majority_class", "adapter": "native"}
        if model_name == "majority_class"
        else {
            "model_name": model_name,
            "parameters": {"C": 1.0, "class_weight": None},
            "track": "controlled",
        }
    )
    identity = BenchmarkRunIdentity(
        track="controlled",
        model_name=model_name,
        run_name=run_name,
        seed=seed,
        feature_representation="tabular_16x4",
        window_length=16,
        feature_columns=("open_gap", "body", "high_from_close", "low_from_close"),
        training_session_range=(0, 495 if output_branch == "runs" else 100),
        validation_session_ranges=(),
        test_session_range=(495, 550),
        model_configuration_sha256=canonical_sha256(configuration),
        representation_sha256=canonical_sha256({"name": "tabular_16x4"}),
        training_session_ids_sha256=canonical_sha256([f"PRIVATE-DEV-{i}" for i in range(495)]),
        validation_session_ids_sha256=canonical_sha256([]),
        test_session_ids_sha256=canonical_sha256([f"PRIVATE-TEST-{i}" for i in range(55)]),
        annotation_snapshot_sha256=canonical_sha256("private annotations"),
        normalized_dataset_sha256=canonical_sha256("private normalized rows"),
        protocol_sha256=canonical_sha256(_PROTOCOL),
        search_space_sha256=canonical_sha256({}),
        label_mapping_sha256=canonical_sha256(["bull", "bear", "range"]),
    )
    paths, _ = prepare_benchmark_run(root / output_branch, identity)
    model = MajorityClassBaseline()
    features = np.ones((6, 64), dtype=np.float32)
    targets = np.arange(6) % 3
    model.fit(features, targets, np.roll(targets, -1))
    frame = _predictions()
    index = {"bull": 0, "bear": 1, "range": 2}
    metrics = evaluate_benchmark_predictions(
        human_current=frame["human_current_regime"].map(index).to_numpy(),
        predicted_current=frame["predicted_current_regime"].map(index).to_numpy(),
        human_anticipated=frame["human_anticipated_regime"].map(index).to_numpy(),
        predicted_anticipated=frame["predicted_anticipated_regime"].map(index).to_numpy(),
        session_indices=frame["session_index"].to_numpy(),
    )
    save_benchmark_run(
        paths,
        identity,
        model=model,
        predictions=frame,
        metrics=metrics,
        model_configuration=configuration,
        dataset_description={
            **_PROTOCOL,
            "snapshot_path": "/private/licensed/snapshot.parquet",
            "snapshot_sha256": canonical_sha256("snapshot"),
            "declared_final_seeds": [seed],
            "test_session_ids": ["PRIVATE-TEST-ROW"],
            "future_private_field": "SECRET-METADATA-SENTINEL",
        },
    )
    return paths.directory


def _study(tmp_path: Path) -> Path:
    study = tmp_path / "private-study"
    models = ["majority_class", "logistic_regression"]
    scope = {"models": models, "tracks": ["controlled"], "snapshot_sha256": canonical_sha256("snapshot")}
    study.mkdir()
    (study / "study_scope.json").write_text(json.dumps(scope), encoding="utf-8")
    (study / "development_frozen.json").write_text(
        json.dumps({"format_version": 1, "state": "frozen", "scope": scope}),
        encoding="utf-8",
    )
    for model in models:
        _create_run(study, model_name=model)
        selected = {
            "format_version": 1,
            "state": "frozen",
            "model_name": model,
            "track": "controlled",
            "parameters": {} if model == "majority_class" else {"C": 1.0, "class_weight": None},
            "tuning_budget": 0 if model == "majority_class" else 12,
            "tuning_seed": 8675309,
            "fold_count": 2,
            "fold_indices": [1, 2],
            "fold_mean_head_macro_f1": [0.4, 0.5],
            "mean_validation_macro_f1": 0.45,
            "protocol_sha256": canonical_sha256(_PROTOCOL),
            "search_space_sha256": _HASH,
            "representation_sha256": _HASH,
            "parameters_sha256": _HASH,
            "test_session_ids": ["PRIVATE-SELECTION-ID"],
        }
        path = study / "selected" / "controlled" / f"{model}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(selected), encoding="utf-8")
    _create_run(
        study, model_name="majority_class", run_name="train_100", output_branch="learning_curves"
    )
    (study / "tuning" / "private.sqlite3").parent.mkdir(parents=True)
    (study / "tuning" / "private.sqlite3").write_bytes(b"PRIVATE-SQLITE-CONTENT")
    return study


def _tree_hashes(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): file_sha256(path)
        for path in sorted(root.rglob("*")) if path.is_file()
    }


def _structured_field_names(value):
    if isinstance(value, dict):
        yield from value
        for child in value.values():
            yield from _structured_field_names(child)
    elif isinstance(value, list):
        for child in value:
            yield from _structured_field_names(child)


def test_study_export_is_complete_allowlisted_and_model_independent(tmp_path):
    study = _study(tmp_path)
    output = publish_benchmark_results(study, tmp_path / "public")

    manifest = validate_public_results(output)
    summary = json.loads((output / "summary.json").read_text())
    assert manifest["protocol"]["total_session_count"] == 550
    assert manifest["protocol"]["development_session_count"] == 495
    assert manifest["protocol"]["test_session_count"] == 55
    assert manifest["protocol"]["rounding_rule"] == "ceil(test_fraction * N)"
    assert len(manifest["population"]["development_population_sha256"]) == 64
    assert len(manifest["population"]["test_population_sha256"]) == 64
    assert {item["model_name"] for item in summary["models"]} == {
        "majority_class", "logistic_regression"
    }
    for model in summary["models"]:
        assert set(model) >= {
            "current", "anticipated", "current_transitions", "anticipated_transitions"
        }
        assert set(model["current_transitions"]) == {
            "exact", "within_one_candle", "within_two_candles"
        }
    assert (output / "model_comparison.csv").is_file()
    assert (output / "learning_curve.csv").is_file()
    tuning = json.loads((output / "tuning_summary.json").read_text())
    assert {item["tuning_budget"] for item in tuning["selections"]} == {0, 12}
    readme = (output / "README.md").read_text()
    assert "550 total / 495 development / 55 test" in readme
    assert "Corpus status: `interim`" in readme
    assert "Source revision:" in readme
    assert "Execution environment evidence:" in readme
    assert "licensed market data" in readme


def test_export_omits_private_rows_paths_files_and_unknown_fields(tmp_path):
    output = publish_benchmark_results(_study(tmp_path), tmp_path / "public")
    assert {path.name for path in output.iterdir()} <= {
        "README.md", "manifest.json", "summary.json", "metrics.csv",
        "model_comparison.csv", "learning_curve.csv", "tuning_summary.json",
    }
    joined = b"\n".join(path.read_bytes() for path in output.iterdir())
    for secret in (
        b"PRIVATE-CANDLE", b"PRIVATE-TEST-ROW", b"PRIVATE-SELECTION-ID",
        b"SECRET-METADATA-SENTINEL", b"PRIVATE-SQLITE-CONTENT", b"/private/licensed",
    ):
        assert secret not in joined
    for path in output.glob("*.json"):
        fields = set(_structured_field_names(json.loads(path.read_text())))
        assert fields.isdisjoint(PROHIBITED_PUBLIC_FIELDS)
    for path in output.glob("*.csv"):
        with path.open(newline="") as stream:
            assert set(csv.DictReader(stream).fieldnames or ()).isdisjoint(PROHIBITED_PUBLIC_FIELDS)


def test_export_is_deterministic_idempotent_and_does_not_mutate_private_artifacts(tmp_path):
    study = _study(tmp_path)
    before = _tree_hashes(study)
    first = publish_benchmark_results(study, tmp_path / "first")
    second = publish_benchmark_results(study, tmp_path / "second")
    assert _tree_hashes(first) == _tree_hashes(second)
    assert {path.name: path.read_bytes() for path in first.iterdir()} == {
        path.name: path.read_bytes() for path in second.iterdir()
    }
    assert publish_benchmark_results(study, first) == first
    assert _tree_hashes(study) == before
    (first / "README.md").write_text("changed", encoding="utf-8")
    with pytest.raises(FileExistsError, match="different content"):
        publish_benchmark_results(study, first)


def test_validator_rejects_injected_private_field_and_individual_run_exports(tmp_path):
    run = _create_run(tmp_path / "one", model_name="majority_class")
    output = publish_benchmark_results(run, tmp_path / "public-run")
    summary = json.loads((output / "summary.json").read_text())
    assert summary["result_kind"] == "completed_run"
    summary["session_ids"] = ["private"]
    (output / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    with pytest.raises(ValueError, match="Prohibited public field"):
        validate_public_results(output)


def test_publish_cli_works_without_config_and_results_are_git_trackable(tmp_path, capsys):
    run = _create_run(tmp_path / "one", model_name="majority_class")
    output = tmp_path / "public-cli"
    assert main([
        "--config", str(tmp_path / "does-not-exist.yaml"),
        "publish-results", "--run", str(run), "--output", str(output),
    ]) == 0
    assert "Published deterministic public benchmark result" in capsys.readouterr().out
    repository = Path(__file__).resolve().parents[1]
    ignored = subprocess.run(
        ["git", "check-ignore", "--no-index", "results/benchmark/example/manifest.json"],
        cwd=repository,
        check=False,
        capture_output=True,
        text=True,
    )
    assert ignored.returncode == 1, ignored.stdout + ignored.stderr
    for private_path in (
        "data/raw/source.dbn",
        "data/interim/candles.parquet",
        "data/processed/normalized.parquet",
        "data/annotations/pricesanity.sqlite3",
        "data/models/benchmark/private/model.bin",
    ):
        private = subprocess.run(
            ["git", "check-ignore", "--no-index", private_path],
            cwd=repository, check=False, capture_output=True, text=True,
        )
        assert private.returncode == 0, private.stdout + private.stderr
