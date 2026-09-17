import json

import numpy as np
import pandas as pd
import pytest

from pricesanity.benchmark.artifacts import (
    BenchmarkRunIdentity,
    audit_benchmark_run,
    canonical_sha256,
    load_benchmark_summary,
    load_benchmark_run,
    prepare_benchmark_run,
    resume_environment_fingerprint,
    save_benchmark_run,
)
from pricesanity.benchmark.metrics import evaluate_benchmark_predictions
from pricesanity.models.baselines import MajorityClassBaseline


def _identity() -> BenchmarkRunIdentity:
    """Describe a development fold with no final-holdout membership."""

    # Independent digests let corruption tests change one identity dimension without needing
    # licensed prices, live annotations, or a completed full-corpus study.
    model_configuration = {"model_name": "majority_class", "adapter": "native"}
    return BenchmarkRunIdentity(
        track="controlled",
        model_name="majority_class",
        run_name="fold_001",
        seed=42,
        feature_representation="tabular_16x4",
        window_length=16,
        feature_columns=("open_gap", "body", "high_from_close", "low_from_close"),
        training_session_range=(0, 1000),
        validation_session_ranges=((1000, 1100),),
        test_session_range=None,
        model_configuration_sha256=canonical_sha256(model_configuration),
        representation_sha256=canonical_sha256({"name": "tabular_16x4"}),
        training_session_ids_sha256=canonical_sha256(list(range(1000))),
        validation_session_ids_sha256=canonical_sha256(list(range(1000, 1100))),
        test_session_ids_sha256=canonical_sha256([]),
        annotation_snapshot_sha256=canonical_sha256("annotations"),
        normalized_dataset_sha256=canonical_sha256("normalized"),
        protocol_sha256=canonical_sha256("protocol-v1"),
        search_space_sha256=canonical_sha256("none"),
        label_mapping_sha256=canonical_sha256(["bull", "bear", "range"]),
    )


def _predictions() -> pd.DataFrame:
    """Match the majority model's native classes and fixed uncertainty schema."""

    # A point distribution is available for this baseline; score columns are deliberately NaN
    # because absent margins must not be represented as measured zero confidence.
    probabilities = [1.0, 0.0, 0.0]
    return pd.DataFrame({
        "candlestick_id": ["candle-1", "candle-2"],
        "timestamp": pd.date_range("2026-01-02T14:30:00Z", periods=2, freq="5min"),
        "session_date": [pd.Timestamp("2026-01-02").date()] * 2,
        "session_index": [1000, 1000],
        "candle_position": [15, 16],
        "uncertainty_kind": ["deterministic_distribution"] * 2,
        "predicted_current_regime": ["bull", "bull"],
        "predicted_anticipated_regime": ["bull", "bull"],
        "current_probability_bull": [probabilities[0]] * 2,
        "current_probability_bear": [probabilities[1]] * 2,
        "current_probability_range": [probabilities[2]] * 2,
        "anticipated_probability_bull": [probabilities[0]] * 2,
        "anticipated_probability_bear": [probabilities[1]] * 2,
        "anticipated_probability_range": [probabilities[2]] * 2,
        "current_score_bull": [np.nan] * 2,
        "current_score_bear": [np.nan] * 2,
        "current_score_range": [np.nan] * 2,
        "anticipated_score_bull": [np.nan] * 2,
        "anticipated_score_bear": [np.nan] * 2,
        "anticipated_score_range": [np.nan] * 2,
        "human_current_regime": ["bull", "bear"],
        "human_anticipated_regime": ["bull", "range"],
    })


def test_benchmark_artifact_identity_resume_and_checksums(tmp_path) -> None:
    """Completed runs verify cleanly and cannot be overwritten or mis-resumed."""

    identity = _identity()
    paths, state = prepare_benchmark_run(tmp_path, identity)
    assert state == "new"
    model = MajorityClassBaseline()
    features = np.ones((2, 64), dtype=np.float32)
    model.fit(features, np.array([0, 1]), np.array([0, 2]))
    metrics = evaluate_benchmark_predictions(
        human_current=np.array([0, 1]),
        predicted_current=np.array([0, 0]),
        human_anticipated=np.array([0, 2]),
        predicted_anticipated=np.array([0, 0]),
        session_indices=np.array([1000, 1000]),
    )
    save_benchmark_run(
        paths,
        identity,
        model=model,
        predictions=_predictions(),
        metrics=metrics,
        model_configuration={"model_name": "majority_class", "adapter": "native"},
        dataset_description={
            "instrument": "ES.v.0",
            "target_interval": "5min",
            "session_timezone": "America/New_York",
            "candlestick_path": "candles.parquet",
        },
    )

    metadata, predictions, loaded_metrics = load_benchmark_run(paths.directory)
    assert metadata["identity"]["model_name"] == "majority_class"
    assert len(predictions) == 2
    assert loaded_metrics["current"]["support"] == 2
    assert prepare_benchmark_run(tmp_path, identity, resume=True)[1] == "complete"
    with pytest.raises(FileExistsError):
        prepare_benchmark_run(tmp_path, identity)

    identity_values = json.loads(paths.identity.read_text())
    identity_values["seed"] = 999
    paths.identity.write_text(json.dumps(identity_values))
    with pytest.raises(ValueError, match="different run identity"):
        prepare_benchmark_run(tmp_path, identity, resume=True)


def test_benchmark_run_resumes_checkpointed_components(tmp_path) -> None:
    """A lost final publication can reuse verified model, prediction, and metric files."""

    identity = _identity()
    paths, _ = prepare_benchmark_run(tmp_path, identity)
    model = MajorityClassBaseline()
    features = np.ones((2, 64), dtype=np.float32)
    model.fit(features, np.array([0, 1]), np.array([0, 2]))
    metrics = evaluate_benchmark_predictions(
        human_current=np.array([0, 1]), predicted_current=np.array([0, 0]),
        human_anticipated=np.array([0, 2]), predicted_anticipated=np.array([0, 0]),
        session_indices=np.array([1000, 1000]),
    )
    arguments = dict(
        model=model, predictions=_predictions(), metrics=metrics,
        model_configuration={"model_name": "majority_class", "adapter": "native"},
        dataset_description={"name": "synthetic"},
    )
    save_benchmark_run(paths, identity, **arguments)
    original_hashes = {
        path.name: path.read_bytes()
        for path in (paths.model, paths.predictions, paths.metrics)
    }

    # Simulate interruption after all components were checkpointed but before completion was
    # published. Resume must not invoke exclusive model saving again.
    paths.metadata.unlink()
    resumed_paths, state = prepare_benchmark_run(tmp_path, identity, resume=True)
    assert state == "resume"
    save_benchmark_run(resumed_paths, identity, **arguments)
    assert {
        path.name: path.read_bytes()
        for path in (paths.model, paths.predictions, paths.metrics)
    } == original_hashes
    assert load_benchmark_run(paths.directory)[0]["state"] == "complete"


def test_fast_summary_does_not_read_prediction_parquet(tmp_path, monkeypatch) -> None:
    """Opening a leaderboard verifies small metrics without loading detailed rows."""

    identity = _identity()
    paths, _ = prepare_benchmark_run(tmp_path, identity)
    model = MajorityClassBaseline()
    model.fit(np.ones((2, 2)), np.array([0, 1]), np.array([0, 2]))
    metrics = evaluate_benchmark_predictions(
        human_current=np.array([0, 1]), predicted_current=np.array([0, 0]),
        human_anticipated=np.array([0, 2]), predicted_anticipated=np.array([0, 0]),
        session_indices=np.array([1000, 1000]),
    )
    save_benchmark_run(
        paths, identity, model=model, predictions=_predictions(), metrics=metrics,
        model_configuration={"model_name": "majority_class", "adapter": "native"},
        dataset_description={"name": "synthetic"},
    )
    monkeypatch.setattr(
        pd, "read_parquet",
        lambda *args, **kwargs: pytest.fail("summary loader read predictions"),
    )
    _, loaded_metrics = load_benchmark_summary(paths.directory)
    assert loaded_metrics["current"]["support"] == 2


def test_identity_hash_changes_with_configuration_and_dataset() -> None:
    """Resume paths cannot collide after a scientific input changes."""

    first = _identity()
    changed_configuration = canonical_sha256({"model_name": "majority_class", "x": 1})
    from dataclasses import replace

    second = replace(first, model_configuration_sha256=changed_configuration)
    third = replace(first, normalized_dataset_sha256=canonical_sha256("other dataset"))
    assert first.experiment_sha256 != second.experiment_sha256
    assert first.experiment_sha256 != third.experiment_sha256


def test_full_audit_recomputes_saved_metrics(tmp_path) -> None:
    """The expensive audit verifies that saved predictions reproduce saved metrics."""

    identity = _identity()
    paths, _ = prepare_benchmark_run(tmp_path, identity)
    model = MajorityClassBaseline()
    model.fit(np.ones((2, 2)), np.array([0, 1]), np.array([0, 2]))
    probabilities = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    metrics = evaluate_benchmark_predictions(
        human_current=np.array([0, 1]), predicted_current=np.array([0, 0]),
        human_anticipated=np.array([0, 2]), predicted_anticipated=np.array([0, 0]),
        current_probabilities=probabilities,
        anticipated_probabilities=probabilities,
        session_indices=np.array([1000, 1000]),
    )
    save_benchmark_run(
        paths, identity, model=model, predictions=_predictions(), metrics=metrics,
        model_configuration={"model_name": "majority_class", "adapter": "native"},
        dataset_description={"name": "synthetic"},
    )
    assert audit_benchmark_run(paths.directory)[2]["current"]["support"] == 2


def test_resume_rejects_a_new_stochastic_state_after_model_checkpoint(
    tmp_path, monkeypatch
) -> None:
    """A crash cannot mix an old fitted checkpoint with new downstream predictions."""

    identity = _identity()
    environment = resume_environment_fingerprint(device="cpu")
    paths, _ = prepare_benchmark_run(
        tmp_path, identity, execution_environment=environment
    )
    first = MajorityClassBaseline()
    first.fit(np.ones((2, 2)), np.array([0, 0]), np.array([0, 0]))
    metrics = evaluate_benchmark_predictions(
        human_current=np.array([0, 1]), predicted_current=np.array([0, 0]),
        human_anticipated=np.array([0, 2]), predicted_anticipated=np.array([0, 0]),
        session_indices=np.array([1000, 1000]),
    )
    original_to_parquet = pd.DataFrame.to_parquet
    with monkeypatch.context() as patcher:
        patcher.setattr(
            pd.DataFrame,
            "to_parquet",
            lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("simulated crash")),
        )
        with pytest.raises(RuntimeError, match="simulated crash"):
            save_benchmark_run(
                paths, identity, model=first, predictions=_predictions(), metrics=metrics,
                model_configuration={"model_name": "majority_class", "adapter": "native"},
                dataset_description={"name": "synthetic"},
            )
    assert original_to_parquet is pd.DataFrame.to_parquet
    resumed, state = prepare_benchmark_run(
        tmp_path, identity, resume=True, execution_environment=environment
    )
    assert state == "resume"
    different = MajorityClassBaseline()
    different.fit(np.ones((2, 2)), np.array([1, 1]), np.array([1, 1]))
    with pytest.raises(ValueError, match="does not match"):
        save_benchmark_run(
            resumed, identity, model=different, predictions=_predictions(), metrics=metrics,
            model_configuration={"model_name": "majority_class", "adapter": "native"},
            dataset_description={"name": "synthetic"},
        )

    restored = MajorityClassBaseline.load(paths.model)
    save_benchmark_run(
        resumed, identity, model=restored, predictions=_predictions(), metrics=metrics,
        model_configuration={"model_name": "majority_class", "adapter": "native"},
        dataset_description={"name": "synthetic"},
    )
    assert load_benchmark_run(paths.directory)[0]["state"] == "complete"


def test_partial_resume_rejects_incompatible_environment(tmp_path) -> None:
    """Scientific identity stays stable while unsafe binary resume remains blocked."""

    identity = _identity()
    environment = resume_environment_fingerprint(device="cpu")
    prepare_benchmark_run(tmp_path, identity, execution_environment=environment)
    incompatible = {**environment, "python": "incompatible-test-version"}
    with pytest.raises(ValueError, match="incompatible execution environment"):
        prepare_benchmark_run(
            tmp_path,
            identity,
            resume=True,
            execution_environment=incompatible,
        )
