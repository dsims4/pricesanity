"""Benchmark integrity boundaries and crash recovery using synthetic sessions."""

from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pricesanity.benchmark.analysis import (
    classification_metrics,
    cluster_bootstrap_pooled_f1,
)
from pricesanity.benchmark.execution import BenchmarkExecutor
from pricesanity.benchmark.protocol import BenchmarkTrack
from pricesanity.benchmark.registry import build_model
from pricesanity.benchmark.search_spaces import track_search_space
from pricesanity.benchmark.snapshot import load_benchmark_snapshot
from pricesanity.models.protocol import PredictionContext
from test_benchmark_execution import _tiny_study


def small_study(tmp_path):
    """Build the smallest two-track study that can exercise freeze and recovery rules."""

    original = _tiny_study(tmp_path)
    return BenchmarkExecutor(
        snapshot=original.snapshot,
        config=replace(
            original.config,
            model_tuning_budgets={"logistic_regression": 1},
        ),
        search_spaces=original.search_spaces,
        study_root=tmp_path / "small",
    )


def select_and_freeze(executor):
    """Complete both declared tracks so tests may legally reach the final path."""

    for track in BenchmarkTrack:
        executor.tune_model("logistic_regression", track=track)
    executor.freeze_development()


def test_histogram_boosting_never_randomly_validates():
    model = build_model("gradient_boosting", random_seed=42)
    assert model.estimator_factory().early_stopping is False


def test_development_does_not_open_holdout_bytes(tmp_path, monkeypatch):
    original = _tiny_study(tmp_path)
    real_open = Path.open

    # Guard both ordinary file reads and pandas reads because either route would violate the
    # physical holdout boundary even if no holdout values reached model fitting.
    def guarded(path, *args, **kwargs):
        assert path.name != "sealed_holdout.parquet"
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded)
    read_parquet = pd.read_parquet

    def guarded_parquet(path, *args, **kwargs):
        assert Path(path).name != "sealed_holdout.parquet"
        return read_parquet(path, *args, **kwargs)

    monkeypatch.setattr(pd, "read_parquet", guarded_parquet)
    snapshot = load_benchmark_snapshot(original.snapshot.directory)
    executor = BenchmarkExecutor(
        snapshot=snapshot,
        config=original.config,
        search_spaces=original.search_spaces,
        study_root=original.paths.root,
    )

    assert len(executor.sessions) == 8
    executor.tune_model(
        "logistic_regression",
        track=BenchmarkTrack.CONTROLLED,
    )
    with pytest.raises(PermissionError, match="global development freeze"):
        executor.run_final(
            "logistic_regression",
            track=BenchmarkTrack.CONTROLLED,
            confirm_final_holdout=True,
        )


def test_global_freeze_requires_every_track_and_blocks_mutation(tmp_path):
    executor = small_study(tmp_path)
    executor.tune_model(
        "logistic_regression",
        track=BenchmarkTrack.CONTROLLED,
    )
    with pytest.raises(ValueError, match="No frozen selected"):
        executor.freeze_development()

    executor.tune_model(
        "logistic_regression",
        track=BenchmarkTrack.BEST_OF_FAMILY,
    )
    executor.freeze_development()
    with pytest.raises(PermissionError, match="frozen"):
        executor.tune_model(
            "logistic_regression",
            track=BenchmarkTrack.CONTROLLED,
        )

    # Changing a search space after freezing must invalidate identity rather than reuse the
    # already selected model under a different declared experiment.
    executor.search_spaces["logistic_regression"]["C"] = {
        "type": "fixed",
        "value": 2.0,
    }
    with pytest.raises(ValueError, match="identity"):
        executor.run_final(
            "logistic_regression",
            track=BenchmarkTrack.CONTROLLED,
            confirm_final_holdout=True,
        )


@pytest.mark.parametrize("mutation", ["space", "protocol", "seed"])
def test_optuna_resume_rejects_changed_identity(tmp_path, mutation):
    executor = small_study(tmp_path)
    executor.tune_model(
        "logistic_regression",
        track=BenchmarkTrack.CONTROLLED,
    )
    if mutation == "space":
        executor.search_spaces["logistic_regression"]["C"] = {
            "type": "fixed",
            "value": 2.0,
        }
    elif mutation == "protocol":
        executor.config = replace(
            executor.config,
            inference_timing_repetitions=2,
        )
    else:
        executor.config = replace(executor.config, tuning_seed=7)

    with pytest.raises(ValueError, match="Optuna study identity"):
        executor.tune_model(
            "logistic_regression",
            track=BenchmarkTrack.CONTROLLED,
        )


def test_tabular_padding_excluded_and_indicators_unchanged():
    mask = np.array([[False, True], [True, True], [False, True]])
    real = np.array(
        [
            [[999.0] * 4, [2.0] * 4],
            [[4.0] * 4, [6.0] * 4],
            [[999.0] * 4, [10.0] * 4],
        ],
        dtype=np.float32,
    )
    features = np.concatenate(
        [real.reshape(3, -1), mask.astype(np.float32)],
        axis=1,
    )
    context = PredictionContext(
        session_indices=np.zeros(3),
        valid_history_mask=mask,
    )
    model = build_model("gaussian_naive_bayes", random_seed=42)
    model.fit(features, np.arange(3), np.arange(3), context=context)

    # The 999 sentinels occupy padded positions. A correct fitted distribution sees only the
    # real candles, while the appended 0/1 validity indicators remain unstandardized.
    np.testing.assert_array_equal(model._standardizer.mean[:4], 4.0)
    np.testing.assert_array_equal(model._standardizer.mean[4:], 6.0)
    transformed = model._prepare(features, context)[0]
    np.testing.assert_array_equal(transformed[:, -2:], mask)
    np.testing.assert_array_equal(transformed[[0, 2], :4], 0.0)


def test_context_cache_keeps_exact_population_and_matches_rebuild(tmp_path):
    from pricesanity.features.causal_window import build_causal_windows

    executor = small_study(tmp_path)
    training, evaluation = executor._fold_sessions(
        executor.plan.chronological_validation_folds[0]
    )
    results = [
        executor._corpora(
            "logistic_regression",
            {"context_length": context_length},
            BenchmarkTrack.BEST_OF_FAMILY,
            training,
            evaluation,
        )[2]
        for context_length in (16, 32, 64)
    ]

    # Context length may alter input history but never the candles being scored.
    assert (
        results[0].candlestick_ids
        == results[1].candlestick_ids
        == results[2].candlestick_ids
    )
    assert len(results[0].candlestick_ids) == 5

    direct = build_causal_windows(
        evaluation,
        feature_columns=executor.config.feature_columns,
        window_length=16,
        allow_partial_history=True,
        target_start_position=1,
    )
    np.testing.assert_array_equal(results[0].features, direct.features)
    np.testing.assert_array_equal(
        results[0].valid_history_mask,
        direct.valid_history_mask,
    )
    assert "context_length" not in track_search_space(
        "gru",
        {"context_length": {"type": "fixed", "value": 64}},
        "controlled",
    )
    assert track_search_space("knn", {}, "best_of_family")["context_length"][
        "values"
    ] == [16, 32, 64]


@pytest.mark.parametrize("crash_stage", ["model", "predictions", "metrics"])
def test_crash_resume_preserves_timing_without_repeating_work(
    tmp_path,
    monkeypatch,
    crash_stage,
):
    import pricesanity.benchmark.artifacts as artifacts
    import pricesanity.benchmark.runner as runner
    from pricesanity.models.sklearn_adapter import SklearnDualHeadAdapter

    executor = small_study(tmp_path)
    select_and_freeze(executor)
    real_write = artifacts._write_json_atomic
    real_eval = runner.evaluate_fitted_model

    def crash_write(path, *args, **kwargs):
        crashes_before_metrics = (
            crash_stage == "predictions" and path.name == "metrics.json"
        )
        crashes_before_metadata = (
            crash_stage == "metrics"
            and path.name == "benchmark_metadata.json"
        )
        if crashes_before_metrics or crashes_before_metadata:
            raise RuntimeError("simulated crash")
        return real_write(path, *args, **kwargs)

    def crash_eval(*args, **kwargs):
        raise RuntimeError("simulated crash")

    monkeypatch.setattr(artifacts, "_write_json_atomic", crash_write)
    if crash_stage == "model":
        monkeypatch.setattr(runner, "evaluate_fitted_model", crash_eval)
    with pytest.raises(RuntimeError, match="simulated crash"):
        executor.run_final(
            "logistic_regression",
            track=BenchmarkTrack.CONTROLLED,
            confirm_final_holdout=True,
        )

    progress_path = next(executor.paths.runs.rglob("run_progress.json"))
    evidence = json.loads(progress_path.read_text())
    monkeypatch.setattr(artifacts, "_write_json_atomic", real_write)
    monkeypatch.setattr(runner, "evaluate_fitted_model", real_eval)

    # Any fit after the crash would create new stochastic state and invalidate the evidence
    # already checkpointed with the partial run.
    def forbidden(*args, **kwargs):
        raise AssertionError("completed work repeated")

    monkeypatch.setattr(SklearnDualHeadAdapter, "fit", forbidden)
    if crash_stage != "model":
        monkeypatch.setattr(SklearnDualHeadAdapter, "predict_output", forbidden)
    paths = executor.run_final(
        "logistic_regression",
        track=BenchmarkTrack.CONTROLLED,
        confirm_final_holdout=True,
    )
    metrics = json.loads((paths[0] / "metrics.json").read_text())["efficiency"]
    assert (
        metrics["training_seconds"]
        == evidence["training_evidence"]["training_seconds"]
        > 0
    )
    if crash_stage != "model":
        assert (
            metrics["inference_seconds"]
            == evidence["efficiency"]["inference_seconds"]
            > 0
        )


def test_bootstrap_matches_reference_draws():
    rng = np.random.default_rng(55)
    labels = np.array(["bull", "bear", "range"])
    sessions = np.repeat([2, 8, 13], [5, 8, 3])
    sample_count = len(sessions)
    predictions = pd.DataFrame(
        {
            "candlestick_id": np.arange(sample_count).astype(str),
            "session_index": sessions,
            **{
                f"{kind}_{head}_regime": labels[
                    rng.integers(0, 3, sample_count)
                ]
                for kind in ("human", "predicted")
                for head in ("current", "anticipated")
            },
        }
    )
    optimized = cluster_bootstrap_pooled_f1(
        predictions,
        repetitions=100,
        random_seed=12,
    )

    # Rebuild the slower row-level reference with the same session draws to prove the
    # confusion-matrix optimization preserves the exact cluster-bootstrap distribution.
    rng = np.random.default_rng(12)
    values = []
    mapping = {"bull": 0, "bear": 1, "range": 2}
    for _ in range(100):
        draw = rng.choice([2, 8, 13], 3, replace=True)
        rows = pd.concat(
            [predictions[predictions.session_index == session] for session in draw]
        )
        values.append(
            classification_metrics(
                rows.human_current_regime.map(mapping).to_numpy(),
                rows.predicted_current_regime.map(mapping).to_numpy(),
            ).macro_f1
        )

    np.testing.assert_allclose(
        [optimized.current.lower, optimized.current.upper],
        np.quantile(values, [0.025, 0.975]),
        rtol=0,
        atol=1e-15,
    )


def test_partial_resume_rejects_changed_source(tmp_path, monkeypatch):
    import pricesanity.benchmark.artifacts as artifacts
    from test_benchmark_artifacts import _identity

    artifacts.prepare_benchmark_run(tmp_path, _identity())
    original = artifacts.source_identity()
    monkeypatch.setattr(
        artifacts,
        "source_identity",
        lambda: {**original, "source_sha256": "changed"},
    )
    with pytest.raises(ValueError, match="incompatible execution environment"):
        artifacts.prepare_benchmark_run(tmp_path, _identity(), resume=True)


def test_seed_aggregation_rejects_same_count_with_wrong_seeds():
    from pricesanity.benchmark.aggregation import aggregate_seed_results

    rows = pd.DataFrame(
        [
            {
                "track": "controlled",
                "model_name": "mlp",
                "model_configuration_sha256": "x",
                "representation_sha256": "y",
                "test_session_ids_sha256": "z",
                "seed": seed,
                "current_macro_f1": 0.3,
                "anticipated_macro_f1": 0.2,
                "declared_final_seeds": [11, 12],
            }
            for seed in (11, 99)
        ]
    )
    with pytest.raises(ValueError, match="declared final seed set"):
        aggregate_seed_results(
            rows,
            stochastic_models=["mlp"],
            expected_stochastic_seed_count=2,
        )


def test_anticipated_transition_events_are_independent_of_current():
    from pricesanity.benchmark.metrics import evaluate_benchmark_predictions

    metrics = evaluate_benchmark_predictions(
        human_current=np.array([0, 0, 0, 0]),
        predicted_current=np.array([0, 0, 0, 0]),
        human_anticipated=np.array([0, 0, 1, 1]),
        predicted_anticipated=np.array([0, 0, 0, 1]),
        session_indices=np.zeros(4),
    )
    assert metrics.transitions.exact.human_transition_count == 0
    assert metrics.anticipated_transitions.exact.human_transition_count == 1
    assert metrics.anticipated_transitions.exact.matched_transition_count == 0
    assert (
        metrics.anticipated_transitions.within_one_candle.matched_transition_count
        == 1
    )
    assert (
        metrics.anticipated_transition_neighborhoods.exact_anticipated.support
        == 1
    )


def test_report_before_final_and_single_session_uncertainty_are_honest(tmp_path):
    from pricesanity.benchmark.notebook_reports import (
        error_analysis_tables,
        final_report_tables,
    )

    executor = small_study(tmp_path)
    executor.tune_model(
        "logistic_regression",
        track=BenchmarkTrack.CONTROLLED,
    )
    tables = final_report_tables(executor.paths.root)
    assert tables["leaderboard"].empty
    assert tables["final_holdout"].empty

    run = next(
        (executor.paths.tuning / "fold_runs").rglob("benchmark_metadata.json")
    ).parent
    analysis = error_analysis_tables(run)
    assert analysis["pooled_session_bootstrap"] is None


def test_explicit_unavailable_accelerator_is_rejected(tmp_path, monkeypatch):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    original = _tiny_study(tmp_path)
    with pytest.raises(ValueError, match="unavailable"):
        BenchmarkExecutor(
            snapshot=original.snapshot,
            config=original.config,
            search_spaces=original.search_spaces,
            study_root=original.paths.root,
            device="cuda",
        )


def test_snapshot_api_cannot_use_per_model_winner_as_global_seal(tmp_path):
    from pricesanity.benchmark.snapshot import load_sealed_holdout

    executor = small_study(tmp_path)
    executor.tune_model(
        "logistic_regression",
        track=BenchmarkTrack.CONTROLLED,
    )
    winner = executor.paths.selected / "controlled/logistic_regression.json"
    with pytest.raises(PermissionError, match="global development seal"):
        load_sealed_holdout(
            executor.snapshot,
            seal_path=winner,
            confirm_final_holdout=True,
        )


def test_paired_confusion_bootstrap_matches_reference_rows():
    from pricesanity.benchmark.analysis import paired_cluster_bootstrap_difference

    rng = np.random.default_rng(22)
    labels = np.array(["bull", "bear", "range"])
    sessions = np.repeat([2, 8, 13], [5, 8, 3])
    sample_count = len(sessions)
    first = pd.DataFrame(
        {
            "candlestick_id": np.arange(sample_count).astype(str),
            "session_index": sessions,
            **{
                f"{kind}_{head}_regime": labels[
                    rng.integers(0, 3, sample_count)
                ]
                for kind in ("human", "predicted")
                for head in ("current", "anticipated")
            },
        }
    )
    second = first.copy()
    for head in ("current", "anticipated"):
        second[f"predicted_{head}_regime"] = labels[
            rng.integers(0, 3, sample_count)
        ]
    optimized = paired_cluster_bootstrap_difference(
        first,
        second,
        repetitions=100,
        random_seed=17,
    )

    rng = np.random.default_rng(17)
    differences = []
    mapping = {"bull": 0, "bear": 1, "range": 2}
    for _ in range(100):
        draw = rng.choice([2, 8, 13], 3, replace=True)
        scores = []
        for frame in (first, second):
            rows = pd.concat(
                [frame[frame.session_index == session] for session in draw]
            )
            head_scores = [
                classification_metrics(
                    rows[f"human_{head}_regime"].map(mapping).to_numpy(),
                    rows[f"predicted_{head}_regime"].map(mapping).to_numpy(),
                ).macro_f1
                for head in ("current", "anticipated")
            ]
            scores.append(np.mean(head_scores))
        differences.append(scores[0] - scores[1])

    np.testing.assert_allclose(
        [optimized.lower, optimized.upper],
        np.quantile(differences, [0.025, 0.975]),
        rtol=0,
        atol=1e-15,
    )
