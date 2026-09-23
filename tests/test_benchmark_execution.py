from dataclasses import replace
from datetime import date, timedelta
import json

import numpy as np
import pandas as pd
import pytest

from pricesanity.benchmark.execution import BenchmarkExecutor
from pricesanity.benchmark.protocol import (
    BenchmarkTrack, load_benchmark_config, resolve_benchmark_config,
)
from pricesanity.benchmark.report import collect_aggregated_runs
from pricesanity.benchmark.snapshot import freeze_benchmark_snapshot_from_sessions


MODEL_NAMES = (
    "logistic_regression", "random_forest", "mlp", "tcn", "gru", "transformer",
)


def _session(session_index: int, length: int = 6) -> pd.DataFrame:
    """Give every synthetic fold all three classes and distinct chronological identities."""

    session_date = date(2026, 1, 5) + timedelta(days=session_index)
    positions = np.arange(length, dtype=np.float32)
    return pd.DataFrame({
        "session_date": [session_date] * length,
        "candlestick_id": [f"{session_index}-{position}" for position in range(length)],
        "ts_event": pd.date_range(
            pd.Timestamp(session_date, tz="UTC") + pd.Timedelta(hours=14, minutes=30),
            periods=length,
            freq="5min",
        ),
        "open_gap": positions + session_index / 10,
        "body": np.sin(positions + session_index).astype(np.float32),
        "high_from_close": positions / 10 + 0.5,
        "low_from_close": -positions / 10 - 0.5,
        "current_target": (np.arange(length) + session_index) % 3,
        "anticipated_target": (np.arange(length) + session_index + 1) % 3,
    })


def _fixed(value):
    return {"type": "fixed", "value": value}


def _tiny_study(tmp_path):
    """Build the smallest study that still exercises every benchmark lifecycle stage."""

    sessions = tuple(_session(index) for index in range(9))
    base = load_benchmark_config("configs/benchmark/default.yaml")
    incumbent = {
        "context_length": 3,
        "model_dimension": 4,
        "attention_head_count": 1,
        "layer_count": 1,
        "feedforward_dimension": 8,
        "dropout": 0.0,
        "learning_rate": 1e-3,
        "epochs": 1,
        "batch_size": 4,
        "weight_decay": 0.0,
    }
    # Shrink the corpus while preserving the real separation between tuning validation,
    # fixed learning-curve evaluation, and the sealed final holdout.
    rules = replace(
        base.rules,
        initial_training_fraction=0.40,
        learning_evaluation_fraction=0.25,
        fold_count=2,
        learning_curve_fractions=(0.34, 0.67, 1.0),
        minimum_training_sessions=1,
        minimum_validation_sessions=1,
        minimum_test_sessions=1,
        minimum_curve_sessions=1,
    )
    config = resolve_benchmark_config(
        9,
        replace(
            base,
            rules=rules,
            window_length=3,
            controlled_first_scored_candle_position=2,
            best_of_family_first_scored_candle_position=1,
            final_seeds=(11, 12),
            tuning_seed=5,
            inference_timing_repetitions=1,
            transformer_incumbent=incumbent,
            model_tuning_budgets={name: 1 for name in MODEL_NAMES},
        ),
    )
    snapshot = freeze_benchmark_snapshot_from_sessions(
        sessions,
        output_directory=tmp_path / "study" / "snapshot",
        expected_session_count=9,
        development_session_count=config.development_session_count,
        source_identities={"kind": "tiny_acceptance"},
    )
    # Fixed candidate spaces exercise persistence and resume cheaply; they do not replace
    # production search budgets or claim meaningful model-performance evidence.
    sequence_training = {
        "context_length": _fixed(3), "learning_rate": _fixed(1e-3),
        "epochs": _fixed(1), "batch_size": _fixed(4), "weight_decay": _fixed(0.0),
    }
    spaces = {
        "logistic_regression": {"C": _fixed(1.0), "class_weight": _fixed(None)},
        "random_forest": {
            "n_estimators": _fixed(5), "max_depth": _fixed(2),
            "min_samples_leaf": _fixed(1),
        },
        "mlp": {
            "hidden_width": _fixed(4), "hidden_layers": _fixed(1),
            "learning_rate_init": _fixed(1e-3), "max_iter": _fixed(2),
        },
        "tcn": {
            **sequence_training, "channel_width": _fixed(4),
            "kernel_size": _fixed(2), "layer_count": _fixed(1),
        },
        "gru": {
            **sequence_training, "hidden_size": _fixed(4), "layer_count": _fixed(1),
        },
        "transformer": {name: _fixed(value) for name, value in incumbent.items()},
    }
    executor = BenchmarkExecutor(
        snapshot=snapshot,
        config=config,
        search_spaces=spaces,
        study_root=tmp_path / "study",
        device="cpu",
    )
    return executor


def test_tiny_multimodel_executor_tunes_finalizes_and_resumes(tmp_path) -> None:
    """Prove the launch-time machine without touching the real annotated corpus."""

    executor = _tiny_study(tmp_path)
    for track in BenchmarkTrack:
        for model_name in MODEL_NAMES:
            selected = executor.tune_model(model_name, track=track)
            assert selected["all_development_folds"] is True
            assert selected["fold_count"] == 2
            assert selected["tuning_seed"] == 5

    # Resuming the persistent study does not add a second completed trial.
    first_selection = executor.tune_model(
        "transformer", track=BenchmarkTrack.CONTROLLED
    )
    assert first_selection["optuna_trial_number"] == 0
    selected_path = (
        tmp_path / "study" / "selected" / "controlled" / "transformer.json"
    )
    assert json.loads(selected_path.read_text())["parameters"]["model_dimension"] == 4

    for track in BenchmarkTrack:
        executor.run_learning_curve("logistic_regression", track=track)
    executor.freeze_development()
    completed = []
    for track in BenchmarkTrack:
        for model_name in MODEL_NAMES:
            paths = executor.run_final(
                model_name, track=track, confirm_final_holdout=True,
            )
            completed.extend(paths)
    mtimes = {path: (path / "benchmark_metadata.json").stat().st_mtime_ns for path in completed}
    for model_name in MODEL_NAMES:
        executor.run_final(
            model_name,
            track=BenchmarkTrack.CONTROLLED,
            confirm_final_holdout=True,
        )
    assert mtimes == {
        path: (path / "benchmark_metadata.json").stat().st_mtime_ns for path in completed
    }

    leaderboard = collect_aggregated_runs(
        tmp_path / "study" / "runs", expected_stochastic_seed_count=2
    )
    assert set(leaderboard["model_name"]) == set(MODEL_NAMES)
    assert len(leaderboard) == len(MODEL_NAMES) * 2
    from pricesanity.benchmark.notebook_reports import final_report_tables
    tables = final_report_tables(executor.paths.root, expected_stochastic_seed_count=2)
    assert not tables["controlled"].empty
    assert not tables["best_of_family"].empty
    assert not tables["learning_curves"].empty


def test_focused_fold_cannot_freeze_or_unlock_final(tmp_path) -> None:
    """A diagnostic subset cannot become the all-fold selection used for final testing."""

    executor = _tiny_study(tmp_path)
    result = executor.tune_model(
        "logistic_regression",
        track=BenchmarkTrack.CONTROLLED,
        fold_indices=(0,),
    )
    assert result["state"] == "diagnostic"
    assert not (
        tmp_path / "study" / "selected" / "controlled" / "logistic_regression.json"
    ).exists()
    with pytest.raises(ValueError, match="No frozen selected configuration"):
        executor.run_final(
            "logistic_regression",
            track=BenchmarkTrack.CONTROLLED,
            confirm_final_holdout=True,
        )


def test_final_holdout_needs_deliberate_unlock(tmp_path) -> None:
    """Completing development selection alone does not authorize holdout access."""

    executor = _tiny_study(tmp_path)
    executor.tune_model("logistic_regression", track=BenchmarkTrack.CONTROLLED)
    with pytest.raises(PermissionError, match="locked"):
        executor.run_final(
            "logistic_regression",
            track=BenchmarkTrack.CONTROLLED,
            confirm_final_holdout=False,
        )
