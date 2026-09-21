import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pricesanity.benchmark.artifacts import (
    BenchmarkRunIdentity,
    canonical_sha256,
    prepare_benchmark_run,
    save_benchmark_run,
)
from pricesanity.benchmark.metrics import evaluate_benchmark_predictions
from pricesanity.config import load_config
from pricesanity.data.identifiers import build_candlestick_id
from pricesanity.gui.model_comparison import (
    _ABSessionCanvas,
    ComparisonRun,
    ModelComparisonWindow,
    _format_run_details,
    format_uncertainty_values,
    load_aligned_comparison_predictions,
    load_comparison_runs,
)
from pricesanity.gui.test_results import load_test_run
from pricesanity.models.baselines import MajorityClassBaseline


def test_comparison_gui_has_safe_empty_state(tmp_path, qt_application) -> None:
    """The model-comparison mode remains usable before benchmark models exist."""

    window = ModelComparisonWindow(load_comparison_runs(tmp_path))
    try:
        assert "No completed benchmark artifacts" in window.status_label.text()
        assert not window.model_a.isEnabled()
    finally:
        window.close()


def test_session_results_loads_model_agnostic_benchmark_artifact(tmp_path) -> None:
    """Saved benchmark predictions enter the existing chart without inference."""

    config = load_config("configs/default.yaml")
    timestamps = pd.date_range("2026-01-02T14:30:00Z", periods=2, freq="5min")
    ids = [
        build_candlestick_id("ES.v.0", timestamp, "5min")
        for timestamp in timestamps
    ]
    ohlc_path = tmp_path / "ohlc.parquet"
    pd.DataFrame({
        "candlestick_id": ids,
        "ts_event": timestamps,
        "instrument": "ES.v.0",
        "open": [100.0, 101.0],
        "high": [102.0, 103.0],
        "low": [99.0, 100.0],
        "close": [101.0, 102.0],
    }).to_parquet(ohlc_path, index=False)

    identity = BenchmarkRunIdentity(
        track="controlled", model_name="majority_class", run_name="fold_001",
        seed=42, feature_representation="tabular_16x4", window_length=16,
        feature_columns=("open_gap", "body", "high_from_close", "low_from_close"),
        training_session_range=(0, 1), validation_session_ranges=((1, 2),),
        test_session_range=(2, 3),
        model_configuration_sha256=canonical_sha256(
            {"model_name": "majority_class", "adapter": "native"}
        ),
        representation_sha256=canonical_sha256({"name": "tabular_16x4"}),
        training_session_ids_sha256=canonical_sha256([0]),
        validation_session_ids_sha256=canonical_sha256([1]),
        test_session_ids_sha256=canonical_sha256([2]),
        annotation_snapshot_sha256=canonical_sha256("annotations"),
        normalized_dataset_sha256=canonical_sha256("normalized"),
        protocol_sha256=canonical_sha256("protocol-v1"),
        search_space_sha256=canonical_sha256("none"),
        label_mapping_sha256=canonical_sha256(["bull", "bear", "range"]),
    )
    paths, _ = prepare_benchmark_run(tmp_path / "benchmark", identity)
    model = MajorityClassBaseline()
    model.fit(np.ones((2, 64), dtype=np.float32), np.array([0, 1]), np.array([0, 2]))
    predictions = pd.DataFrame({
        "candlestick_id": ids,
        "timestamp": timestamps,
        "session_date": [pd.Timestamp("2026-01-02").date()] * 2,
        "session_index": [2, 2],
        "candle_position": [15, 16],
        "uncertainty_kind": ["deterministic_distribution"] * 2,
        "predicted_current_regime": ["bull", "bull"],
        "predicted_anticipated_regime": ["bull", "bull"],
        "current_probability_bull": [1.0, 1.0],
        "current_probability_bear": [0.0, 0.0],
        "current_probability_range": [0.0, 0.0],
        "anticipated_probability_bull": [1.0, 1.0],
        "anticipated_probability_bear": [0.0, 0.0],
        "anticipated_probability_range": [0.0, 0.0],
        "current_score_bull": [np.nan, np.nan],
        "current_score_bear": [np.nan, np.nan],
        "current_score_range": [np.nan, np.nan],
        "anticipated_score_bull": [np.nan, np.nan],
        "anticipated_score_bear": [np.nan, np.nan],
        "anticipated_score_range": [np.nan, np.nan],
        "human_current_regime": ["bull", "bear"],
        "human_anticipated_regime": ["bull", "range"],
    })
    metrics = evaluate_benchmark_predictions(
        human_current=np.array([0, 1]), predicted_current=np.array([0, 0]),
        human_anticipated=np.array([0, 2]), predicted_anticipated=np.array([0, 0]),
        session_indices=np.array([2, 2]),
    )
    save_benchmark_run(
        paths, identity, model=model, predictions=predictions, metrics=metrics,
        model_configuration={"model_name": "majority_class", "adapter": "native"},
        dataset_description={
            "instrument": "ES.v.0", "target_interval": "5min",
            "session_timezone": "America/New_York",
            "candlestick_path": str(ohlc_path),
        },
    )

    loaded = load_test_run(paths.directory, None, config=config)
    assert loaded.metadata["artifact_kind"] == "benchmark"
    assert loaded.metadata["model_name"] == "majority_class"
    assert loaded.candlesticks["candlestick_id"].tolist() == ids
    assert len(load_comparison_runs(tmp_path / "benchmark")) == 1


def test_ab_alignment_rejects_different_candles(monkeypatch, tmp_path) -> None:
    """A/B comparison cannot hide mismatches with an inner join."""

    first = pd.DataFrame({
        "candlestick_id": ["a"], "timestamp": pd.to_datetime(["2026-01-01"], utc=True),
        "session_date": [pd.Timestamp("2026-01-01").date()], "session_index": [1],
        "candle_position": [15], "human_current_regime": ["bull"],
        "human_anticipated_regime": ["range"],
    })
    second = first.copy()
    second["candlestick_id"] = ["b"]
    calls = iter([({}, first, {}), ({}, second, {})])
    monkeypatch.setattr(
        "pricesanity.gui.model_comparison.load_benchmark_run",
        lambda path: next(calls),
    )
    run_a = ComparisonRun("A", tmp_path / "a", {}, {})
    run_b = ComparisonRun("B", tmp_path / "b", {}, {})
    with pytest.raises(ValueError, match="exact ordered"):
        load_aligned_comparison_predictions(run_a, run_b)


def test_uncertainty_display_distinguishes_scores_from_probabilities() -> None:
    """The GUI must not present uncalibrated margins as probability estimates."""

    probability_row = {
        "uncertainty_kind": "probability_estimate",
        "current_probability_bull": 0.7,
        "current_probability_bear": 0.1,
        "current_probability_range": 0.2,
    }
    score_row = {
        "uncertainty_kind": "uncalibrated_decision_score",
        "current_score_bull": 1.2,
        "current_score_bear": -0.4,
        "current_score_range": 0.3,
    }
    assert format_uncertainty_values(probability_row, head="current").startswith(
        "Probability estimate"
    )
    assert format_uncertainty_values(score_row, head="current").startswith(
        "Uncalibrated score"
    )


def test_run_details_display_current_and_anticipated_transition_diagnostics() -> None:
    """The explorer reports each persisted head with the same exact/±1/±2 semantics."""

    metrics = evaluate_benchmark_predictions(
        human_current=np.array([0, 0, 1, 1, 2, 2]),
        predicted_current=np.array([0, 0, 0, 1, 2, 2]),
        human_anticipated=np.array([0, 1, 1, 2, 2, 0]),
        predicted_anticipated=np.array([0, 0, 1, 1, 2, 0]),
        session_indices=np.zeros(6, dtype=np.int64),
    ).to_dict()
    details = _format_run_details(
        ComparisonRun("diagnostic run", Path("/unused"), {}, metrics)
    )

    assert "Current exact transitions: 1" in details
    assert "Current transitions within ±1 candle: 2" in details
    assert "Current transitions within ±2 candles: 2" in details
    assert "Anticipated exact transitions: 1" in details
    assert "Anticipated transitions within ±1 candle: 3" in details
    assert "Anticipated transitions within ±2 candles: 3" in details


def test_ab_canvas_draws_exact_aligned_regimes_uncertainty_and_ohlc(
    qt_application,
) -> None:
    """One aligned population should drive all three linked comparison panels."""

    timestamps = pd.date_range("2026-01-02T14:30:00Z", periods=3, freq="5min")
    base = pd.DataFrame({
        "candlestick_id": ["a", "b", "c"], "timestamp": timestamps,
        "session_date": [pd.Timestamp("2026-01-02").date()] * 3,
        "session_index": [1] * 3, "candle_position": [1, 2, 3],
        "uncertainty_kind": ["probability_estimate"] * 3,
        "human_current_regime": ["bull", "range", "bear"],
        "human_anticipated_regime": ["range", "bear", "bear"],
        "predicted_current_regime": ["bull", "bull", "bear"],
        "predicted_anticipated_regime": ["range", "range", "bear"],
    })
    for head in ("current", "anticipated"):
        base[f"{head}_probability_bull"] = [0.7, 0.4, 0.1]
        base[f"{head}_probability_bear"] = [0.1, 0.2, 0.8]
        base[f"{head}_probability_range"] = [0.2, 0.4, 0.1]
    second = base.copy()
    second["predicted_current_regime"] = ["range", "range", "bear"]
    ohlc = pd.DataFrame({
        "timestamp": timestamps, "open": [100, 101, 100], "high": [102, 102, 101],
        "low": [99, 99, 98], "close": [101, 100, 99],
    })
    canvas = _ABSessionCanvas()
    try:
        canvas.show_session(base, second, first_name="A", second_name="B", ohlc=ohlc)
        assert len(canvas.figure.axes) == 3
        assert len(canvas.figure.axes[0].patches) == 3
        assert len(canvas.figure.axes[1].images) == 1
        assert len(canvas.figure.axes[2].lines) == 4
    finally:
        canvas.close()


def test_gui_rejects_incomplete_stochastic_seed_set(qt_application) -> None:
    """A partial stochastic result must remain visible but stay off the leaderboard."""

    classification = {
        "accuracy": 0.5, "macro_f1": 0.4,
        "per_class": {
            regime: {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 1}
            for regime in ("bull", "bear", "range")
        },
        "confusion_matrix": [[1, 0, 0], [0, 0, 1], [0, 1, 0]],
    }
    transition = {
        key: {"matched_transition_count": 0}
        for key in ("exact", "within_one_candle", "within_two_candles")
    }
    run = ComparisonRun(
        "MLP seed 1",
        Path("/does/not/exist"),
        {"identity": {
            "track": "controlled", "model_name": "mlp",
            "run_name": "final_seed_1", "seed": 1,
            "model_configuration_sha256": "a", "representation_sha256": "b",
            "test_session_ids_sha256": "c",
        }, "model_configuration": {"parameters": {"hidden_width": 4}}},
        {"current": classification, "anticipated": classification,
         "mean_head_macro_f1": 0.4, "transitions": transition, "efficiency": {}},
    )
    window = ModelComparisonWindow((run,))
    try:
        assert "Incomplete final seed set" in window.status_label.text()
        assert window.leaderboard.rowCount() == 0
    finally:
        window.close()
