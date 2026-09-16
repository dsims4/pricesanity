import numpy as np
import pandas as pd

from pricesanity.benchmark.runner import build_benchmark_prediction_frame, run_model_once
from pricesanity.features import FEATURE_COLUMNS, build_causal_windows
from pricesanity.models.baselines import MajorityClassBaseline


def _session(day: int, targets: list[int]) -> pd.DataFrame:
    length = len(targets)
    return pd.DataFrame({
        "session_date": [pd.Timestamp(2026, 1, day).date()] * length,
        "candlestick_id": [f"{day}-{index}" for index in range(length)],
        "ts_event": pd.date_range(
            f"2026-01-{day:02d}T14:30:00Z", periods=length, freq="5min"
        ),
        "open_gap": np.arange(length, dtype=np.float32),
        "body": 0.1,
        "high_from_close": 0.2,
        "low_from_close": -0.2,
        "current_target": targets,
        "anticipated_target": targets[::-1],
    })


def test_common_runner_fits_and_scores_both_heads() -> None:
    """One runner handles model timing, predictions, probabilities, and metrics."""

    training = build_causal_windows(
        [_session(2, [0] * 10 + [1] * 8)],
        feature_columns=FEATURE_COLUMNS,
        window_length=16,
    )
    evaluation = build_causal_windows(
        [_session(3, [0, 1, 2] * 6)],
        feature_columns=FEATURE_COLUMNS,
        window_length=16,
    )
    result = run_model_once(
        MajorityClassBaseline(),
        training=training,
        evaluation=evaluation,
        representation="tabular",
    )

    assert result.predictions.current.shape == (3,)
    assert result.probabilities.current.shape == (3, 3)
    assert result.metrics.current.support == 3
    assert result.metrics.efficiency is not None
    assert result.metrics.efficiency.training_seconds >= 0.0
    prediction_frame = build_benchmark_prediction_frame(evaluation, result)
    assert prediction_frame["candlestick_id"].tolist() == list(
        evaluation.candlestick_ids
    )
    assert prediction_frame["human_current_regime"].tolist() == ["bull", "bear", "range"]
