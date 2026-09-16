from dataclasses import replace

import pytest

from pricesanity.benchmark.learning_curve import plan_learning_curve
from pricesanity.benchmark.protocol import load_benchmark_config, plan_benchmark


def test_benchmark_plan_keeps_final_holdout_out_of_development() -> None:
    """Configured tuning folds end before the separately locked final history."""

    config = load_benchmark_config("configs/benchmark/default.yaml")
    plan = plan_benchmark(2690, config)

    assert list(plan.development_indices()) == list(range(2190))
    assert plan.final_holdout.start_index == 2190
    assert plan.final_holdout.end_index == 2690
    assert all(
        fold.validation.end_index <= plan.development.end_index
        for fold in plan.chronological_validation_folds
    )
    with pytest.raises(PermissionError, match="unavailable"):
        plan.holdout_indices()
    assert list(plan.holdout_indices(final_evaluation=True)) == list(range(2190, 2690))


def test_benchmark_plan_refuses_unaccounted_sessions_and_holdout_folds() -> None:
    """Changing corpus size or leaking a validation fold requires deliberate config edits."""

    config = load_benchmark_config("configs/benchmark/default.yaml")
    with pytest.raises(ValueError, match="accounts for 2690"):
        plan_benchmark(2689, config)

    invalid = replace(
        config,
        chronological_validation_folds=((2190, 2190, 2290),),
    )
    with pytest.raises(ValueError, match="final holdout"):
        plan_benchmark(2690, invalid)

    with pytest.raises(ValueError):
        plan_learning_curve(
            (250, 2200), development_session_count=2190,
            evaluation_range=(2090, 2190),
        )


def test_learning_curve_uses_development_prefixes_only() -> None:
    """Every learning-curve point begins at the oldest development session."""

    points = plan_learning_curve(
        (250, 500, 1000, 2000),
        development_session_count=2190,
        evaluation_range=(2100, 2190),
    )
    assert [(point.training_start_index, point.training_end_index) for point in points] == [
        (0, 250), (0, 500), (0, 1000), (0, 2000)
    ]
    assert {
        (point.evaluation_start_index, point.evaluation_end_index)
        for point in points
    } == {(2100, 2190)}

    config = load_benchmark_config("configs/benchmark/default.yaml")
    assert config.learning_curve_evaluation_range[0] >= max(
        fold[2] for fold in config.chronological_validation_folds
    )
