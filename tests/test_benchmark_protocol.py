from dataclasses import replace
from pathlib import Path

import pytest

from pricesanity.benchmark.learning_curve import plan_learning_curve
from pricesanity.benchmark.protocol import (
    load_benchmark_config, plan_benchmark, resolve_benchmark_config, minimum_session_count,
)


@pytest.mark.parametrize("count,development,test", [(550, 495, 55), (551, 495, 56),
                                                    (1000, 900, 100), (2690, 2421, 269)])
def test_dynamic_chronological_split_and_folds(count, development, test):
    config = resolve_benchmark_config(count, load_benchmark_config("configs/benchmark/default.yaml"))
    plan = plan_benchmark(count, config)
    assert list(plan.development_indices()) == list(range(development))
    assert plan.final_holdout.session_count == test
    with pytest.raises(PermissionError, match="unavailable"):
        plan.holdout_indices()
    holdout = list(plan.holdout_indices(final_evaluation=True))
    assert holdout == list(range(development, count))
    assert list(plan.development_indices()) + holdout == list(range(count))
    previous = 0
    for fold in plan.chronological_validation_folds:
        assert fold.training.start_index == 0
        assert previous < fold.training.end_index == fold.validation.start_index
        assert fold.validation.end_index <= config.learning_curve_evaluation_range[0] < development
        previous = fold.training.end_index
    assert len(plan.chronological_validation_folds) == 5
    points = plan_learning_curve(config.learning_curve_session_counts,
        development_session_count=development, evaluation_range=config.learning_curve_evaluation_range)
    assert len(set(config.learning_curve_session_counts)) == len(points)
    assert {point.evaluation_end_index for point in points} == {development}
    assert all(point.training_start_index == 0 and point.training_end_index <= point.evaluation_start_index
               for point in points)


def test_resolved_population_cannot_change_and_invalid_folds_are_rejected():
    config = resolve_benchmark_config(550, load_benchmark_config("configs/benchmark/default.yaml"))
    with pytest.raises(ValueError, match="rebound"):
        plan_benchmark(551, config)
    with pytest.raises(ValueError, match="final holdout"):
        plan_benchmark(550, replace(config, chronological_validation_folds=((495, 495, 510),)))
    with pytest.raises(ValueError):
        plan_learning_curve((250, 496), development_session_count=495, evaluation_range=(445, 495))


def test_minimum_and_rounded_duplicate_checkpoints():
    config = load_benchmark_config("configs/benchmark/default.yaml")
    minimum = minimum_session_count(config.rules)
    assert minimum < 550
    for count in range(1, minimum):
        with pytest.raises(ValueError, match=f"at least {minimum} complete eligible sessions"):
            plan_benchmark(count, config)
    for count in range(minimum, minimum + 100):
        plan_benchmark(count, config)
    config = replace(config, rules=replace(config.rules, learning_curve_fractions=(.1, .101, .5, 1.0)))
    counts = resolve_benchmark_config(minimum, config).learning_curve_session_counts
    assert len(counts) == 3
    assert tuple(sorted(set(counts))) == counts


@pytest.mark.parametrize("changes", [{"test_fraction": .2}, {"version": "other"},
    {"initial_training_fraction": .95}, {"learning_evaluation_fraction": 0},
    {"fold_count": 0}, {"learning_curve_fractions": (.1, .1, 1.)},
    {"learning_curve_fractions": (.1, float("nan"), 1.)}])
def test_invalid_rules_fail(changes):
    config = load_benchmark_config("configs/benchmark/default.yaml")
    with pytest.raises(ValueError):
        plan_benchmark(550, replace(config, rules=replace(config.rules, **changes)))


def test_absolute_format_one_configuration_is_rejected(tmp_path):
    """New studies must derive boundaries from the current generalized protocol."""

    current = Path("configs/benchmark/default.yaml").read_text(encoding="utf-8")
    legacy = tmp_path / "legacy.yaml"
    legacy.write_text(current.replace("format_version: 2", "format_version: 1", 1))
    with pytest.raises(ValueError, match="format 2 is required"):
        load_benchmark_config(legacy)
