"""Verify expanding-history boundaries without training a model."""

import pandas as pd
import pytest

from pricesanity.training.walk_forward import plan_walk_forward_runs


def _build_sessions(session_count: int) -> list[pd.DataFrame]:
    """Supply chronological trading dates independent of calendar-day gaps."""

    return [
        pd.DataFrame({"session_date": [session_date]})
        for session_date in pd.bdate_range("2016-01-04", periods=session_count).date
    ]


def test_first_four_expanding_runs_have_exact_boundaries() -> None:
    """The oldest session stays in training while validation and test keep fixed sizes."""

    sessions = _build_sessions(490)
    plan = plan_walk_forward_runs(sessions)
    expected_boundaries = (
        ((0, 100), (100, 120), (120, 130)),
        ((0, 220), (220, 240), (240, 250)),
        ((0, 340), (340, 360), (360, 370)),
        ((0, 460), (460, 480), (480, 490)),
    )
    assert len(plan.runs) == 4
    assert plan.training_growth_session_count == 120

    for run, expected_roles in zip(plan.runs, expected_boundaries, strict=True):
        for boundary, (start_index, end_index) in zip(
            (run.training, run.validation, run.test), expected_roles, strict=True
        ):
            assert (boundary.start_index, boundary.end_index) == (start_index, end_index)
            assert boundary.start_date == sessions[start_index]["session_date"].iloc[0]
            assert boundary.end_date == sessions[end_index - 1]["session_date"].iloc[0]

        split = run.select_sessions(sessions)
        assert len(split.training) == expected_roles[0][1]
        assert len(split.validation) == 20
        assert len(split.test) == 10
        assert split.training[0] is sessions[0]


def test_active_roles_are_disjoint_and_old_tests_become_history() -> None:
    """Test labels become legal training history only in a later chronological experiment."""

    sessions = _build_sessions(490)
    plan = plan_walk_forward_runs(sessions)
    previous_test_indices = set()

    for run in plan.runs:
        training_indices = set(range(run.training.start_index, run.training.end_index))
        validation_indices = set(range(run.validation.start_index, run.validation.end_index))
        test_indices = set(range(run.test.start_index, run.test.end_index))
        assert training_indices.isdisjoint(validation_indices | test_indices)
        assert validation_indices.isdisjoint(test_indices)
        assert previous_test_indices <= training_indices
        assert run.training.end_date < run.validation.start_date
        assert run.validation.end_date < run.test.start_date
        previous_test_indices = test_indices


@pytest.mark.parametrize("session_count, run_count, trailing", [
    (129, 0, 129), (130, 1, 0), (249, 1, 119), (250, 2, 0),
    (369, 2, 119), (370, 3, 0), (489, 3, 119), (495, 4, 5),
])
def test_insufficient_history_never_creates_partial_run(
    session_count, run_count, trailing
) -> None:
    """A new experiment waits until both evaluation blocks are fully available."""

    plan = plan_walk_forward_runs(_build_sessions(session_count))
    assert len(plan.runs) == run_count
    assert plan.trailing_session_count == trailing


@pytest.mark.parametrize("damage", ["reordered", "duplicate", "empty"])
def test_expanding_planner_rejects_invalid_session_order(damage) -> None:
    """Index arithmetic cannot repair an ambiguous or nonchronological source corpus."""

    sessions = _build_sessions(130)
    if damage == "reordered":
        sessions.reverse()
    elif damage == "duplicate":
        sessions[1] = sessions[0]
    else:
        sessions[1] = pd.DataFrame()

    with pytest.raises(ValueError):
        plan_walk_forward_runs(sessions)
