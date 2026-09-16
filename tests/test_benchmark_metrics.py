import numpy as np

from pricesanity.benchmark.metrics import (
    classification_metrics,
    evaluate_benchmark_predictions,
)


def test_common_metrics_report_every_class_and_confusion_cell() -> None:
    """The shared schema retains class detail rather than one opaque score."""

    metrics = classification_metrics(
        np.array([0, 0, 1, 1, 2, 2]),
        np.array([0, 1, 1, 1, 2, 0]),
    )

    assert metrics.support == 6
    assert metrics.confusion_matrix == ((1, 1, 0), (0, 2, 0), (1, 0, 1))
    assert set(metrics.per_class) == {"bull", "bear", "range"}
    assert metrics.accuracy == 4 / 6
    assert 0.0 <= metrics.macro_f1 <= 1.0


def test_transition_metrics_never_match_across_session_boundaries() -> None:
    """A new session's first regime is a start, not a transition from yesterday."""

    human = np.array([0, 0, 1, 1, 2, 2])
    predicted = np.array([0, 0, 1, 0, 2, 2])
    sessions = np.array([0, 0, 0, 1, 1, 1])
    metrics = evaluate_benchmark_predictions(
        human_current=human,
        predicted_current=predicted,
        human_anticipated=human,
        predicted_anticipated=predicted,
        session_indices=sessions,
    )

    assert metrics.transitions.exact.human_transition_count == 2
    assert metrics.transitions.exact.predicted_transition_count == 2
    assert metrics.transitions.exact.matched_transition_count == 1
    assert metrics.mean_head_macro_f1 == metrics.current.macro_f1


def test_transition_match_requires_the_same_from_and_to_regimes() -> None:
    """Arriving at one regime from a different regime is not the same transition."""

    metrics = evaluate_benchmark_predictions(
        human_current=np.array([0, 1]),
        predicted_current=np.array([2, 1]),
        human_anticipated=np.array([0, 1]),
        predicted_anticipated=np.array([2, 1]),
        session_indices=np.array([0, 0]),
    )
    assert metrics.transitions.exact.human_transition_count == 1
    assert metrics.transitions.exact.predicted_transition_count == 1
    assert metrics.transitions.exact.matched_transition_count == 0
