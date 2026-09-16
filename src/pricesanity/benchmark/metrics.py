"""Calculate one transparent metric schema for every benchmark model."""

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class PerClassMetrics:
    """Precision, recall, F1, and support for one named regime."""

    precision: float
    recall: float
    f1: float
    support: int


@dataclass(frozen=True)
class ClassificationMetrics:
    """Overall and class-balanced measurements for one prediction head."""

    accuracy: float
    macro_f1: float
    support: int
    per_class: dict[str, PerClassMetrics]
    confusion_matrix: tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class TransitionToleranceMetrics:
    """One-to-one transition matches at a particular candle tolerance."""

    tolerance_candles: int
    human_transition_count: int
    predicted_transition_count: int
    matched_transition_count: int
    precision: float
    recall: float


@dataclass(frozen=True)
class TransitionMetrics:
    """Exact and near-candle current-regime transition measurements."""

    exact: TransitionToleranceMetrics
    within_one_candle: TransitionToleranceMetrics
    within_two_candles: TransitionToleranceMetrics


@dataclass(frozen=True)
class TransitionNeighborhoodMetrics:
    """Classification quality at and around human current-regime transitions."""

    exact_current: ClassificationMetrics | None
    exact_anticipated: ClassificationMetrics | None
    within_one_current: ClassificationMetrics | None
    within_one_anticipated: ClassificationMetrics | None
    within_two_current: ClassificationMetrics | None
    within_two_anticipated: ClassificationMetrics | None


@dataclass(frozen=True)
class ProbabilityMetrics:
    """Secondary probability quality for one head when estimates are available."""

    multiclass_log_loss: float
    multiclass_brier_score: float
    support: int


@dataclass(frozen=True)
class EfficiencyMetrics:
    """Comparable runtime and serialized-cost evidence."""

    training_seconds: float
    inference_seconds: float
    serialized_model_bytes: int
    parameter_count: int | None
    training_sample_count: int = 0
    inference_sample_count: int = 0
    training_samples_per_second: float = 0.0
    inference_samples_per_second: float = 0.0
    device: str = "unknown"
    cpu_worker_count: int = 1
    timing_repetitions: int = 1
    data_loader_worker_count: int = 0
    hardware_fingerprint: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BenchmarkMetrics:
    """Complete shared result for the two labels, transitions, and efficiency."""

    current: ClassificationMetrics
    anticipated: ClassificationMetrics
    transitions: TransitionMetrics
    transition_neighborhoods: TransitionNeighborhoodMetrics
    current_probability: ProbabilityMetrics | None = None
    anticipated_probability: ProbabilityMetrics | None = None
    efficiency: EfficiencyMetrics | None = None
    anticipated_transitions: TransitionMetrics | None = None
    anticipated_transition_neighborhoods: TransitionNeighborhoodMetrics | None = None

    @property
    def mean_head_macro_f1(self) -> float:
        """Average the two required tasks without hiding either task's score."""

        return (self.current.macro_f1 + self.anticipated.macro_f1) / 2.0

    def to_dict(self) -> dict[str, Any]:
        """Return only JSON-compatible named metric values."""

        values = asdict(self)
        values["mean_head_macro_f1"] = self.mean_head_macro_f1
        return values


def classification_metrics(
    human_targets: np.ndarray,
    predicted_targets: np.ndarray,
    *,
    class_names: tuple[str, ...] = ("bull", "bear", "range"),
) -> ClassificationMetrics:
    """Calculate an explicit confusion matrix and derived per-class values."""

    human_targets = np.asarray(human_targets, dtype=np.int64)
    predicted_targets = np.asarray(predicted_targets, dtype=np.int64)
    if (
        human_targets.ndim != 1
        or predicted_targets.shape != human_targets.shape
        or human_targets.size == 0
    ):
        raise ValueError("Classification targets must be aligned nonempty vectors.")
    class_count = len(class_names)
    if (
        (human_targets < 0).any()
        or (human_targets >= class_count).any()
        or (predicted_targets < 0).any()
        or (predicted_targets >= class_count).any()
    ):
        raise ValueError("Classification targets contain an unknown regime class.")

    # Encoding each truth/prediction pair as one integer builds the full confusion matrix
    # without a Python loop and fixes the row=true, column=predicted convention explicitly.
    confusion = np.bincount(
        human_targets * class_count + predicted_targets,
        minlength=class_count**2,
    ).reshape(class_count, class_count)
    class_metrics: dict[str, PerClassMetrics] = {}
    f1_values = []
    for class_index, class_name in enumerate(class_names):
        true_positive = int(confusion[class_index, class_index])
        false_positive = int(confusion[:, class_index].sum() - true_positive)
        false_negative = int(confusion[class_index, :].sum() - true_positive)
        support = int(confusion[class_index, :].sum())
        precision = _safe_ratio(true_positive, true_positive + false_positive)
        recall = _safe_ratio(true_positive, true_positive + false_negative)
        f1 = _safe_ratio(2.0 * precision * recall, precision + recall)
        class_metrics[class_name] = PerClassMetrics(
            precision=precision,
            recall=recall,
            f1=f1,
            support=support,
        )
        f1_values.append(f1)

    return ClassificationMetrics(
        accuracy=float(np.trace(confusion) / human_targets.size),
        macro_f1=float(np.mean(f1_values)),
        support=int(human_targets.size),
        per_class=class_metrics,
        confusion_matrix=tuple(tuple(int(value) for value in row) for row in confusion),
    )


def evaluate_benchmark_predictions(
    *,
    human_current: np.ndarray,
    predicted_current: np.ndarray,
    human_anticipated: np.ndarray,
    predicted_anticipated: np.ndarray,
    session_indices: np.ndarray,
    current_probabilities: np.ndarray | None = None,
    anticipated_probabilities: np.ndarray | None = None,
    efficiency: EfficiencyMetrics | None = None,
) -> BenchmarkMetrics:
    """Evaluate both labels and current-regime transitions without crossing sessions."""

    human_current = np.asarray(human_current, dtype=np.int64)
    predicted_current = np.asarray(predicted_current, dtype=np.int64)
    session_indices = np.asarray(session_indices, dtype=np.int64)
    if session_indices.shape != human_current.shape:
        raise ValueError("Transition session identities must align with current targets.")

    human_anticipated = np.asarray(human_anticipated, dtype=np.int64)
    predicted_anticipated = np.asarray(predicted_anticipated, dtype=np.int64)
    # Neighborhoods are anchored to human transitions. Anchoring them to model predictions
    # would give each model a different and potentially easier diagnostic population.
    transition_masks = {
        tolerance: _human_transition_neighborhood_mask(
            human_current,
            session_indices,
            tolerance=tolerance,
        )
        for tolerance in (0, 1, 2)
    }

    anticipated_masks = {
        tolerance: _human_transition_neighborhood_mask(human_anticipated, session_indices, tolerance=tolerance)
        for tolerance in (0, 1, 2)
    }

    return BenchmarkMetrics(
        current=classification_metrics(human_current, predicted_current),
        anticipated=classification_metrics(
            human_anticipated, predicted_anticipated
        ),
        transitions=TransitionMetrics(
            exact=_transition_metrics(
                human_current, predicted_current, session_indices, tolerance=0
            ),
            within_one_candle=_transition_metrics(
                human_current, predicted_current, session_indices, tolerance=1
            ),
            within_two_candles=_transition_metrics(
                human_current, predicted_current, session_indices, tolerance=2
            ),
        ),
        transition_neighborhoods=TransitionNeighborhoodMetrics(
            exact_current=_masked_classification(
                human_current, predicted_current, transition_masks[0]
            ),
            exact_anticipated=_masked_classification(
                human_anticipated, predicted_anticipated, transition_masks[0]
            ),
            within_one_current=_masked_classification(
                human_current, predicted_current, transition_masks[1]
            ),
            within_one_anticipated=_masked_classification(
                human_anticipated, predicted_anticipated, transition_masks[1]
            ),
            within_two_current=_masked_classification(
                human_current, predicted_current, transition_masks[2]
            ),
            within_two_anticipated=_masked_classification(
                human_anticipated, predicted_anticipated, transition_masks[2]
            ),
        ),
        anticipated_transitions=TransitionMetrics(
            exact=_transition_metrics(
                human_anticipated, predicted_anticipated, session_indices, tolerance=0
            ),
            within_one_candle=_transition_metrics(
                human_anticipated, predicted_anticipated, session_indices, tolerance=1
            ),
            within_two_candles=_transition_metrics(
                human_anticipated, predicted_anticipated, session_indices, tolerance=2
            ),
        ),
        anticipated_transition_neighborhoods=TransitionNeighborhoodMetrics(
            exact_current=_masked_classification(
                human_current, predicted_current, anticipated_masks[0]
            ),
            exact_anticipated=_masked_classification(
                human_anticipated, predicted_anticipated, anticipated_masks[0]
            ),
            within_one_current=_masked_classification(
                human_current, predicted_current, anticipated_masks[1]
            ),
            within_one_anticipated=_masked_classification(
                human_anticipated, predicted_anticipated, anticipated_masks[1]
            ),
            within_two_current=_masked_classification(
                human_current, predicted_current, anticipated_masks[2]
            ),
            within_two_anticipated=_masked_classification(
                human_anticipated, predicted_anticipated, anticipated_masks[2]
            ),
        ),
        current_probability=_probability_metrics(
            human_current, current_probabilities
        ),
        anticipated_probability=_probability_metrics(
            human_anticipated, anticipated_probabilities
        ),
        efficiency=efficiency,
    )


def _probability_metrics(
    targets: np.ndarray,
    probabilities: np.ndarray | None,
) -> ProbabilityMetrics | None:
    """Measure log loss and Brier score only for genuine probability estimates."""

    if probabilities is None:
        return None
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.shape != (len(targets), 3):
        raise ValueError("Probability estimates must have sample and three class axes.")
    if (
        not np.isfinite(probabilities).all()
        or (probabilities < 0.0).any()
        or (probabilities > 1.0).any()
        or not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6)
    ):
        raise ValueError("Probability estimates must be finite rows summing to one.")
    # Clipping protects log(0) numerically; Brier score still uses the original probabilities.
    clipped = np.clip(probabilities, 1e-15, 1.0)
    one_hot_targets = np.eye(3, dtype=np.float64)[targets]
    return ProbabilityMetrics(
        multiclass_log_loss=float(
            -np.log(clipped[np.arange(len(targets)), targets]).mean()
        ),
        multiclass_brier_score=float(
            np.square(probabilities - one_hot_targets).sum(axis=1).mean()
        ),
        support=len(targets),
    )


def _human_transition_neighborhood_mask(
    human_current: np.ndarray,
    session_indices: np.ndarray,
    *,
    tolerance: int,
) -> np.ndarray:
    """Select positions near human changes without spilling into another session."""

    mask = np.zeros(len(human_current), dtype=bool)
    transition_positions = [
        position
        for position in range(1, len(human_current))
        if session_indices[position] == session_indices[position - 1]
        and human_current[position] != human_current[position - 1]
    ]
    for transition_position in transition_positions:
        session_index = session_indices[transition_position]
        start = max(0, transition_position - tolerance)
        end = min(len(mask), transition_position + tolerance + 1)
        local_positions = np.arange(start, end)
        mask[local_positions[session_indices[local_positions] == session_index]] = True
    return mask


def _masked_classification(
    targets: np.ndarray,
    predictions: np.ndarray,
    mask: np.ndarray,
) -> ClassificationMetrics | None:
    """Return no diagnostic rather than inventing a score when no transition exists."""

    if not mask.any():
        return None
    return classification_metrics(targets[mask], predictions[mask])


def _transition_metrics(
    human_regimes: np.ndarray,
    predicted_regimes: np.ndarray,
    session_indices: np.ndarray,
    *,
    tolerance: int,
) -> TransitionToleranceMetrics:
    """Match each predicted change to at most one same-direction human change."""

    human_transitions = _transition_events(human_regimes, session_indices)
    predicted_transitions = _transition_events(predicted_regimes, session_indices)
    unmatched_human = set(range(len(human_transitions)))
    matched_count = 0
    for (
        predicted_session,
        predicted_position,
        predicted_from_regime,
        predicted_to_regime,
    ) in predicted_transitions:
        candidates = [
            human_index
            for human_index in unmatched_human
            if human_transitions[human_index][0] == predicted_session
            and human_transitions[human_index][2] == predicted_from_regime
            and human_transitions[human_index][3] == predicted_to_regime
            and abs(human_transitions[human_index][1] - predicted_position) <= tolerance
        ]
        if candidates:
            # Consuming the nearest compatible event enforces one-to-one credit: several
            # noisy predicted flips cannot all claim the same human regime transition.
            nearest = min(
                candidates,
                key=lambda index: abs(human_transitions[index][1] - predicted_position),
            )
            unmatched_human.remove(nearest)
            matched_count += 1

    return TransitionToleranceMetrics(
        tolerance_candles=tolerance,
        human_transition_count=len(human_transitions),
        predicted_transition_count=len(predicted_transitions),
        matched_transition_count=matched_count,
        precision=_safe_ratio(matched_count, len(predicted_transitions)),
        recall=_safe_ratio(matched_count, len(human_transitions)),
    )


def _transition_events(
    regimes: np.ndarray,
    session_indices: np.ndarray,
) -> list[tuple[int, int, int, int]]:
    """Locate changes while treating the first sample of every session as a start."""

    if regimes.shape != session_indices.shape:
        raise ValueError("Transition regimes and session identities must align.")
    events = []
    session_positions: dict[int, int] = {int(session_indices[0]): 0}
    for sample_index in range(1, len(regimes)):
        session_index = int(session_indices[sample_index])
        session_positions[session_index] = session_positions.get(session_index, -1) + 1
        if session_index != int(session_indices[sample_index - 1]):
            continue
        if regimes[sample_index] != regimes[sample_index - 1]:
            events.append(
                (
                    session_index,
                    session_positions[session_index],
                    int(regimes[sample_index - 1]),
                    int(regimes[sample_index]),
                )
            )
    return events


def _safe_ratio(numerator: float, denominator: float) -> float:
    """Use zero for undefined precision, recall, or class F1."""

    return float(numerator / denominator) if denominator else 0.0
