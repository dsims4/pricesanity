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

    # Normalize caller-owned arrays to one integer representation so validation and indexing
    # use the same class encoding for every model family.
    human_targets = np.asarray(human_targets, dtype=np.int64)
    predicted_targets = np.asarray(predicted_targets, dtype=np.int64)

    # Metrics are meaningful only for the exact aligned scored population. Empty or differently
    # shaped vectors would make cross-model comparisons use different evidence.
    if (
        human_targets.ndim != 1
        or predicted_targets.shape != human_targets.shape
        or human_targets.size == 0
    ):
        raise ValueError("Classification targets must be aligned nonempty vectors.")

    # The class-name order fixes the row and column meaning in every persisted confusion matrix.
    # Reject unknown integer labels rather than silently expanding or reordering that schema.
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

    # Derive each class from the same row=true, column=predicted matrix so per-class values and
    # the displayed confusion matrix cannot disagree about label orientation.
    for class_index, class_name in enumerate(class_names):
        true_positive = int(confusion[class_index, class_index])
        false_positive = int(confusion[:, class_index].sum() - true_positive)
        false_negative = int(confusion[class_index, :].sum() - true_positive)
        support = int(confusion[class_index, :].sum())

        # Undefined ratios become zero through one shared policy. This keeps absent or never-
        # predicted classes in the macro average instead of rewarding models that omit them.
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

    # Macro F1 weights all three regimes equally, while accuracy retains the overall candle-level
    # view. Persisting both prevents class imbalance from being hidden by a single score.
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

    # Current-regime transitions are session-local events, so their session identity vector must
    # align exactly with the current targets before any neighborhood is constructed.
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

    # Anticipated labels have their own transition process. Build separate masks so anticipated
    # transition diagnostics are not incorrectly anchored to changes in the current-regime head.
    anticipated_masks = {
        tolerance: _human_transition_neighborhood_mask(
            human_anticipated,
            session_indices,
            tolerance=tolerance,
        )
        for tolerance in (0, 1, 2)
    }

    # Assemble every head and transition view from the same aligned arrays. The combined score
    # remains a property of the finished object so neither head disappears during construction.
    return BenchmarkMetrics(
        # Head-level classification stays explicit because the benchmark requires both tasks.
        current=classification_metrics(human_current, predicted_current),
        anticipated=classification_metrics(
            human_anticipated, predicted_anticipated
        ),
        transitions=TransitionMetrics(
            # Exact and tolerant matches reveal whether a model detects a change late or early,
            # rather than treating every near-boundary candle as an unrelated classification.
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
            # Score both heads around human current-regime changes to expose local behavior at
            # precisely the candles where discretionary interpretation changes state.
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
            # Preserve the same matching tolerances for anticipated-label changes so the two
            # transition reports retain comparable semantics.
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
            # Reuse the common head metrics but select candles around anticipated transitions.
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

    # Models exposing only decision scores receive no probability metrics. Treating margins as
    # probabilities would make log loss and Brier score scientifically invalid.
    if probabilities is None:
        return None

    # Require the fixed bull/bear/range class axis and normalized finite rows before applying
    # proper scoring rules shared across model families.
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

    # One-hot targets put the multiclass Brier calculation on the same fixed class ordering as
    # the probability columns and confusion matrices.
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

    # Start false everywhere so only candles explicitly connected to a within-session human
    # transition enter the diagnostic population.
    mask = np.zeros(len(human_current), dtype=bool)

    # A label difference across adjacent rows is not a transition when those rows belong to
    # different sessions; overnight boundaries cannot borrow causal context from the next day.
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

        # Clip the numeric neighborhood again by session identity because the transition may sit
        # near a session edge even after its center was validated as a within-session change.
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

    # The same mask selects truth and prediction so neighborhood metrics preserve exact target
    # alignment rather than evaluating the two arrays on different candle subsets.
    return classification_metrics(targets[mask], predictions[mask])


def _transition_metrics(
    human_regimes: np.ndarray,
    predicted_regimes: np.ndarray,
    session_indices: np.ndarray,
    *,
    tolerance: int,
) -> TransitionToleranceMetrics:
    """Match each predicted change to at most one same-direction human change."""

    # Convert both label sequences into session-relative directed changes. Direction matters:
    # a bull-to-range prediction cannot receive credit for a nearby bull-to-bear human change.
    human_transitions = _transition_events(human_regimes, session_indices)
    predicted_transitions = _transition_events(predicted_regimes, session_indices)

    # Human events are consumed after matching so each annotated transition contributes at most
    # one true positive even when the prediction oscillates repeatedly nearby.
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

    # Precision uses predicted changes as its population; recall uses human changes. The shared
    # zero policy keeps no-transition sessions representable without division failures.
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
    # Session-relative positions allow the same tolerance meaning on every session regardless of
    # its location in the concatenated evaluation vector.
    events = []
    session_positions: dict[int, int] = {int(session_indices[0]): 0}
    for sample_index in range(1, len(regimes)):
        session_index = int(session_indices[sample_index])
        session_positions[session_index] = session_positions.get(session_index, -1) + 1

        # The first scored candle of a new session establishes state but is never interpreted as
        # a transition from the previous session's final label.
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
