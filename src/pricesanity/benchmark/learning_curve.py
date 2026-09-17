"""Plan comparable development-history prefixes for learning curves."""

from dataclasses import dataclass


@dataclass(frozen=True)
class LearningCurvePoint:
    """One training-prefix size awaiting repeated model evaluation."""

    training_session_count: int
    training_start_index: int
    training_end_index: int
    evaluation_start_index: int
    evaluation_end_index: int


def plan_learning_curve(
    session_counts: tuple[int, ...],
    *,
    development_session_count: int,
    evaluation_range: tuple[int, int],
) -> tuple[LearningCurvePoint, ...]:
    """Pair growing training prefixes with one unchanged future development block."""

    # Distinct increasing sizes define the requested curve; do not silently repair its plan.
    if tuple(sorted(set(session_counts))) != session_counts:
        raise ValueError("Learning-curve session counts must increase without duplicates.")
    if not session_counts or session_counts[0] <= 0:
        raise ValueError("Learning-curve session counts must be positive.")
    if session_counts[-1] > development_session_count:
        raise ValueError("Learning curves cannot use final-holdout sessions.")
    # Fix evaluation beyond the largest prefix so changes reflect added training evidence.
    evaluation_start, evaluation_end = evaluation_range
    if not (
        session_counts[-1] <= evaluation_start
        < evaluation_end
        <= development_session_count
    ):
        raise ValueError(
            "The fixed learning-curve evaluation block must follow every training prefix."
        )
    # Share the oldest session and half-open bounds; only training-history size changes.
    return tuple(
        LearningCurvePoint(
            session_count,
            0,
            session_count,
            evaluation_start,
            evaluation_end,
        )
        for session_count in session_counts
    )
