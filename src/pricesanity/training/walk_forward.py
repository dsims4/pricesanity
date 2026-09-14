"""Plan auditable chronological windows for repeated model experiments."""

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date

import pandas as pd

from pricesanity.training.dataset import AnnotatedSessionSplit


@dataclass(frozen=True)
class SessionBoundary:
    """Half-open session indices and inclusive dates for one experiment role."""

    start_index: int
    end_index: int
    start_date: date
    end_date: date


@dataclass(frozen=True)
class WalkForwardRun:
    """One expanding-history train/validation/test experiment in historical order."""

    run_index: int
    training: SessionBoundary
    validation: SessionBoundary
    test: SessionBoundary

    def select_sessions(
        self,
        sessions: Sequence[pd.DataFrame],
    ) -> AnnotatedSessionSplit:
        """Select this run's nonoverlapping roles from the audited corpus."""

        if self.test.end_index > len(sessions):
            raise ValueError("Walk-forward run exceeds the supplied session corpus.")

        # A previous run's test sessions may appear here only after time has
        # advanced. The active test slice remains outside this run's fitting roles.
        return AnnotatedSessionSplit(
            training=tuple(
                sessions[self.training.start_index : self.training.end_index]
            ),
            validation=tuple(
                sessions[self.validation.start_index : self.validation.end_index]
            ),
            test=tuple(sessions[self.test.start_index : self.test.end_index]),
        )


@dataclass(frozen=True)
class WalkForwardPlan:
    """Complete experiment windows plus history too short for another run."""

    initial_training_session_count: int
    validation_session_count: int
    test_session_count: int
    training_growth_session_count: int
    runs: tuple[WalkForwardRun, ...]
    trailing_session_count: int


def plan_walk_forward_runs(
    sessions: Sequence[pd.DataFrame],
    *,
    initial_training_session_count: int = 100,
    validation_session_count: int = 20,
    test_session_count: int = 10,
) -> WalkForwardPlan:
    """Build chronological expanding histories with fixed validation and test blocks.

    The default training counts are 100, 220, 340, and 460. Each run keeps every
    earlier training session; its active validation and test are excluded from fitting.

    Args:
        sessions: Complete annotated sessions in chronological trading order.
        initial_training_session_count: Historical sessions used to fit the first model.
        validation_session_count: Following sessions used only for model selection.
        test_session_count: Final sessions held untouched until selection ends.

    Returns:
        Explicit boundaries for every complete run and any unused trailing history.

    Raises:
        ValueError: If counts, session identities, or chronology are invalid.
    """

    role_counts = (
        initial_training_session_count,
        validation_session_count,
        test_session_count,
    )
    if any(session_count <= 0 for session_count in role_counts):
        raise ValueError("Walk-forward role counts must all be positive.")

    # The next training history gains 120 sessions by default: the prior 20 validation
    # sessions, prior 10 test sessions, and 90 newly elapsed sessions. Do not slide the start.
    training_growth_session_count = initial_training_session_count + validation_session_count

    session_dates = []
    for session in sessions:
        if session.empty or "session_date" not in session.columns:
            raise ValueError("Every walk-forward member must be a nonempty session.")
        unique_dates = session["session_date"].drop_duplicates()
        if len(unique_dates) != 1:
            raise ValueError("Each walk-forward member must contain one session date.")
        session_dates.append(pd.Timestamp(unique_dates.iloc[0]).date())

    if session_dates != sorted(session_dates) or len(session_dates) != len(
        set(session_dates)
    ):
        raise ValueError("Walk-forward sessions must have unique chronological dates.")

    runs = []

    # Keep the oldest history instead of sliding forward. Validation and test stay fixed in
    # size; only the historical training pool grows as more trading sessions become available.
    training_end_index = initial_training_session_count
    while training_end_index + validation_session_count + test_session_count <= len(sessions):
        validation_end_index = training_end_index + validation_session_count
        test_end_index = validation_end_index + test_session_count

        runs.append(
            WalkForwardRun(
                run_index=len(runs) + 1,
                training=_build_boundary(
                    session_dates,
                    0,
                    training_end_index,
                ),
                validation=_build_boundary(
                    session_dates,
                    training_end_index,
                    validation_end_index,
                ),
                test=_build_boundary(
                    session_dates,
                    validation_end_index,
                    test_end_index,
                ),
            )
        )

        # Old test sessions are legal training history in a later run, once their dates are
        # earlier than that run's validation period. Their original saved results stay fixed.
        training_end_index += training_growth_session_count

    if runs:
        trailing_session_count = len(sessions) - runs[-1].test.end_index
    else:
        trailing_session_count = len(sessions)

    return WalkForwardPlan(
        initial_training_session_count=initial_training_session_count,
        validation_session_count=validation_session_count,
        test_session_count=test_session_count,
        training_growth_session_count=training_growth_session_count,
        runs=tuple(runs),
        trailing_session_count=trailing_session_count,
    )


def _build_boundary(
    session_dates: Sequence[date],
    start_index: int,
    end_index: int,
) -> SessionBoundary:
    """Attach readable inclusive dates to one half-open session slice."""

    return SessionBoundary(
        start_index=start_index,
        end_index=end_index,
        start_date=session_dates[start_index],
        end_date=session_dates[end_index - 1],
    )
