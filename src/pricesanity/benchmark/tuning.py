"""Candidate-selection contracts for chronological development tuning."""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CandidateResult:
    """Mean fold score and readable settings for one tried configuration."""

    candidate_index: int
    parameters: dict[str, Any]
    fold_macro_f1: tuple[float, ...]

    @property
    def mean_macro_f1(self) -> float:
        """Average only chronological validation folds."""

        # An unevaluated candidate has no score; zero would conceal missing validation.
        if not self.fold_macro_f1:
            raise ValueError("A tuning candidate requires validation-fold scores.")
        return sum(self.fold_macro_f1) / len(self.fold_macro_f1)


def select_best_candidate(
    candidates: list[CandidateResult],
) -> CandidateResult:
    """Select validation macro-F1 with stable earliest-candidate tie breaking."""

    # Unique candidate indices supply provenance and deterministic tie breaking.
    if not candidates:
        raise ValueError("Tuning requires at least one completed candidate.")
    if len({candidate.candidate_index for candidate in candidates}) != len(candidates):
        raise ValueError("Tuning candidate indices must be unique.")
    # Equal scores prefer the earliest candidate rather than arbitrary container order.
    return max(
        candidates,
        key=lambda candidate: (candidate.mean_macro_f1, -candidate.candidate_index),
    )
