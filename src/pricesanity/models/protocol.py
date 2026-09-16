"""Minimal contracts shared by tabular and sequence benchmark runners."""

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np


@dataclass(frozen=True)
class PredictionContext:
    """Audit context needed only by reference baselines, never learned models."""

    session_indices: np.ndarray
    previous_current_targets: np.ndarray | None = None
    previous_anticipated_targets: np.ndarray | None = None
    valid_history_mask: np.ndarray | None = None


@dataclass(frozen=True)
class DualRegimePredictions:
    """Aligned class predictions for both human annotation questions."""

    current: np.ndarray
    anticipated: np.ndarray


@dataclass(frozen=True)
class DualRegimeProbabilities:
    """Aligned three-class probability estimates for both annotation questions."""

    current: np.ndarray
    anticipated: np.ndarray


@dataclass(frozen=True)
class DualRegimeScores:
    """Uncalibrated class scores whose values need not sum to one."""

    current: np.ndarray
    anticipated: np.ndarray


@dataclass(frozen=True)
class DualRegimeOutput:
    """One inference pass containing authoritative classes and optional uncertainty values.

    Native predicted classes remain authoritative. ``probabilities`` are model-reported
    probability estimates when the model has a defensible probability path. ``scores`` are
    uncalibrated decision values and must never be presented as certainty.
    """

    predictions: DualRegimePredictions
    probabilities: DualRegimeProbabilities | None
    scores: DualRegimeScores | None
    uncertainty_kind: str


@runtime_checkable
class BenchmarkModel(Protocol):
    """Small adapter surface needed by the common experiment runner."""

    @property
    def name(self) -> str:
        """Return the stable model-family identifier."""

    def fit(
        self,
        features: np.ndarray,
        current_targets: np.ndarray,
        anticipated_targets: np.ndarray,
        *,
        context: PredictionContext | None = None,
    ) -> None:
        """Fit both target heads using development training data only."""

    def predict_output(
        self,
        features: np.ndarray,
        *,
        context: PredictionContext | None = None,
    ) -> DualRegimeOutput:
        """Return both heads from one timed inference operation."""

    def save(self, path: str | Path) -> None:
        """Persist fitted model state inside its run directory."""

    def describe(self) -> Mapping[str, Any]:
        """Return a JSON-compatible model configuration."""

    def state_fingerprint(self) -> str:
        """Identify exact fitted state for stage-coherent resume."""
