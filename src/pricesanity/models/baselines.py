"""Transparent reference models requiring no hyperparameter search."""

import json
import hashlib
from pathlib import Path
from typing import Any

import numpy as np

from pricesanity.models.protocol import (
    DualRegimeOutput,
    DualRegimePredictions,
    DualRegimeProbabilities,
    PredictionContext,
)


class MajorityClassBaseline:
    """Predict each training-only majority class for every future sample."""

    name = "majority_class"

    def __init__(self) -> None:
        self._current_class: int | None = None
        self._anticipated_class: int | None = None

    def fit(
        self,
        features: np.ndarray,
        current_targets: np.ndarray,
        anticipated_targets: np.ndarray,
        *,
        context: PredictionContext | None = None,
    ) -> None:
        """Learn both class frequencies only from the supplied training labels."""

        _validate_fit_arrays(features, current_targets, anticipated_targets)
        # The annotation questions have different class distributions, so one shared
        # majority would weaken the baseline and obscure which head is genuinely difficult.
        self._current_class = _majority_class(current_targets)
        self._anticipated_class = _majority_class(anticipated_targets)

    def predict(
        self,
        features: np.ndarray,
        *,
        context: PredictionContext | None = None,
    ) -> DualRegimePredictions:
        """Repeat the two fitted majority classes."""

        current_class, anticipated_class = self._require_fitted()
        # Derive the output length from the same feature contract used by learned models so
        # the reference cannot score a different number of examples.
        sample_count = _sample_count(features)
        return DualRegimePredictions(
            current=np.full(sample_count, current_class, dtype=np.int64),
            anticipated=np.full(sample_count, anticipated_class, dtype=np.int64),
        )

    def predict_proba(
        self,
        features: np.ndarray,
        *,
        context: PredictionContext | None = None,
    ) -> DualRegimeProbabilities:
        """Represent deterministic baseline choices as one-hot probabilities."""

        # The shared evaluator expects a three-column distribution even when a strategy has
        # no uncertainty to estimate.
        predictions = self.predict(features, context=context)
        return DualRegimeProbabilities(
            current=_one_hot(predictions.current),
            anticipated=_one_hot(predictions.anticipated),
        )

    def predict_output(
        self,
        features: np.ndarray,
        *,
        context: PredictionContext | None = None,
    ) -> DualRegimeOutput:
        """Return deterministic classes and their degenerate class distributions."""

        predictions = self.predict(features, context=context)
        return DualRegimeOutput(
            predictions=predictions,
            probabilities=DualRegimeProbabilities(
                current=_one_hot(predictions.current),
                anticipated=_one_hot(predictions.anticipated),
            ),
            scores=None,
            uncertainty_kind="deterministic_distribution",
        )

    def describe(self) -> dict[str, Any]:
        """Describe the fixed strategy and fitted class choices."""

        return {
            "model_name": self.name,
            "tuning_required": False,
            "current_class": self._current_class,
            "anticipated_class": self._anticipated_class,
        }

    def save(self, path: str | Path) -> None:
        """Persist the tiny fitted state as readable JSON."""

        self._require_fitted()
        # Exclusive creation protects an existing benchmark artifact from accidental reuse.
        with Path(path).open("x", encoding="utf-8") as model_file:
            json.dump(self.describe(), model_file, indent=2, sort_keys=True)
            model_file.write("\n")

    def state_fingerprint(self) -> str:
        """Identify the fitted choices independently of JSON serialization details."""

        encoded = json.dumps(self.describe(), sort_keys=True).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    @classmethod
    def load(cls, path: str | Path) -> "MajorityClassBaseline":
        """Restore a baseline written by :meth:`save`."""

        with Path(path).open(encoding="utf-8") as model_file:
            state = json.load(model_file)
        # Model identity is checked before accepting class values from a generic JSON file.
        if state.get("model_name") != cls.name:
            raise ValueError("Saved baseline has the wrong model identity.")
        model = cls()
        model._current_class = _validate_class(state.get("current_class"))
        model._anticipated_class = _validate_class(state.get("anticipated_class"))
        return model

    def _require_fitted(self) -> tuple[int, int]:
        """Keep accidental pre-fit evaluation from looking like real output."""

        if self._current_class is None or self._anticipated_class is None:
            raise RuntimeError("Majority baseline must be fitted before prediction.")
        return self._current_class, self._anticipated_class


class PreviousRegimeBaseline:
    """Repeat the prior candle's human regimes as an online reference."""

    name = "previous_regime"

    def fit(
        self,
        features: np.ndarray,
        current_targets: np.ndarray,
        anticipated_targets: np.ndarray,
        *,
        context: PredictionContext | None = None,
    ) -> None:
        """Validate training alignment; this reference has no learned parameters."""

        _validate_fit_arrays(features, current_targets, anticipated_targets)

    def predict(
        self,
        features: np.ndarray,
        *,
        context: PredictionContext | None = None,
    ) -> DualRegimePredictions:
        """Use only labels from the immediately preceding candle."""

        sample_count = _sample_count(features)
        # Prior human labels are passed as audit context rather than market features. This
        # deliberately strong persistence reference is not deployable without annotations.
        if (
            context is None
            or context.previous_current_targets is None
            or context.previous_anticipated_targets is None
        ):
            raise ValueError("Previous-regime prediction requires prior-label context.")
        current = np.asarray(context.previous_current_targets, dtype=np.int64)
        anticipated = np.asarray(context.previous_anticipated_targets, dtype=np.int64)
        # Exact one-dimensional alignment prevents NumPy broadcasting from concealing a
        # missing or duplicated prior-label row.
        if current.shape != (sample_count,) or anticipated.shape != (sample_count,):
            raise ValueError("Prior-label context must align with benchmark samples.")
        if not np.isin(current, (0, 1, 2)).all() or not np.isin(
            anticipated, (0, 1, 2)
        ).all():
            raise ValueError("Prior-label context contains an unknown regime class.")
        return DualRegimePredictions(current=current.copy(), anticipated=anticipated.copy())

    def predict_proba(
        self,
        features: np.ndarray,
        *,
        context: PredictionContext | None = None,
    ) -> DualRegimeProbabilities:
        """Represent repeated prior labels as deterministic probabilities."""

        predictions = self.predict(features, context=context)
        return DualRegimeProbabilities(
            current=_one_hot(predictions.current),
            anticipated=_one_hot(predictions.anticipated),
        )

    def predict_output(
        self,
        features: np.ndarray,
        *,
        context: PredictionContext | None = None,
    ) -> DualRegimeOutput:
        """Return repeated prior labels and deterministic distributions."""

        predictions = self.predict(features, context=context)
        return DualRegimeOutput(
            predictions=predictions,
            probabilities=DualRegimeProbabilities(
                current=_one_hot(predictions.current),
                anticipated=_one_hot(predictions.anticipated),
            ),
            scores=None,
            uncertainty_kind="deterministic_distribution",
        )

    def describe(self) -> dict[str, Any]:
        """Identify the special prior-human-label information used by this baseline."""

        return {
            "model_name": self.name,
            "tuning_required": False,
            "uses_prior_human_label": True,
            "deployable_without_annotations": False,
        }

    def save(self, path: str | Path) -> None:
        """Record the parameter-free strategy for artifact completeness."""

        # Persisting even a parameter-free baseline gives every benchmark row a verifiable
        # model artifact and a uniform publication lifecycle.
        with Path(path).open("x", encoding="utf-8") as model_file:
            json.dump(self.describe(), model_file, indent=2, sort_keys=True)
            model_file.write("\n")

    @classmethod
    def load(cls, path: str | Path) -> "PreviousRegimeBaseline":
        """Restore and validate the parameter-free strategy marker."""

        with Path(path).open(encoding="utf-8") as model_file:
            state = json.load(model_file)
        if state.get("model_name") != cls.name:
            raise ValueError("Saved baseline has the wrong model identity.")
        return cls()

    def state_fingerprint(self) -> str:
        """Identify this parameter-free reference strategy."""

        return hashlib.sha256(b"previous_regime_v1").hexdigest()


def _validate_fit_arrays(
    features: np.ndarray,
    current_targets: np.ndarray,
    anticipated_targets: np.ndarray,
) -> None:
    """Keep baseline alignment under the same contract as learned models."""

    # Baselines must fail on the same malformed corpus as learned models; otherwise their
    # apparently valid scores could come from a different or misaligned evaluation problem.
    sample_count = _sample_count(features)
    current_targets = np.asarray(current_targets)
    anticipated_targets = np.asarray(anticipated_targets)
    if current_targets.shape != (sample_count,) or anticipated_targets.shape != (
        sample_count,
    ):
        raise ValueError("Benchmark targets must align with feature samples.")
    if not np.isin(current_targets, (0, 1, 2)).all() or not np.isin(
        anticipated_targets, (0, 1, 2)
    ).all():
        raise ValueError("Benchmark targets contain an unknown regime class.")


def _sample_count(features: np.ndarray) -> int:
    """Accept tabular or sequential arrays with one nonempty sample axis."""

    features = np.asarray(features)
    if features.ndim not in (2, 3) or features.shape[0] == 0:
        raise ValueError("Benchmark features must contain nonempty tabular or sequence data.")
    return int(features.shape[0])


def _majority_class(targets: np.ndarray) -> int:
    """Use the lowest class index as the deterministic tie breaker."""

    # np.argmax returns the first maximum, making tied training frequencies reproducible.
    return int(np.bincount(np.asarray(targets, dtype=np.int64), minlength=3).argmax())


def _one_hot(predictions: np.ndarray) -> np.ndarray:
    """Convert deterministic predictions into the common probability shape."""

    # These rows express a deterministic strategy, not calibrated confidence estimates.
    probabilities = np.zeros((len(predictions), 3), dtype=np.float64)
    probabilities[np.arange(len(predictions)), predictions] = 1.0
    return probabilities


def _validate_class(value: object) -> int:
    """Validate one decoded JSON class before trusting saved baseline state."""

    if not isinstance(value, int) or value not in (0, 1, 2):
        raise ValueError("Saved baseline contains an unknown regime class.")
    return value
