"""Adapt two independent scikit-learn classifiers to the common two-head task."""

from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
import pickle
from typing import Any

import numpy as np

from pricesanity.features.representations import ArrayStandardizer
from pricesanity.models.protocol import (
    DualRegimeOutput,
    DualRegimePredictions,
    DualRegimeProbabilities,
    DualRegimeScores,
    PredictionContext,
)


@dataclass
class SklearnDualHeadAdapter:
    """Fit matching estimator instances for current and anticipated regimes."""

    model_name: str
    estimator_factory: Callable[[], Any]
    configuration: Mapping[str, Any]
    standardize: bool = True
    output_kind: str = "probability_estimate"

    def __post_init__(self) -> None:
        self._current_estimator: Any | None = None
        self._anticipated_estimator: Any | None = None
        self._standardizer: ArrayStandardizer | None = None

    @property
    def name(self) -> str:
        """Return the registry identity rather than the library class name."""

        return self.model_name

    def fit(
        self,
        features: np.ndarray,
        current_targets: np.ndarray,
        anticipated_targets: np.ndarray,
        *,
        context: PredictionContext | None = None,
    ) -> None:
        """Fit preprocessing and both estimators using training rows only."""

        features = _tabular_features(features)
        current_targets = np.asarray(current_targets, dtype=np.int64)
        anticipated_targets = np.asarray(anticipated_targets, dtype=np.int64)
        if current_targets.shape != (len(features),) or anticipated_targets.shape != (
            len(features),
        ):
            raise ValueError("Estimator targets must align with feature samples.")

        if self.standardize:
            market, validity, indicators = _market_features(features, context)
            self._standardizer = ArrayStandardizer.fit(market, valid_values=validity)
            fitted_features = self._transform(features, context)
        else:
            fitted_features = features

        # Separate instances prevent one target fit from replacing the other's state.
        self._current_estimator = self.estimator_factory()
        self._anticipated_estimator = deepcopy(self._current_estimator)
        self._current_estimator.fit(fitted_features, current_targets)
        self._anticipated_estimator.fit(fitted_features, anticipated_targets)

    def predict(
        self,
        features: np.ndarray,
        *,
        context: PredictionContext | None = None,
    ) -> DualRegimePredictions:
        """Decode both fitted estimators to aligned class vectors."""

        return self.predict_output(features, context=context).predictions

    def predict_proba(
        self,
        features: np.ndarray,
        *,
        context: PredictionContext | None = None,
    ) -> DualRegimeProbabilities:
        """Normalize estimator-specific class order into Bull, Bear, Range columns."""

        output = self.predict_output(features, context=context)
        if output.probabilities is None:
            raise ValueError(
                f"{self.model_name} exposes uncalibrated scores, not probabilities."
            )
        return output.probabilities

    def predict_output(
        self,
        features: np.ndarray,
        *,
        context: PredictionContext | None = None,
    ) -> DualRegimeOutput:
        """Run each fitted head once and preserve its native prediction semantics."""

        transformed, current_estimator, anticipated_estimator = self._prepare(features, context)
        current_predictions = np.asarray(
            current_estimator.predict(transformed), dtype=np.int64
        )
        anticipated_predictions = np.asarray(
            anticipated_estimator.predict(transformed), dtype=np.int64
        )
        predictions = DualRegimePredictions(
            current=current_predictions,
            anticipated=anticipated_predictions,
        )

        if self.output_kind == "uncalibrated_score":
            # SVC probability=True performs an internal cross-validation calibration whose
            # row split is inappropriate for overlapping time-series windows. We keep native
            # SVC.predict classes and store only its uncalibrated decision-function scores.
            return DualRegimeOutput(
                predictions=predictions,
                probabilities=None,
                scores=DualRegimeScores(
                    current=_ordered_decision_scores(current_estimator, transformed),
                    anticipated=_ordered_decision_scores(
                        anticipated_estimator, transformed
                    ),
                ),
                uncertainty_kind="uncalibrated_decision_score",
            )

        return DualRegimeOutput(
            predictions=predictions,
            probabilities=DualRegimeProbabilities(
                current=_ordered_probabilities(current_estimator, transformed),
                anticipated=_ordered_probabilities(
                    anticipated_estimator, transformed
                ),
            ),
            scores=None,
            uncertainty_kind=self.output_kind,
        )

    def describe(self) -> dict[str, Any]:
        """Record readable estimator choices independently of pickle state."""

        return {
            "model_name": self.model_name,
            "adapter": "sklearn_dual_head_v1",
            "configuration": dict(self.configuration),
            "standardize": self.standardize,
            "output_kind": self.output_kind,
        }

    def save(self, path: str | Path) -> None:
        """Write internal model state; run checksums protect later loading."""

        self._require_fitted()
        state = {
            "format_version": 1,
            "model_name": self.model_name,
            "configuration": dict(self.configuration),
            "standardize": self.standardize,
            "output_kind": self.output_kind,
            "current_estimator": self._current_estimator,
            "anticipated_estimator": self._anticipated_estimator,
            "standardizer": self._standardizer,
        }
        with Path(path).open("xb") as model_file:
            # The construction factory may be a local callable and is unnecessary after fit.
            # Persist only fitted state so artifacts remain serializable across adapters.
            pickle.dump(state, model_file, protocol=pickle.HIGHEST_PROTOCOL)

    def state_fingerprint(self) -> str:
        """Hash fitted estimator and standardizer state for coherent crash recovery."""

        import hashlib

        current_estimator, anticipated_estimator = self._require_fitted()
        payload = pickle.dumps(
            (current_estimator, anticipated_estimator, self._standardizer),
            protocol=pickle.HIGHEST_PROTOCOL,
        )
        return hashlib.sha256(payload).hexdigest()

    @classmethod
    def load(cls, path: str | Path) -> "SklearnDualHeadAdapter":
        """Load only trusted local artifacts after external checksum validation."""

        with Path(path).open("rb") as model_file:
            state = pickle.load(model_file)
        if not isinstance(state, dict) or state.get("format_version") != 1:
            raise ValueError("Saved estimator does not contain supported adapter state.")
        model = cls(
            model_name=str(state["model_name"]),
            estimator_factory=_loaded_model_cannot_refit,
            configuration=dict(state["configuration"]),
            standardize=bool(state["standardize"]),
            output_kind=str(state.get("output_kind", "probability_estimate")),
        )
        model._current_estimator = state["current_estimator"]
        model._anticipated_estimator = state["anticipated_estimator"]
        model._standardizer = state["standardizer"]
        model._require_fitted()
        return model

    def _prepare(self, features: np.ndarray, context: PredictionContext | None = None) -> tuple[np.ndarray, Any, Any]:
        """Apply frozen training preprocessing before inference."""

        current_estimator, anticipated_estimator = self._require_fitted()
        features = _tabular_features(features)
        if self.standardize:
            if self._standardizer is None:
                raise RuntimeError("Fitted estimator is missing its standardizer.")
            features = self._transform(features, context)
        return features, current_estimator, anticipated_estimator

    def _transform(self, features: np.ndarray, context: PredictionContext | None) -> np.ndarray:
        market, validity, indicators = _market_features(features, context)
        transformed = self._standardizer.transform(market)
        if validity is not None:
            transformed = np.where(validity, transformed, 0).astype(np.float32)
        # Boolean history indicators carry availability, not OHLC geometry. Keep 0/1 intact.
        return np.concatenate([transformed, indicators], axis=1) if indicators is not None else transformed

    def _require_fitted(self) -> tuple[Any, Any]:
        """Reject partially fitted or uninitialized adapters."""

        if self._current_estimator is None or self._anticipated_estimator is None:
            raise RuntimeError("Estimator adapter must be fitted before use.")
        return self._current_estimator, self._anticipated_estimator


def _tabular_features(features: np.ndarray) -> np.ndarray:
    """Require already flattened data so representation choices remain outside adapters."""

    features = np.asarray(features, dtype=np.float32)
    if features.ndim != 2 or features.shape[0] == 0 or not np.isfinite(features).all():
        raise ValueError("Scikit-learn adapters require finite nonempty tabular features.")
    return features


def _market_features(features: np.ndarray, context: PredictionContext | None):
    if context is None or context.valid_history_mask is None:
        return features, None, None
    history = np.asarray(context.valid_history_mask, dtype=bool)
    length = history.shape[1]
    has_indicators = features.shape[1] == length * 5
    market = features[:, :length * 4] if has_indicators else features
    if market.shape[1] != length * 4:
        raise ValueError("Tabular OHLC and history mask dimensions disagree.")
    return market, np.repeat(history, 4, axis=1), features[:, length * 4:] if has_indicators else None


def _ordered_probabilities(estimator: Any, features: np.ndarray) -> np.ndarray:
    """Place estimator probabilities into the shared three-class column order."""

    if not hasattr(estimator, "predict_proba") or not hasattr(estimator, "classes_"):
        raise ValueError("Estimator must support predict_proba and expose fitted classes.")
    raw_probabilities = np.asarray(estimator.predict_proba(features), dtype=np.float64)
    classes = np.asarray(estimator.classes_, dtype=np.int64)
    if raw_probabilities.shape != (len(features), len(classes)):
        raise ValueError("Estimator returned malformed class probabilities.")
    probabilities = np.zeros((len(features), 3), dtype=np.float64)
    for source_index, class_index in enumerate(classes):
        if class_index not in (0, 1, 2):
            raise ValueError("Estimator returned an unknown regime class.")
        probabilities[:, class_index] = raw_probabilities[:, source_index]
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("Estimator probabilities do not sum to one.")
    return probabilities


def _ordered_decision_scores(estimator: Any, features: np.ndarray) -> np.ndarray:
    """Place uncalibrated decision values into the shared class order."""

    if not hasattr(estimator, "decision_function") or not hasattr(estimator, "classes_"):
        raise ValueError("Score-only estimator must expose decision_function and classes.")
    raw_scores = np.asarray(estimator.decision_function(features), dtype=np.float64)
    classes = np.asarray(estimator.classes_, dtype=np.int64)
    if raw_scores.ndim == 1 and len(classes) == 2:
        raw_scores = np.column_stack((-raw_scores, raw_scores))
    if raw_scores.shape != (len(features), len(classes)):
        raise ValueError("Estimator returned malformed class scores.")
    scores = np.full((len(features), 3), np.nan, dtype=np.float64)
    for source_index, class_index in enumerate(classes):
        if class_index not in (0, 1, 2):
            raise ValueError("Estimator returned an unknown regime class.")
        scores[:, class_index] = raw_scores[:, source_index]
    return scores


def _loaded_model_cannot_refit() -> Any:
    """Keep inference artifacts from silently pretending to retain construction code."""

    raise RuntimeError("A loaded estimator artifact is inference-only.")
