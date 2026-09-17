"""Expose identical causal values in sequential and flattened forms."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from pricesanity.features.causal_window import CausalWindowCorpus


def sequential_representation(corpus: CausalWindowCorpus) -> np.ndarray:
    """Return samples by time by feature without changing their information."""

    # Contiguity gives every sequence adapter the same dense memory layout without changing the
    # corpus's sample, time, feature ordering.
    return np.ascontiguousarray(corpus.features, dtype=np.float32)


def tabular_representation(corpus: CausalWindowCorpus) -> np.ndarray:
    """Flatten values and, when needed, explicitly append real-history indicators."""

    sample_count = corpus.features.shape[0]

    # Flatten time-major windows consistently so the same lag/feature coordinate occupies one
    # fixed tabular column for every sample.
    values = corpus.features.reshape(sample_count, -1)

    # Fully observed controlled contexts need no availability indicators; appending constant ones
    # would alter their feature dimension without adding information.
    if corpus.valid_history_mask.all():
        return np.ascontiguousarray(values)
    # A zero-valued standardized candle can be real. Appending this mask prevents a tabular
    # model from having to guess whether a zero means average price geometry or absent history.
    return np.ascontiguousarray(
        np.concatenate(
            [values, corpus.valid_history_mask.astype(np.float32)],
            axis=1,
        )
    )


@dataclass(frozen=True)
class ArrayStandardizer:
    """Training-only location and scale for NumPy benchmark representations."""

    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(
        cls,
        training_features: np.ndarray,
        *,
        valid_values: np.ndarray | None = None,
    ) -> "ArrayStandardizer":
        """Fit only values explicitly assigned to model training."""

        training_features = np.asarray(training_features)

        # Fit only on a nonempty numeric array whose complete contents are usable observations or
        # explicitly masked below; invalid values must not enter persisted preprocessing state.
        if training_features.ndim < 2 or training_features.shape[0] == 0:
            raise ValueError("Standardization requires nonempty training features.")
        if not np.issubdtype(training_features.dtype, np.number) or not np.isfinite(
            training_features
        ).all():
            raise ValueError("Standardization requires finite numeric features.")

        if valid_values is None:
            # Fully observed controlled representations give every training value one equal vote.
            mean = training_features.mean(axis=0, dtype=np.float64)
            scale = training_features.std(axis=0, dtype=np.float64)
        else:
            if valid_values.shape != training_features.shape:
                raise ValueError("Validity must align with every market feature.")
            # Absent history is not a market observation. Empty lag columns stay neutral
            # until inference supplies real history, using a unit scale rather than NaN.
            counts = np.maximum(valid_values.sum(axis=0), 1)
            mean = np.where(valid_values, training_features, 0).sum(
                axis=0,
                dtype=np.float64,
            ) / counts
            variance = np.where(
                valid_values,
                (training_features - mean) ** 2,
                0,
            ).sum(axis=0) / counts
            scale = np.sqrt(variance)
        scale = np.where(scale > np.finfo(np.float64).eps, scale, 1.0)

        # Unit scale keeps constant training coordinates neutral and finite at inference instead of
        # dividing by numerical zero.
        return cls(mean=mean, scale=scale)

    def transform(self, features: np.ndarray) -> np.ndarray:
        """Apply frozen training statistics to any matching partition."""

        features = np.asarray(features)

        # Shape validation prevents a standardizer fitted for one context or feature order from
        # being reused on a different representation.
        if features.shape[1:] != self.mean.shape:
            raise ValueError("Features do not match the fitted standardizer shape.")
        transformed = (features - self.mean) / self.scale
        return transformed.astype(np.float32, copy=False)


def fit_unique_candle_standardizer(
    training_sessions: list[pd.DataFrame] | tuple[pd.DataFrame, ...],
    *,
    feature_columns: tuple[str, ...],
) -> ArrayStandardizer:
    """Fit four feature statistics once per unique training candle, before overlap."""

    if not training_sessions:
        raise ValueError("Controlled standardization requires training sessions.")
    # Window rows repeat interior candles. Fitting before window construction gives every real
    # candle one vote instead of weighting it by how many overlapping windows contain it.
    candle_features = np.concatenate([
        session.loc[:, feature_columns].to_numpy(dtype=np.float32, copy=True)
        for session in training_sessions
    ])

    # Fit only after concatenating the training sessions; evaluation candles never influence these
    # persisted location and scale statistics.
    return ArrayStandardizer.fit(candle_features)


def transform_sessions(
    sessions: list[pd.DataFrame] | tuple[pd.DataFrame, ...],
    standardizer: ArrayStandardizer,
    *,
    feature_columns: tuple[str, ...],
) -> tuple[pd.DataFrame, ...]:
    """Copy sessions and apply one already-fitted candle-level transformation."""

    transformed_sessions = []

    # Copy each frame before transformation so cached raw snapshot sessions remain immutable and
    # can support other folds with independently fitted training statistics.
    for session in sessions:
        transformed = session.copy()
        transformed.loc[:, feature_columns] = standardizer.transform(
            transformed.loc[:, feature_columns].to_numpy(dtype=np.float32)
        )
        transformed_sessions.append(transformed)
    return tuple(transformed_sessions)
