"""Session-aware uncertainty helpers for dependent candlestick predictions."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from pricesanity.benchmark.metrics import classification_metrics


@dataclass(frozen=True)
class BootstrapInterval:
    """Observed statistic and session-resampled 95 percent interval."""

    estimate: float
    lower: float
    upper: float
    session_count: int
    bootstrap_repetitions: int


@dataclass(frozen=True)
class PooledF1Bootstrap:
    """Cluster-bootstrap intervals for the three pooled benchmark scores."""

    current: BootstrapInterval
    anticipated: BootstrapInterval
    mean_head: BootstrapInterval


def per_session_macro_f1(predictions: pd.DataFrame, *, head: str) -> pd.DataFrame:
    """Calculate macro-F1 independently inside each trading session."""

    # Limit callers to the two persisted prediction contracts so column construction cannot
    # select arbitrary data from the artifact frame.
    if head not in {"current", "anticipated"}:
        raise ValueError("Prediction head must be current or anticipated.")

    # Session identity is required because these rows are the independent bootstrap units, not
    # merely a grouping convenience for candle-level scores.
    required = {
        "session_index",
        f"human_{head}_regime",
        f"predicted_{head}_regime",
    }
    missing = required.difference(predictions.columns)
    if missing:
        raise ValueError("Predictions are missing session-analysis columns.")

    # Reuse the benchmark's fixed class order before delegating to the shared metric function.
    labels = {"bull": 0, "bear": 1, "range": 2}
    rows = []

    # Preserve chronological session ordering so paired model tables can later prove that their
    # bootstrap units align exactly rather than relying on a join that might drop sessions.
    for session_index, session in predictions.groupby("session_index", sort=True):
        human = session[f"human_{head}_regime"].map(labels).to_numpy()
        predicted = session[f"predicted_{head}_regime"].map(labels).to_numpy()
        rows.append({
            "session_index": int(session_index),
            "macro_f1": classification_metrics(human, predicted).macro_f1,
            "sample_count": len(session),
        })
    return pd.DataFrame(rows)


def bootstrap_session_mean(
    session_values: np.ndarray,
    *,
    repetitions: int = 2000,
    random_seed: int = 42,
) -> BootstrapInterval:
    """Resample whole sessions so overlapping candles are never treated as independent."""

    # Require multiple finite session summaries because resampling one session would create a
    # degenerate interval that overstates the available evidence.
    values = np.asarray(session_values, dtype=np.float64)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("Session bootstrap requires at least two finite values.")
    if repetitions <= 0:
        raise ValueError("Bootstrap repetitions must be positive.")

    # A local seeded generator makes the reported interval reproducible without changing random
    # state used by model training elsewhere in the process.
    generator = np.random.default_rng(random_seed)

    # Draw whole sessions with replacement. Candles within one chart remain dependent and must
    # never receive independent bootstrap weights.
    samples = generator.choice(values, size=(repetitions, len(values)), replace=True)
    means = samples.mean(axis=1)

    # Percentile bounds describe the empirical session-resampled distribution while the point
    # estimate remains the observed mean over the original sessions.
    lower, upper = np.quantile(means, [0.025, 0.975])
    return BootstrapInterval(
        estimate=float(values.mean()),
        lower=float(lower),
        upper=float(upper),
        session_count=len(values),
        bootstrap_repetitions=repetitions,
    )


def paired_session_bootstrap_difference(
    first: pd.DataFrame,
    second: pd.DataFrame,
    *,
    repetitions: int = 2000,
    random_seed: int = 42,
) -> BootstrapInterval:
    """Estimate a paired model difference only after exact session alignment."""

    # Pairing is meaningful only when each row identifies the same chronological session in both
    # model tables; an inner join could conceal a missing or reordered evaluation session.
    required = {"session_index", "macro_f1"}
    if not required.issubset(first.columns) or not required.issubset(second.columns):
        raise ValueError("Paired bootstrap inputs need session_index and macro_f1.")
    if first["session_index"].tolist() != second["session_index"].tolist():
        raise ValueError("Paired model comparison requires identical ordered sessions.")

    # Resample within-session score differences so common easy or difficult sessions cancel from
    # the comparison rather than inflating uncertainty as two independent samples.
    differences = (
        first["macro_f1"].to_numpy(dtype=float)
        - second["macro_f1"].to_numpy(dtype=float)
    )
    return bootstrap_session_mean(
        differences,
        repetitions=repetitions,
        random_seed=random_seed,
    )


def cluster_bootstrap_pooled_f1(
    predictions: pd.DataFrame,
    *,
    repetitions: int = 2000,
    random_seed: int = 42,
) -> PooledF1Bootstrap:
    """Resample sessions and recompute pooled candle-level macro-F1 each time."""

    # Decode and validate persisted labels once; every repetition must use the same exact candle
    # population and class ordering as the official pooled benchmark metric.
    arrays = _prediction_arrays(predictions)

    # dict preserves first appearance, which keeps session order deterministic while collapsing
    # the repeated candle rows into bootstrap cluster identities.
    session_ids = tuple(dict.fromkeys(arrays["session_index"].tolist()))
    if len(session_ids) < 2 or repetitions <= 0:
        raise ValueError("Cluster bootstrap needs two sessions and positive repetitions.")
    matrices = _session_confusions(arrays, session_ids)

    # Separate arrays retain head-level uncertainty before their equally weighted mean is formed.
    generator = np.random.default_rng(random_seed)
    current_values = np.empty(repetitions)
    anticipated_values = np.empty(repetitions)
    for repetition in range(repetitions):
        # Resampling cached confusion counts is equivalent to resampling every candle row,
        # while avoiding a costly DataFrame reconstruction on each bootstrap repetition.
        sampled = generator.choice(len(session_ids), size=len(session_ids), replace=True)
        scores = _confusion_f1(matrices[sampled].sum(axis=0))
        current_values[repetition], anticipated_values[repetition] = scores

    # Point estimates come from the observed, unresampled rows; the bootstrap samples supply only
    # the uncertainty bounds around those canonical scores.
    current_estimate = classification_metrics(
        arrays["human_current"], arrays["predicted_current"]
    ).macro_f1
    anticipated_estimate = classification_metrics(
        arrays["human_anticipated"], arrays["predicted_anticipated"]
    ).macro_f1
    return PooledF1Bootstrap(
        current=_interval(current_estimate, current_values, len(session_ids)),
        anticipated=_interval(
            anticipated_estimate, anticipated_values, len(session_ids)
        ),
        mean_head=_interval(
            (current_estimate + anticipated_estimate) / 2.0,
            (current_values + anticipated_values) / 2.0,
            len(session_ids),
        ),
    )


def paired_cluster_bootstrap_difference(
    first: pd.DataFrame,
    second: pd.DataFrame,
    *,
    repetitions: int = 2000,
    random_seed: int = 42,
) -> BootstrapInterval:
    """Resample the same sessions and compare pooled mean-head macro-F1."""

    # Exact ordered candle identity is stricter than matching sessions alone. It ensures both
    # models contribute predictions for the identical target rows before paired resampling.
    identity_columns = ["candlestick_id", "session_index"]
    if not first[identity_columns].equals(second[identity_columns]):
        raise ValueError("Paired pooled bootstrap requires identical ordered candles.")
    first_arrays = _prediction_arrays(first)
    second_arrays = _prediction_arrays(second)

    session_ids = tuple(dict.fromkeys(first_arrays["session_index"].tolist()))
    if len(session_ids) < 2:
        raise ValueError("Paired pooled bootstrap requires at least two sessions.")
    if repetitions <= 0:
        raise ValueError("Bootstrap repetitions must be positive.")
    for head in ("current", "anticipated"):
        # A paired model comparison requires shared ground truth. Different human labels would
        # describe different experiments even if candle IDs happened to match.
        if not np.array_equal(first_arrays[f"human_{head}"], second_arrays[f"human_{head}"]):
            raise ValueError("Paired bootstrap requires identical human targets.")

    # Cache each model's session-level sufficient statistics before resampling so both models use
    # the same cluster draw without rebuilding candle frames in every repetition.
    first_matrices = _session_confusions(first_arrays, session_ids)
    second_matrices = _session_confusions(second_arrays, session_ids)
    generator = np.random.default_rng(random_seed)
    differences = np.empty(repetitions)
    for repetition in range(repetitions):
        # The same session draw preserves pairing, including repeated copies of a session.
        sampled = generator.choice(len(session_ids), size=len(session_ids), replace=True)
        differences[repetition] = (
            _confusion_f1(first_matrices[sampled].sum(axis=0)).mean()
            - _confusion_f1(second_matrices[sampled].sum(axis=0)).mean()
        )
    all_positions = np.arange(len(first))

    # The reported point difference uses every observed candle once; repeated session draws are
    # reserved for the uncertainty distribution.
    estimate = (
        _mean_head_score(first_arrays, all_positions)
        - _mean_head_score(second_arrays, all_positions)
    )
    return _interval(estimate, differences, len(session_ids))


def _prediction_arrays(predictions: pd.DataFrame) -> dict[str, np.ndarray]:
    """Decode persisted labels once for repeated cluster calculations."""

    # Every required identity and label must be present before decoding. Partial artifact rows
    # cannot define the same evaluation population as the published benchmark result.
    required = {
        "candlestick_id", "session_index", "human_current_regime",
        "predicted_current_regime", "human_anticipated_regime",
        "predicted_anticipated_regime",
    }
    if not required.issubset(predictions.columns) or predictions.empty:
        raise ValueError("Cluster bootstrap predictions are incomplete.")

    labels = {"bull": 0, "bear": 1, "range": 2}
    arrays = {"session_index": predictions["session_index"].to_numpy()}

    # Decode all four truth/prediction streams with one fixed mapping so current and anticipated
    # confusion matrices retain identical class axes.
    for name in (
        "human_current", "predicted_current",
        "human_anticipated", "predicted_anticipated",
    ):
        values = predictions[f"{name}_regime"].map(labels)
        if values.isna().any():
            raise ValueError("Cluster bootstrap contains an unknown regime label.")
        arrays[name] = values.to_numpy(dtype=np.int64)
    return arrays


def _mean_head_score(arrays: dict[str, np.ndarray], positions: np.ndarray) -> float:
    """Score both annotation questions equally on one selected candle population."""

    # Score each head on the same selected candle positions before averaging. This prevents one
    # head from receiving a larger or easier population inside a paired difference.
    current = classification_metrics(
        arrays["human_current"][positions], arrays["predicted_current"][positions]
    ).macro_f1
    anticipated = classification_metrics(
        arrays["human_anticipated"][positions],
        arrays["predicted_anticipated"][positions],
    ).macro_f1
    return (current + anticipated) / 2.0


def _interval(
    estimate: float,
    bootstrap_values: np.ndarray,
    session_count: int,
) -> BootstrapInterval:
    """Summarize an empirical bootstrap distribution with percentile bounds."""

    # Keep the observed estimate separate from percentile bounds; substituting the bootstrap mean
    # would make the reported point score depend on the arbitrary repetition count and seed.
    lower, upper = np.quantile(bootstrap_values, [0.025, 0.975])
    return BootstrapInterval(
        estimate=float(estimate), lower=float(lower), upper=float(upper),
        session_count=session_count, bootstrap_repetitions=len(bootstrap_values),
    )


def _session_confusions(arrays: dict, session_ids: tuple) -> np.ndarray:
    """Three-class confusion counts are exact sufficient statistics for pooled F1."""

    # Map arbitrary persisted session identifiers onto a dense axis used only for accumulation.
    indices = {value: index for index, value in enumerate(session_ids)}
    row_sessions = np.array([indices[value] for value in arrays["session_index"]])

    # Store one 3×3 matrix per session and head. These counts are sufficient to reconstruct exact
    # pooled macro-F1 after any whole-session bootstrap draw.
    matrices = np.zeros((len(session_ids), 2, 3, 3), dtype=np.int64)
    for head_index, head in enumerate(("current", "anticipated")):
        np.add.at(matrices[:, head_index],
                  (row_sessions, arrays[f"human_{head}"], arrays[f"predicted_{head}"]), 1)
    return matrices


def _confusion_f1(matrices: np.ndarray) -> np.ndarray:
    """Recover macro-F1 directly from one or more two-head confusion matrices."""

    # Combine predicted and human class totals in one denominator, which is algebraically the
    # per-class F1 formula and preserves the shared zero-division policy for absent classes.
    true_positive = np.diagonal(matrices, axis1=-2, axis2=-1)
    denominator = matrices.sum(axis=-1) + matrices.sum(axis=-2)
    per_class = np.divide(2.0 * true_positive, denominator,
                          out=np.zeros_like(denominator, dtype=float), where=denominator != 0)
    return per_class.mean(axis=-1)
