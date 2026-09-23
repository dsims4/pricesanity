"""Paired learning-curve evidence, with practical thresholds separate from uncertainty."""

from collections.abc import Sequence

import numpy as np
import pandas as pd

from pricesanity.benchmark.analysis import (
    _confusion_f1,
    _prediction_arrays,
    _session_confusions,
)


SCORE_NAMES = ("current_macro_f1", "anticipated_macro_f1", "mean_head_macro_f1")
IDENTITY_COLUMNS = (
    "candlestick_id", "session_index", "human_current_regime", "human_anticipated_regime",
)


def paired_curve_difference(
    earlier: Sequence[pd.DataFrame],
    later: Sequence[pd.DataFrame],
    *,
    repetitions: int = 2000,
    random_seed: int = 42,
) -> dict:
    """Bootstrap pooled F1 differences with common session draws across sizes and seeds.

    Seeds are held fixed, not resampled or pooled as additional candles. Each draw scores
    each seed independently, then averages seed-level F1. Its interval describes session
    uncertainty conditional on these seeds; between-seed variability is reported separately.
    """

    if not earlier or len(earlier) != len(later) or repetitions <= 0:
        raise ValueError("Paired curves require matching nonempty seed sets and repetitions.")
    reference = earlier[0].loc[:, IDENTITY_COLUMNS]
    if reference.candlestick_id.duplicated().any():
        raise ValueError("Evaluation candle identities must be unique.")
    session_ids = tuple(dict.fromkeys(reference.session_index))
    if len(session_ids) < 2:
        raise ValueError("Paired uncertainty requires at least two evaluation sessions.")
    matrices = []
    for predictions in (*earlier, *later):
        if not reference.equals(predictions.loc[:, IDENTITY_COLUMNS]):
            raise ValueError("Every curve point must have identical ordered evaluation targets.")
        matrices.append(_session_confusions(_prediction_arrays(predictions), session_ids))
    seed_count = len(earlier)
    counts = np.asarray(matrices)

    def scores(session_positions):
        # Session confusion counts exactly reproduce pooled candle F1, including repeated
        # copies of a sampled session. Averaging per-session F1 would be a different metric.
        head_scores = _confusion_f1(counts[:, session_positions].sum(axis=1))
        return np.column_stack((head_scores, head_scores.mean(axis=1)))

    observed = scores(np.arange(len(session_ids)))
    seed_differences = observed[seed_count:] - observed[:seed_count]
    generator = np.random.default_rng(random_seed)
    differences = np.empty((repetitions, 3))
    for repetition in range(repetitions):
        sampled = generator.integers(0, len(session_ids), size=len(session_ids))
        sampled_scores = scores(sampled)
        differences[repetition] = (
            sampled_scores[seed_count:] - sampled_scores[:seed_count]
        ).mean(axis=0)
    bounds = np.quantile(differences, [0.025, 0.975], axis=0)
    return {
        "session_count": len(session_ids),
        "bootstrap_repetitions": repetitions,
        "bootstrap_seed": random_seed,
        "uncertainty": "paired whole-session percentile 95% CI, conditional on fixed seeds",
        "scores": {
            name: {
                "gain": float(seed_differences[:, index].mean()),
                "lower": float(bounds[0, index]),
                "upper": float(bounds[1, index]),
                "seed_difference_sd": (
                    float(seed_differences[:, index].std(ddof=0)) if seed_count > 1 else None
                ),
            }
            for index, name in enumerate(SCORE_NAMES)
        },
    }


def summarize_curve_point(predictions: Sequence[pd.DataFrame]) -> dict:
    """Report seed means and observed-seed SDs for each head and their equal-weight mean."""

    scores = []
    for frame in predictions:
        arrays = _prediction_arrays(frame)
        sessions = tuple(dict.fromkeys(arrays["session_index"]))
        heads = _confusion_f1(_session_confusions(arrays, sessions).sum(axis=0))
        scores.append([*heads, heads.mean()])
    values = np.asarray(scores)
    return {
        "seed_count": len(values),
        **{
            name: float(values[:, index].mean())
            for index, name in enumerate(SCORE_NAMES)
        },
        **{
            name + "_seed_sd": (
                float(values[:, index].std(ddof=0)) if len(values) > 1 else None
            ) for index, name in enumerate(SCORE_NAMES)
        },
    }


def assess_curve(
    gains: Sequence[dict],
    *,
    evaluation_sessions: int,
    stochastic: bool,
    seed_count: int,
    latest_seed_sd: float | None,
    meaningful_gain: float = 0.01,
    small_gain: float = 0.005,
    reference_only: bool = False,
) -> dict:
    """Interpret two recent gains per 100 sessions without predicting an unseen result."""

    if not np.isfinite([meaningful_gain, small_gain]).all() or not (
        0 <= small_gain < meaningful_gain <= 1
    ):
        raise ValueError("Thresholds require 0 <= small-gain < meaningful-gain <= 1.")
    label = "INCONCLUSIVE"
    reason = "Two adjacent recent gains are needed to judge the direction of the curve."
    if reference_only:
        reason = "A reference baseline cannot establish whether learned models need more data."
    elif evaluation_sessions < 20:
        reason = "Fewer than 20 evaluation sessions cannot support this assessment."
    elif stochastic and seed_count < 2:
        reason = "A single stochastic seed cannot establish seed stability."
    elif len(gains) >= 2:
        previous, latest = gains[-2:]
        previous_scores = previous["per_100_sessions"]
        latest_scores = latest["per_100_sessions"]
        mean = latest_scores["mean_head_macro_f1"]
        seed_noise = max(
            latest_seed_sd or 0,
            latest["scores"]["mean_head_macro_f1"]["seed_difference_sd"] or 0,
        )
        # Seed SD is on the observed interval's scale, whereas configurable practical gains
        # use 100 additional sessions. Compare variability to the actual observed gain.
        actual_gain = abs(latest["scores"]["mean_head_macro_f1"]["gain"])
        noisy_seeds = stochastic and seed_noise >= max(actual_gain, small_gain)
        wide_interval = mean["upper"] - mean["lower"] > 4 * meaningful_gain
        if noisy_seeds or wide_interval:
            reason = "Seed variability or session uncertainty is large relative to recent gains."
        elif all(
            abs(point[name]["gain"]) <= small_gain
            for point in (previous_scores, latest_scores) for name in SCORE_NAMES
        ) and all(latest_scores[name]["upper"] <= small_gain for name in SCORE_NAMES):
            label = "PLAUSIBLY_PLATEAUING"
            reason = (
                "Both recent gains and the latest upper bounds are below the small-gain limit."
            )
        elif any(
            latest_scores[name]["gain"] >= meaningful_gain
            and latest_scores[name]["lower"] > 0
            and previous_scores[name]["gain"] >= 0
            for name in SCORE_NAMES
        ) and all(latest_scores[name]["gain"] >= -small_gain for name in SCORE_NAMES):
            label = "STILL_RISING"
            reason = (
                "A recent gain clears the practical threshold with positive paired support; "
                "additional annotation toward 1,000 is worth testing."
            )
        else:
            reason = "Recent gains, head behavior, or paired bounds do not support a clear trend."
    return {
        "label": label,
        "reason": reason,
        "thresholds": {"meaningful_gain": meaningful_gain, "small_gain": small_gain},
        "threshold_units": "absolute macro-F1 gain per 100 additional training sessions",
        "interpretation": (
            f"Under the configured {meaningful_gain:g} practical-improvement threshold "
            "per 100 additional sessions. "
            "This is a user-defined criterion, not a statistical law or a final benchmark. "
            "The unseen result at 1,000 sessions is unknown."
        ),
        "evaluation_strength": "preliminary" if evaluation_sessions < 50 else "stronger",
        "projection": "Not fitted; directly observed paired gains are the primary evidence.",
    }
