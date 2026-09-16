"""Thin artifact-only tables used by notebooks and exploratory reports."""

from pathlib import Path
from typing import Any

import pandas as pd

from pricesanity.benchmark.analysis import cluster_bootstrap_pooled_f1, per_session_macro_f1
from pricesanity.benchmark.artifacts import load_benchmark_run
from pricesanity.benchmark.report import collect_aggregated_runs, collect_completed_runs


def final_report_tables(
    artifact_root: str | Path,
    *,
    expected_stochastic_seed_count: int = 3,
) -> dict[str, pd.DataFrame]:
    """Build reusable final-report sections, returning empty tables before a study exists."""

    raw = collect_completed_runs(artifact_root)
    if raw.empty:
        empty = pd.DataFrame()
        return {
            name: empty.copy()
            for name in (
                "leaderboard", "controlled", "best_of_family", "baseline_deltas",
                "learning_curves", "seed_variation", "efficiency", "final_holdout",
            )
        }
    leaderboard = collect_aggregated_runs(
        artifact_root,
        expected_stochastic_seed_count=expected_stochastic_seed_count,
    )
    learning = raw.loc[raw["run_name"].astype(str).str.startswith("train_")].copy()
    if not learning.empty:
        learning["training_session_count"] = (
            learning["run_name"].str.removeprefix("train_").astype(int)
        )
    final = raw.loc[raw["run_name"].astype(str).str.startswith("final_seed_")].copy()
    return {
        "leaderboard": leaderboard,
        "controlled": leaderboard.loc[leaderboard["track"] == "controlled"].copy(),
        "best_of_family": leaderboard.loc[
            leaderboard["track"] == "best_of_family"
        ].copy(),
        "baseline_deltas": leaderboard.loc[:, [
            column for column in (
                "track", "model_name", "mean_head_macro_f1",
                "delta_vs_majority", "delta_vs_persistence",
            ) if column in leaderboard
        ]].copy(),
        "learning_curves": learning,
        "seed_variation": final.loc[:, [
            column for column in (
                "track", "model_name", "seed", "current_macro_f1",
                "anticipated_macro_f1", "mean_head_macro_f1",
            ) if column in final
        ]].copy(),
        "efficiency": final.loc[:, [
            column for column in (
                "track", "model_name", "seed", "training_seconds",
                "inference_seconds", "inference_samples_per_second",
                "serialized_model_bytes", "device", "hardware_fingerprint",
            ) if column in final
        ]].copy(),
        "final_holdout": final,
    }


def class_distribution(run_directory: str | Path) -> pd.DataFrame:
    """Count human labels in one exact persisted evaluation population."""

    _, predictions, _ = load_benchmark_run(run_directory)
    rows = []
    for head in ("current", "anticipated"):
        counts = predictions[f"human_{head}_regime"].value_counts()
        for regime, count in counts.items():
            rows.append({"head": head, "regime": regime, "count": int(count)})
    return pd.DataFrame(rows)


def error_analysis_tables(run_directory: str | Path) -> dict[str, Any]:
    """Collect confusion, class, transition, difficulty, and bootstrap evidence."""

    metadata, predictions, metrics = load_benchmark_run(run_directory)
    session_scores = per_session_macro_f1(predictions, head="current").rename(
        columns={"macro_f1": "current_macro_f1"}
    ).merge(
        per_session_macro_f1(predictions, head="anticipated").rename(
            columns={"macro_f1": "anticipated_macro_f1"}
        ),
        on=["session_index", "sample_count"],
    )
    session_scores["mean_head_macro_f1"] = (
        session_scores["current_macro_f1"]
        + session_scores["anticipated_macro_f1"]
    ) / 2.0
    return {
        "metadata": metadata,
        "confusion": {
            head: pd.DataFrame(
                metrics[head]["confusion_matrix"],
                index=["human_bull", "human_bear", "human_range"],
                columns=["pred_bull", "pred_bear", "pred_range"],
            )
            for head in ("current", "anticipated")
        },
        "per_class": pd.concat({
            head: pd.DataFrame(metrics[head]["per_class"]).T
            for head in ("current", "anticipated")
        }, names=["head", "regime"]),
        "transition_neighborhoods": metrics["transition_neighborhoods"],
        "hardest_sessions": session_scores.sort_values("mean_head_macro_f1").head(20),
        "head_difficulty": pd.DataFrame([{
            "current_macro_f1": metrics["current"]["macro_f1"],
            "anticipated_macro_f1": metrics["anticipated"]["macro_f1"],
            "anticipated_minus_current": (
                metrics["anticipated"]["macro_f1"] - metrics["current"]["macro_f1"]
            ),
        }]),
        "pooled_session_bootstrap": cluster_bootstrap_pooled_f1(predictions),
        "predictions": predictions,
    }


def model_disagreements(first_run: Any, second_run: Any) -> pd.DataFrame:
    """Return exact aligned candles where either prediction head disagrees."""

    first_directory = getattr(first_run, "directory", first_run)
    second_directory = getattr(second_run, "directory", second_run)
    _, first, _ = load_benchmark_run(first_directory)
    _, second, _ = load_benchmark_run(second_directory)
    identity_columns = [
        "candlestick_id", "timestamp", "session_date", "session_index",
        "candle_position", "human_current_regime", "human_anticipated_regime",
    ]
    if not first[identity_columns].equals(second[identity_columns]):
        raise ValueError("Model disagreement requires exact ordered candle alignment.")
    disagrees = (
        first["predicted_current_regime"].ne(second["predicted_current_regime"])
        | first["predicted_anticipated_regime"].ne(
            second["predicted_anticipated_regime"]
        )
    )
    result = first.loc[disagrees, [
        "candlestick_id", "timestamp", "session_date", "candle_position",
        "human_current_regime", "human_anticipated_regime",
        "predicted_current_regime", "predicted_anticipated_regime",
    ]].copy()
    result = result.rename(columns={
        "predicted_current_regime": "model_a_current",
        "predicted_anticipated_regime": "model_a_anticipated",
    })
    result["model_b_current"] = second.loc[disagrees, "predicted_current_regime"].to_numpy()
    result["model_b_anticipated"] = second.loc[
        disagrees, "predicted_anticipated_regime"
    ].to_numpy()
    return result


def session_lookup(predictions: pd.DataFrame, session: int | str) -> pd.DataFrame:
    """Select one session by stable index or ISO date for chart-oriented exploration."""

    if isinstance(session, int):
        selected = predictions.loc[predictions["session_index"] == session]
    else:
        selected = predictions.loc[predictions["session_date"].astype(str) == session]
    if selected.empty:
        raise ValueError("Requested session is absent from these predictions.")
    return selected.reset_index(drop=True)
