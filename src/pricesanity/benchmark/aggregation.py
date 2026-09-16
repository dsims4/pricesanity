"""Aggregate frozen configurations across seeds without rewarding lucky runs."""

from collections.abc import Sequence
import json

import numpy as np
import pandas as pd


AGGREGATION_KEYS = (
    "track",
    "model_name",
    "model_configuration_sha256",
    "representation_sha256",
    "test_session_ids_sha256",
)


def aggregate_seed_results(
    runs: pd.DataFrame,
    *,
    stochastic_models: Sequence[str],
    expected_stochastic_seed_count: int,
) -> pd.DataFrame:
    """Return one leaderboard row per configuration and evaluation population."""

    required = {
        *AGGREGATION_KEYS,
        "seed",
        "current_macro_f1",
        "anticipated_macro_f1",
    }
    missing = required.difference(runs.columns)
    if missing:
        raise ValueError("Seed results are missing columns: " + ", ".join(sorted(missing)))
    if runs.empty:
        return pd.DataFrame()
    if runs.duplicated([*AGGREGATION_KEYS, "seed"]).any():
        raise ValueError("One frozen configuration cannot contain a duplicate seed.")

    stochastic_models = set(stochastic_models)
    rows: list[dict[str, object]] = []
    for keys, group in runs.groupby(list(AGGREGATION_KEYS), sort=True, dropna=False):
        model_name = str(group["model_name"].iloc[0])
        expected_count = (
            expected_stochastic_seed_count if model_name in stochastic_models else 1
        )
        if len(group) != expected_count:
            raise ValueError(
                f"{model_name} requires {expected_count} frozen seed result(s), "
                f"but {len(group)} were provided."
            )
        current = group["current_macro_f1"].to_numpy(dtype=float)
        anticipated = group["anticipated_macro_f1"].to_numpy(dtype=float)
        row = dict(zip(AGGREGATION_KEYS, keys, strict=True))
        row.update({
            "seed_count": len(group),
            "seeds": tuple(int(value) for value in group["seed"]),
            "current_macro_f1_mean": float(current.mean()),
            "current_macro_f1_std": float(current.std(ddof=0)),
            "anticipated_macro_f1_mean": float(anticipated.mean()),
            "anticipated_macro_f1_std": float(anticipated.std(ddof=0)),
            "mean_head_macro_f1": float(((current + anticipated) / 2.0).mean()),
            "individual_current_macro_f1": tuple(current.tolist()),
            "individual_anticipated_macro_f1": tuple(anticipated.tolist()),
        })
        hardware_values = (
            group["hardware_fingerprint"].dropna().map(_canonical_hardware).tolist()
            if "hardware_fingerprint" in group.columns else []
        )
        hardware_compatible = bool(hardware_values) and len(set(hardware_values)) == 1
        row["hardware_compatible"] = hardware_compatible
        row["hardware_fingerprint"] = (
            group["hardware_fingerprint"].dropna().iloc[0]
            if hardware_compatible else None
        )
        # Predictive metrics remain comparable across machines. Runtime means do not: averaging
        # an Intel CPU fit with an Apple MPS fit would manufacture a number no machine achieved.
        for source, destination in (
            ("training_seconds", "training_seconds_mean"),
            ("inference_seconds", "inference_seconds_mean"),
            ("inference_samples_per_second", "inference_samples_per_second_mean"),
        ):
            row[destination] = (
                float(group[source].astype(float).mean())
                if hardware_compatible and source in group.columns
                and group[source].notna().all()
                else None
            )
        for column in (
            "model_configuration", "parameter_count", "serialized_model_bytes",
            "device", "uncertainty_kind",
        ):
            if column in group.columns:
                values = group[column].dropna()
                row[column] = values.iloc[0] if not values.empty else None
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["track", "mean_head_macro_f1"], ascending=[True, False]
    ).reset_index(drop=True)


def add_baseline_deltas(leaderboard: pd.DataFrame) -> pd.DataFrame:
    """Add transparent score differences without changing leaderboard ordering."""

    if leaderboard.empty:
        return leaderboard.copy()
    required = {"track", "test_session_ids_sha256", "model_name", "mean_head_macro_f1"}
    if not required.issubset(leaderboard.columns):
        raise ValueError("Baseline deltas require track, population, model, and score fields.")
    result = leaderboard.copy()
    result["delta_vs_majority"] = np.nan
    result["delta_vs_persistence"] = np.nan
    for _, group in result.groupby(["track", "test_session_ids_sha256"], dropna=False):
        majority = group.loc[
            group["model_name"] == "majority_class", "mean_head_macro_f1"
        ]
        persistence = group.loc[
            group["model_name"] == "previous_regime", "mean_head_macro_f1"
        ]
        if len(majority) == 1:
            result.loc[group.index, "delta_vs_majority"] = (
                group["mean_head_macro_f1"] - float(majority.iloc[0])
            )
        if len(persistence) == 1:
            result.loc[group.index, "delta_vs_persistence"] = (
                group["mean_head_macro_f1"] - float(persistence.iloc[0])
            )
    return result


def _canonical_hardware(value: object) -> str:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return value
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
