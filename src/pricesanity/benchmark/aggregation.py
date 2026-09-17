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

    # Refuse partially shaped inputs before the empty-frame shortcut. An empty but malformed
    # result should not be mistaken for a valid study that simply produced no runs.
    required = {
        *AGGREGATION_KEYS,
        "seed",
        "current_macro_f1",
        "anticipated_macro_f1",
    }
    missing = required.difference(runs.columns)
    if missing:
        raise ValueError("Seed results are missing columns: " + ", ".join(sorted(missing)))

    # A correctly shaped empty input has no configurations to aggregate. Preserve that state
    # as an empty frame rather than manufacturing a leaderboard schema with implied results.
    if runs.empty:
        return pd.DataFrame()

    # Optional hashes become part of identity when present so results from different source
    # snapshots or protocols can never collapse into one apparently replicated experiment.
    optional_identity_keys = (
        "protocol_sha256",
        "annotation_snapshot_sha256",
        "normalized_dataset_sha256",
        "label_mapping_sha256",
    )
    identity_keys = (
        *AGGREGATION_KEYS,
        *(key for key in optional_identity_keys if key in runs.columns),
    )

    # A seed is one replicate of one frozen configuration. Duplicate seed rows would count the
    # same experiment twice and corrupt both the mean and its reported variability.
    if runs.duplicated([*identity_keys, "seed"]).any():
        raise ValueError("One frozen configuration cannot contain a duplicate seed.")

    # Deterministic families need one completed run, whereas stochastic families must supply
    # the protocol's full replicate set before they are eligible for comparison.
    stochastic_models = set(stochastic_models)
    rows: list[dict[str, object]] = []

    # Group by every available scientific identity so each output row describes exactly one
    # model, track, representation, configuration, and ordered evaluation population.
    for keys, group in runs.groupby(list(identity_keys), sort=True, dropna=False):
        model_name = str(group["model_name"].iloc[0])
        expected_count = (
            expected_stochastic_seed_count if model_name in stochastic_models else 1
        )

        # Newer artifacts declare their exact final seeds. Prefer that stronger contract over
        # the legacy count because it detects a substituted seed as well as a missing seed.
        if "declared_final_seeds" in group:
            declared = group["declared_final_seeds"].dropna()
            if not declared.empty:
                sets = {
                    tuple(sorted(int(seed) for seed in seeds))
                    for seeds in declared
                }
                if len(sets) != 1 or len(declared) != len(group):
                    raise ValueError("Runs disagree about the declared final seed set.")

                expected_seeds = next(iter(sets))
                if tuple(sorted(int(seed) for seed in group["seed"])) != expected_seeds:
                    raise ValueError(
                        "Completed results do not match the declared final seed set."
                    )
                expected_count = len(expected_seeds)

        if len(group) != expected_count:
            # Partial stochastic seed sets stay off the leaderboard: reporting their mean
            # would let an interrupted or selectively completed configuration look stronger.
            raise ValueError(
                f"{model_name} requires {expected_count} frozen seed result(s), "
                f"but {len(group)} were provided."
            )

        # Keep the two prediction heads separate through aggregation. Averaging them first
        # would conceal a model that is stable on one label and unstable on the other.
        current = group["current_macro_f1"].to_numpy(dtype=float)
        anticipated = group["anticipated_macro_f1"].to_numpy(dtype=float)

        # Begin with the canonical group identity, then attach statistics and provenance. This
        # makes every leaderboard row independently traceable to its frozen evaluation inputs.
        row = dict(zip(identity_keys, keys, strict=True))
        row.update(
            {
                "seed_count": len(group),
                "seeds": tuple(int(value) for value in group["seed"]),
                "current_macro_f1_mean": float(current.mean()),
                "current_macro_f1_std": float(current.std(ddof=0)),
                "anticipated_macro_f1_mean": float(anticipated.mean()),
                "anticipated_macro_f1_std": float(anticipated.std(ddof=0)),
                "mean_head_macro_f1": float(
                    ((current + anticipated) / 2.0).mean()
                ),
                "individual_current_macro_f1": tuple(current.tolist()),
                "individual_anticipated_macro_f1": tuple(anticipated.tolist()),
            }
        )

        # Timing can be pooled only when every replicate reports the same canonical hardware.
        # Missing fingerprints therefore make timing unavailable rather than implicitly equal.
        hardware_values = (
            group["hardware_fingerprint"]
            .dropna()
            .map(_canonical_hardware)
            .tolist()
            if "hardware_fingerprint" in group.columns
            else []
        )
        hardware_compatible = (
            bool(hardware_values) and len(set(hardware_values)) == 1
        )
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

        # These fields describe the shared configuration rather than a per-seed measurement.
        # Propagate one nonmissing value for display without inventing values for old artifacts.
        for column in (
            "model_configuration",
            "parameter_count",
            "serialized_model_bytes",
            "device",
            "uncertainty_kind",
        ):
            if column in group.columns:
                values = group[column].dropna()
                row[column] = values.iloc[0] if not values.empty else None

        # Emit only after seed completeness and hardware treatment have both been resolved, so
        # no partially validated row can leak into the official leaderboard.
        rows.append(row)

    # Stable track grouping and descending score order make repeated report generation
    # deterministic without changing the scientific identity of any aggregate.
    return pd.DataFrame(rows).sort_values(
        ["track", "mean_head_macro_f1"], ascending=[True, False]
    ).reset_index(drop=True)


def add_baseline_deltas(leaderboard: pd.DataFrame) -> pd.DataFrame:
    """Add transparent score differences without changing leaderboard ordering."""

    # Empty reports remain empty, including their existing columns, so read-only callers can
    # pass through a not-yet-run study without special-case schema construction.
    if leaderboard.empty:
        return leaderboard.copy()

    # Delta calculations require both the score and the evaluation-population identity. A
    # model score must never be compared with a baseline evaluated on different candles.
    required = {"track", "test_session_ids_sha256", "model_name", "mean_head_macro_f1"}
    if not required.issubset(leaderboard.columns):
        raise ValueError("Baseline deltas require track, population, model, and score fields.")

    # Work on a copy because reporting annotations must not mutate the canonical aggregates
    # supplied by another view or notebook.
    result = leaderboard.copy()
    result["delta_vs_majority"] = np.nan
    result["delta_vs_persistence"] = np.nan
    optional_population_keys = (
        "protocol_sha256",
        "annotation_snapshot_sha256",
        "normalized_dataset_sha256",
        "label_mapping_sha256",
    )
    population_keys = ["track", "test_session_ids_sha256"] + [
        key for key in optional_population_keys if key in result.columns
    ]

    # Deltas are meaningful only against baselines that saw the exact same ordered population
    # under the same frozen scientific identities.
    for _, group in result.groupby(population_keys, dropna=False):
        majority = group.loc[
            group["model_name"] == "majority_class", "mean_head_macro_f1"
        ]
        persistence = group.loc[
            group["model_name"] == "previous_regime", "mean_head_macro_f1"
        ]

        # Exactly one matching baseline is required. Missing or duplicate baselines leave N/A
        # rather than silently choosing an arbitrary reference score.
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
    """Normalize stored hardware evidence before deciding whether timings can be averaged."""

    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            # Older artifacts may contain an opaque string. Preserve it as a comparable token
            # rather than rejecting otherwise valid predictive results.
            return value

    # Canonical key ordering makes equivalent mapping objects compare equal regardless of the
    # order in which their hardware fields were originally serialized.
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def partition_seed_results(
    runs: pd.DataFrame,
    *,
    stochastic_models: Sequence[str],
    expected_stochastic_seed_count: int,
):
    """Keep incomplete configurations visible without assigning partial-seed scores."""

    # Match the strict aggregator's identity partition so the GUI can explain incomplete groups
    # without combining experiments that the leaderboard itself would keep separate.
    optional_identity_keys = (
        "protocol_sha256",
        "annotation_snapshot_sha256",
        "normalized_dataset_sha256",
        "label_mapping_sha256",
    )
    keys = list(AGGREGATION_KEYS) + [
        key for key in optional_identity_keys if key in runs.columns
    ]
    complete = []
    incomplete: list[dict[str, object]] = []

    # Reuse the strict aggregator per identity so this reporting helper cannot silently
    # weaken the completeness rules used by the official leaderboard.
    for _, group in runs.groupby(keys, sort=True, dropna=False):
        try:
            complete.append(
                aggregate_seed_results(
                    group,
                    stochastic_models=stochastic_models,
                    expected_stochastic_seed_count=expected_stochastic_seed_count,
                )
            )
        except ValueError as error:
            # Preserve enough identity and the exact validation reason for read-only reporting;
            # an incomplete group receives no aggregate score.
            first = group.iloc[0]
            incomplete.append(
                {
                    "track": first["track"],
                    "model_name": first["model_name"],
                    "model_configuration": first.get("model_configuration"),
                    "seed_count": len(group),
                    "reason": str(error),
                }
            )

    aggregated = (
        pd.concat(complete, ignore_index=True) if complete else pd.DataFrame()
    )
    return aggregated, incomplete
