"""Summarize completed benchmark artifacts without training or inference."""

import argparse
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from pricesanity.benchmark.aggregation import add_baseline_deltas, aggregate_seed_results
from pricesanity.benchmark.artifacts import load_benchmark_summary
from pricesanity.benchmark.registry import list_model_families


def collect_completed_runs(artifact_root: str | Path) -> pd.DataFrame:
    """Load every verified complete run into one leaderboard-shaped table."""

    # Gather small summaries first; model loading and inference have no role in reporting.
    rows = []

    # Discover only final metadata commit markers. Partial run directories intentionally remain
    # invisible because their component files have not yet become a completed experiment.
    for metadata_path in sorted(Path(artifact_root).rglob("benchmark_metadata.json")):
        # A leaderboard needs persisted metrics, not hundreds of prediction tables. Detailed
        # predictions remain lazy until a person selects one run for inspection.
        metadata, metrics = load_benchmark_summary(metadata_path.parent)
        identity = metadata["identity"]
        efficiency = metrics.get("efficiency") or {}

        # Flatten identity, predictive scores, and optional efficiency evidence into one row so
        # later aggregation never needs to reopen model or prediction artifacts.
        rows.append({
            "track": identity["track"],
            "model_name": identity["model_name"],
            "run_name": identity["run_name"],
            "seed": identity["seed"],
            "model_configuration_sha256": identity["model_configuration_sha256"],
            "representation_sha256": identity["representation_sha256"],
            "test_session_ids_sha256": identity["test_session_ids_sha256"],
            **{
                key: identity.get(key)
                for key in (
                    "protocol_sha256",
                    "annotation_snapshot_sha256",
                    "normalized_dataset_sha256",
                    "label_mapping_sha256",
                )
            },
            "declared_final_seeds": metadata.get("dataset", {}).get("declared_final_seeds"),
            "current_macro_f1": metrics["current"]["macro_f1"],
            "anticipated_macro_f1": metrics["anticipated"]["macro_f1"],
            "mean_head_macro_f1": metrics["mean_head_macro_f1"],
            "model_configuration": metadata.get("model_configuration"),
            "training_seconds": efficiency.get("training_seconds"),
            "inference_seconds": efficiency.get("inference_seconds"),
            "inference_samples_per_second": efficiency.get(
                "inference_samples_per_second"
            ),
            "serialized_model_bytes": efficiency.get("serialized_model_bytes"),
            "parameter_count": efficiency.get("parameter_count"),
            "device": efficiency.get("device"),
            "hardware_fingerprint": efficiency.get("hardware_fingerprint"),
            # Older summary artifacts did not persist one run-level uncertainty kind. Preserve a
            # visible N/A rather than inferring probability semantics from the model family.
            "uncertainty_kind": None,
            "run_directory": str(metadata_path.parent),
        })

    # Sorted discovery order makes this raw report deterministic even before aggregation applies
    # its score ordering.
    return pd.DataFrame(rows)


def collect_aggregated_runs(
    artifact_root: str | Path,
    *,
    expected_stochastic_seed_count: int = 3,
) -> pd.DataFrame:
    """Return one honest leaderboard row per frozen model configuration."""

    raw_runs = collect_completed_runs(artifact_root)

    # A study in progress is a legitimate report state. Keep its empty frame untouched instead of
    # fabricating baseline columns that imply final evaluation has occurred.
    if raw_runs.empty:
        return raw_runs

    # Candidate and learning-curve artifacts are development evidence, not final leaderboard
    # replicates. Restrict aggregation to explicitly named final seed runs.
    raw_runs = raw_runs.loc[
        raw_runs["run_name"].astype(str).str.startswith("final_seed_")
    ].reset_index(drop=True)
    if raw_runs.empty:
        return raw_runs

    # Registry metadata determines which families require replicated seeds; report code must not
    # duplicate or drift from the model family's deterministic/stochastic declaration.
    stochastic = [family.name for family in list_model_families() if family.stochastic]
    return add_baseline_deltas(
        aggregate_seed_results(
            raw_runs,
            stochastic_models=stochastic,
            expected_stochastic_seed_count=expected_stochastic_seed_count,
        )
    )


def main(arguments: Sequence[str] | None = None) -> int:
    """Print verified completed runs or an honest empty-state message."""

    parser = argparse.ArgumentParser(
        description="Report completed Price Sanity benchmark artifacts without retraining."
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=Path("data/models/benchmark"),
    )
    parsed = parser.parse_args(arguments)

    # The CLI delegates grouping and completeness to the same helper used by notebooks.
    leaderboard = collect_aggregated_runs(parsed.artifact_root)

    # Make an unfinished corpus explicit in terminal output rather than printing an ambiguous
    # blank table that could be mistaken for a filtering error.
    if leaderboard.empty:
        print("No completed benchmark runs. Infrastructure is awaiting the full corpus.")
        return 0
    print(
        leaderboard.sort_values(
            ["track", "mean_head_macro_f1"], ascending=[True, False]
        ).to_string(index=False)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
