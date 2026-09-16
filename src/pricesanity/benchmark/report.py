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

    rows = []
    for metadata_path in sorted(Path(artifact_root).rglob("benchmark_metadata.json")):
        # A leaderboard needs persisted metrics, not hundreds of prediction tables. Detailed
        # predictions remain lazy until a person selects one run for inspection.
        metadata, metrics = load_benchmark_summary(metadata_path.parent)
        identity = metadata["identity"]
        efficiency = metrics.get("efficiency") or {}
        rows.append({
            "track": identity["track"],
            "model_name": identity["model_name"],
            "run_name": identity["run_name"],
            "seed": identity["seed"],
            "model_configuration_sha256": identity["model_configuration_sha256"],
            "representation_sha256": identity["representation_sha256"],
            "test_session_ids_sha256": identity["test_session_ids_sha256"],
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
            "uncertainty_kind": None,
            "run_directory": str(metadata_path.parent),
        })
    return pd.DataFrame(rows)


def collect_aggregated_runs(
    artifact_root: str | Path,
    *,
    expected_stochastic_seed_count: int = 3,
) -> pd.DataFrame:
    """Return one honest leaderboard row per frozen model configuration."""

    raw_runs = collect_completed_runs(artifact_root)
    if raw_runs.empty:
        return raw_runs
    raw_runs = raw_runs.loc[
        raw_runs["run_name"].astype(str).str.startswith("final_seed_")
    ].reset_index(drop=True)
    if raw_runs.empty:
        return raw_runs
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
    leaderboard = collect_aggregated_runs(parsed.artifact_root)
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
