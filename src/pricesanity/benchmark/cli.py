"""Inspect and validate benchmark infrastructure without launching expensive searches."""

import argparse
from collections.abc import Sequence
import json
from pathlib import Path

from pricesanity.benchmark.protocol import load_benchmark_config, plan_benchmark
from pricesanity.benchmark.protocol import BenchmarkTrack
from pricesanity.benchmark.registry import list_model_families
from pricesanity.benchmark.preflight import build_scalability_preflight
from pricesanity.benchmark.search_spaces import load_search_spaces
from pricesanity.benchmark.snapshot import freeze_benchmark_snapshot, load_benchmark_snapshot
from pricesanity.benchmark.execution import BenchmarkExecutor
from pricesanity.benchmark.profile import (
    profile_synthetic_infrastructure,
    write_profile_report,
)
from pricesanity.config import load_config


def build_argument_parser() -> argparse.ArgumentParser:
    """Build benchmark inspection, initialization, and execution commands."""

    parser = argparse.ArgumentParser(
        description="Inspect and execute leakage-safe Price Sanity benchmark studies."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/benchmark/default.yaml"),
        help="Benchmark YAML configuration.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("hardware", help="Inspect Python, WSL, PyTorch build, and detected accelerators.")
    smoke_parser = subparsers.add_parser("device-smoke", help="Synthetic neural fit/save/reload and machine-specific timing.")
    smoke_parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    smoke_parser.add_argument("--compare", action="store_true")
    subparsers.add_parser("models", help="List registered model families and readiness.")
    profile_parser = subparsers.add_parser(
        "profile", help="Measure benchmark infrastructure without training study models."
    )
    profile_parser.add_argument("--synthetic", action="store_true", required=True)
    profile_parser.add_argument("--session-count", type=int, default=2690)
    profile_parser.add_argument("--candles-per-session", type=int, default=81)
    profile_parser.add_argument("--artifact-root", type=Path)
    profile_parser.add_argument("--output", type=Path)
    initialize_parser = subparsers.add_parser(
        "initialize", help="Freeze one immutable benchmark snapshot from current annotations."
    )
    initialize_parser.add_argument("--normalized", type=Path, required=True)
    initialize_parser.add_argument("--database", type=Path, required=True)
    initialize_parser.add_argument(
        "--candlesticks",
        type=Path,
        required=True,
        help="Paired clean OHLC Parquet proving strict session/reference eligibility.",
    )
    initialize_parser.add_argument("--project-config", type=Path, required=True)
    initialize_parser.add_argument("--study-directory", type=Path, required=True)
    freeze_parser = subparsers.add_parser("freeze-development", help="Seal winners across both declared tracks before final access.")
    freeze_parser.add_argument("--study-directory", type=Path, required=True)
    freeze_parser.add_argument("--search-spaces", type=Path, default=Path("configs/benchmark/search_spaces.yaml"))

    for command, help_text in (
        ("tune", "Run or resume chronological development tuning."),
        ("pilot", "Run one candidate on one development fold."),
        ("final", "Fit frozen winners and explicitly evaluate the final holdout."),
        ("learning-curve", "Run fixed-evaluation development learning curves."),
    ):
        action_parser = subparsers.add_parser(command, help=help_text)
        action_parser.add_argument("--study-directory", type=Path, required=True)
        model_selection = action_parser.add_mutually_exclusive_group(required=True)
        model_selection.add_argument("--model", choices=[
            family.name for family in list_model_families()
        ])
        model_selection.add_argument(
            "--all-models",
            action="store_true",
            help="Run the action for every registered family in registry order.",
        )
        action_parser.add_argument(
            "--track", choices=("controlled", "best_of_family"), required=True
        )
        action_parser.add_argument(
            "--device", choices=("cpu", "mps", "cuda"), default="cpu"
        )
        action_parser.add_argument(
            "--search-spaces", type=Path,
            default=Path("configs/benchmark/search_spaces.yaml"),
        )
        if command == "tune":
            action_parser.add_argument(
                "--fold", type=int,
                help="Optional one-based fold for a focused execution test.",
            )
            action_parser.add_argument("--acknowledge-scaling-risk", action="store_true")
        if command == "final":
            action_parser.add_argument(
                "--confirm-final-holdout", action="store_true",
                help="Explicitly unlock the frozen final holdout after development is complete.",
            )
    plan_parser = subparsers.add_parser(
        "plan", help="Validate and print chronological session boundaries."
    )
    plan_parser.add_argument(
        "--session-count",
        type=int,
        required=True,
        help="Number of complete annotated sessions expected at benchmark launch.",
    )
    run_parser = subparsers.add_parser(
        "run",
        help="Validate one benchmark action without launching model execution.",
    )
    model_selection = run_parser.add_mutually_exclusive_group(required=True)
    model_selection.add_argument("--model", choices=[
        family.name for family in list_model_families()
    ])
    model_selection.add_argument("--all-models", action="store_true")
    run_parser.add_argument(
        "--track", choices=("controlled", "best_of_family"), required=True
    )
    run_parser.add_argument(
        "--mode", choices=("tuning", "final", "learning_curve"), required=True
    )
    run_parser.add_argument("--session-count", type=int, required=True)
    run_parser.add_argument("--resume", action="store_true")
    run_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Required for this validation-only command; never launches training.",
    )
    run_parser.add_argument(
        "--search-spaces",
        type=Path,
        default=Path("configs/benchmark/search_spaces.yaml"),
    )
    return parser


def main(arguments: Sequence[str] | None = None) -> int:
    """Inspect, initialize, execute, or report one benchmark action."""

    parser = build_argument_parser()
    parsed = parser.parse_args(arguments)
    if parsed.command in {"hardware", "device-smoke"}:
        from pricesanity.benchmark.resources import hardware_diagnostic, smoke_devices
        report = hardware_diagnostic() if parsed.command == "hardware" else smoke_devices(device=parsed.device, compare=parsed.compare)
        print(json.dumps(report, indent=2, default=str))
        return 0
    config = load_benchmark_config(parsed.config)
    if parsed.command == "profile":
        report = profile_synthetic_infrastructure(
            session_count=parsed.session_count,
            candles_per_session=parsed.candles_per_session,
            context_length=config.window_length,
            artifact_root=parsed.artifact_root,
        )
        if parsed.output is not None:
            write_profile_report(report, parsed.output)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    if parsed.command == "initialize":
        snapshot = freeze_benchmark_snapshot(
            normalized_path=parsed.normalized,
            annotation_database_path=parsed.database,
            app_config=load_config(parsed.project_config),
            output_directory=parsed.study_directory / "snapshot",
            expected_session_count=config.expected_session_count,
            candlestick_path=parsed.candlesticks,
            development_session_count=config.development_session_count,
        )
        print(
            f"Frozen {snapshot.session_count} sessions and {snapshot.row_count} candles "
            f"as snapshot {snapshot.identity_sha256}."
        )
        return 0

    if parsed.command in {"tune", "pilot", "final", "learning-curve", "freeze-development"}:
        snapshot = load_benchmark_snapshot(parsed.study_directory / "snapshot")
        executor = BenchmarkExecutor(
            snapshot=snapshot,
            config=config,
            search_spaces=load_search_spaces(parsed.search_spaces),
            study_root=parsed.study_directory,
            device=getattr(parsed, "device", "cpu"),
        )
        if parsed.command == "freeze-development":
            print(json.dumps(executor.freeze_development(), indent=2, sort_keys=True))
            return 0
        track = BenchmarkTrack(parsed.track)
        model_names = (
            [family.name for family in list_model_families()]
            if parsed.all_models
            else [parsed.model]
        )
        if parsed.command == "tune":
            fold_indices = None if parsed.fold is None else (parsed.fold - 1,)
            for model_name in model_names:
                selected = executor.tune_model(
                    model_name,
                    track=track,
                    fold_indices=fold_indices,
                    acknowledge_scaling_risk=parsed.acknowledge_scaling_risk,
                )
                print(json.dumps(selected, indent=2, sort_keys=True))
        elif parsed.command == "pilot":
            for model_name in model_names:
                print(json.dumps(
                    executor.pilot(model_name, track=track), indent=2, default=str
                ))
        elif parsed.command == "final":
            for model_name in model_names:
                paths = executor.run_final(
                    model_name,
                    track=track,
                    confirm_final_holdout=parsed.confirm_final_holdout,
                )
                print("\n".join(str(path) for path in paths))
        else:
            for model_name in model_names:
                paths = executor.run_learning_curve(model_name, track=track)
                print("\n".join(str(path) for path in paths))
        return 0
    if parsed.command == "models":
        for family in list_model_families():
            readiness = "ready" if family.available else "planned"
            print(
                f"{family.name:24} {readiness:8} "
                f"{family.representation:12} {family.display_name}"
            )
        return 0

    if parsed.command == "run":
        if not parsed.dry_run:
            parser.error(
                "Full benchmark execution is intentionally disabled until annotation is "
                "complete; use --dry-run to validate the planned command."
            )
        try:
            plan_benchmark(parsed.session_count, config)
            load_search_spaces(parsed.search_spaces)
        except ValueError as error:
            parser.error(str(error))
        selected_models = (
            [family.name for family in list_model_families()]
            if parsed.all_models
            else [parsed.model]
        )
        print(f"Track: {parsed.track}")
        print(f"Mode: {parsed.mode}")
        print("Models: " + ", ".join(selected_models))
        print(f"Resume: {'enabled' if parsed.resume else 'disabled'}")
        for selected_model in selected_models:
            family = next(
                family for family in list_model_families()
                if family.name == selected_model
            )
            preflight = build_scalability_preflight(
                model_name=selected_model,
                training_samples=max(1, config.development_session_count * 66),
                evaluation_samples=max(1, config.final_holdout_session_count * 66),
                feature_dimension=config.window_length * len(config.feature_columns),
                candidate_count=max(
                    1, config.model_tuning_budgets.get(selected_model, 0)
                ),
                fold_count=len(config.chronological_validation_folds),
                seed_count=len(config.final_seeds) if family.stochastic else 1,
                polynomial_degree=(
                    2 if selected_model == "polynomial_logistic" else None
                ),
            )
            print(
                f"Preflight {selected_model}: "
                f"{preflight.training_samples:,} training samples; "
                f"{preflight.evaluation_samples:,} evaluation samples; "
                f"{preflight.transformed_feature_dimension:,} features; "
                f"approximately {preflight.approximate_job_count:,} jobs."
            )
            for warning in preflight.warnings:
                print(f"Warning ({selected_model}): {warning}")
        print("Dry run complete. No model was trained and no holdout was evaluated.")
        return 0

    try:
        plan = plan_benchmark(parsed.session_count, config)
    except ValueError as error:
        parser.error(str(error))
    print(
        "Development: "
        f"sessions {plan.development.start_index + 1}-"
        f"{plan.development.end_index}"
    )
    print(
        "Final holdout: "
        f"sessions {plan.final_holdout.start_index + 1}-"
        f"{plan.final_holdout.end_index} (locked during development)"
    )
    for fold_index, fold in enumerate(plan.chronological_validation_folds, start=1):
        print(
            f"Fold {fold_index}: train 1-{fold.training.end_index}; "
            f"validate {fold.validation.start_index + 1}-{fold.validation.end_index}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
