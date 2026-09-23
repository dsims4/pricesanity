"""Inspect, freeze, and execute chronological benchmark studies at the available corpus size."""

import argparse
from dataclasses import replace
from collections.abc import Sequence
import json
import sqlite3
from pathlib import Path

from pricesanity.benchmark.protocol import (
    load_benchmark_config, plan_benchmark, resolve_benchmark_config, describe_protocol,
)
from pricesanity.benchmark.protocol import BenchmarkTrack
from pricesanity.benchmark.registry import list_model_families
from pricesanity.benchmark.preflight import build_scalability_preflight
from pricesanity.benchmark.search_spaces import load_search_spaces
from pricesanity.benchmark.snapshot import load_benchmark_snapshot
from pricesanity.benchmark.population import discover_population, freeze_population
from pricesanity.benchmark.artifacts import canonical_sha256
from pricesanity.benchmark.orchestration import execute_suite
from pricesanity.benchmark.execution import BenchmarkExecutor, validate_training_support
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

    # Environment and infrastructure checks are usable without private data or a study. Keep
    # their arguments separate from commands that publish or unlock scientific artifacts.
    subparsers.add_parser(
        "hardware",
        help="Inspect Python, WSL, PyTorch build, and detected accelerators.",
    )
    smoke_parser = subparsers.add_parser(
        "device-smoke",
        help="Synthetic neural fit/save/reload and machine-specific timing.",
    )
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
    publish_parser = subparsers.add_parser(
        "publish-results", help="Export one deterministic, allowlisted public result bundle."
    )
    publish_parser.add_argument("--run", type=Path, required=True)
    publish_parser.add_argument("--output", type=Path, required=True)

    diagnostic = subparsers.add_parser(
        "data-sufficiency", help="Development-only fixed-population data-value diagnostic.",
    )
    for source in ("normalized", "candlesticks", "database", "project-config"):
        diagnostic.add_argument("--" + source, type=Path, required=True)
    diagnostic.add_argument("--output-directory", type=Path, required=True)
    selection = diagnostic.add_mutually_exclusive_group(required=True)
    selection.add_argument("--model", nargs="+", choices=[
        family.name for family in list_model_families()
    ])
    selection.add_argument("--all-models", action="store_true")
    diagnostic.add_argument("--track", choices=("controlled",), default="controlled")
    diagnostic.add_argument(
        "--train-sizes", nargs="+", type=int, default=[100, 200, 300, 400, 500],
    )
    diagnostic.add_argument("--seeds", nargs="+", type=int)
    diagnostic.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    diagnostic.add_argument(
        "--fixed-config", type=Path,
        help="JSON mapping each selected model name to fixed conceptual parameters; never tuned.",
    )
    diagnostic.add_argument(
        "--benchmark-snapshot", type=Path,
        help="Optional official snapshot directory proving the development/final boundary.",
    )
    diagnostic.add_argument("--allow-small-evaluation", action="store_true")
    diagnostic.add_argument("--meaningful-gain", type=float, default=0.01)
    diagnostic.add_argument("--small-gain", type=float, default=0.005)
    diagnostic.add_argument("--bootstrap-repetitions", type=int, default=2000)
    diagnostic.add_argument("--resume", action="store_true")

    # Snapshot initialization and global freezing each have their own evidence requirements;
    # neither accepts the model-specific execution options below.
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
    initialize_parser.add_argument("--session-count", type=int)
    freeze_parser = subparsers.add_parser(
        "freeze-development",
        help="Seal winners across both declared tracks before final access.",
    )
    freeze_parser.add_argument("--study-directory", type=Path, required=True)
    freeze_parser.add_argument(
        "--search-spaces",
        type=Path,
        default=Path("configs/benchmark/search_spaces.yaml"),
    )

    # Share model, track, and device vocabulary across execution stages while limiting final
    # confirmation and scaling acknowledgement to the stages that consume those decisions.
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
    # Planning surfaces validate configuration without invoking the executor or fitting models.
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
        help="Freeze the available corpus, tune both tracks, and evaluate the complete selected suite.",
    )
    model_selection = run_parser.add_mutually_exclusive_group()
    model_selection.add_argument("--model", choices=[
        family.name for family in list_model_families()
    ])
    model_selection.add_argument("--all-models", action="store_true")
    run_parser.add_argument(
        "--track", choices=("controlled", "best_of_family"),
        help="Optional single-track study; default runs both tracks."
    )
    run_parser.add_argument("--session-count", type=int, help="Use the first N complete eligible sessions; default uses all.")
    for source in ("normalized", "candlesticks", "database", "project-config"):
        run_parser.add_argument("--" + source, type=Path)
    run_parser.add_argument("--artifact-root", type=Path)
    run_parser.add_argument("--study-directory", type=Path, help="Resume this exact frozen population without rediscovery.")
    run_parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    run_parser.add_argument("--acknowledge-scaling-risk", action="store_true")
    run_parser.add_argument("--resume", action="store_true")
    run_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the derived population and preflights without writing artifacts or training.",
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

    # Hardware inspection must remain usable before a snapshot exists, so its optional
    # PyTorch imports and return path precede benchmark configuration loading.
    if parsed.command in {"hardware", "device-smoke"}:
        from pricesanity.benchmark.resources import hardware_diagnostic, smoke_devices

        report = (
            hardware_diagnostic()
            if parsed.command == "hardware"
            else smoke_devices(device=parsed.device, compare=parsed.compare)
        )
        print(json.dumps(report, indent=2, default=str))
        return 0

    # WHY: exporting verified summaries needs neither private data nor benchmark configuration.
    if parsed.command == "publish-results":
        from pricesanity.benchmark.publication import publish_benchmark_results

        try:
            output = publish_benchmark_results(parsed.run, parsed.output)
        except (ValueError, OSError) as error:
            parser.error(str(error))
        print(f"Published deterministic public benchmark result: {output}")
        return 0

    config = load_benchmark_config(parsed.config)
    if parsed.command == "data-sufficiency":
        from pricesanity.benchmark.data_sufficiency import run_data_sufficiency

        try:
            run_data_sufficiency(
                normalized_path=parsed.normalized, candlestick_path=parsed.candlesticks,
                database_path=parsed.database, app_config=load_config(parsed.project_config),
                benchmark_config=config, output_directory=parsed.output_directory,
                model_names=tuple(
                    family.name for family in list_model_families()
                ) if parsed.all_models else tuple(parsed.model),
                train_sizes=tuple(parsed.train_sizes),
                seeds=None if parsed.seeds is None else tuple(parsed.seeds),
                device=parsed.device, fixed_config=parsed.fixed_config,
                benchmark_snapshot=parsed.benchmark_snapshot,
                allow_small_evaluation=parsed.allow_small_evaluation,
                meaningful_gain=parsed.meaningful_gain, small_gain=parsed.small_gain,
                bootstrap_repetitions=parsed.bootstrap_repetitions, resume=parsed.resume,
            )
        except (ValueError, OSError, RuntimeError, sqlite3.Error) as error:
            parser.error(str(error))
        return 0
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
        # Official initialization reads mutable annotations; every later official
        # study action consumes the resulting immutable snapshot.
        population = discover_population(
            normalized_path=parsed.normalized, candlestick_path=parsed.candlesticks,
            database_path=parsed.database, app_config=load_config(parsed.project_config),
            config=config, session_count=parsed.session_count,
        )
        snapshot = freeze_population(
            population, parsed.study_directory,
            timestamp_column=load_config(parsed.project_config).data.timestamp_column,
        )
        print(json.dumps(describe_protocol(population.config), indent=2))
        print(
            f"Frozen {snapshot.session_count} sessions and {snapshot.row_count} candles "
            f"as snapshot {snapshot.identity_sha256}."
        )
        return 0

    if parsed.command in {"tune", "pilot", "final", "learning-curve", "freeze-development"}:
        # One executor applies the same source, snapshot, resume, and holdout gates to every
        # command capable of creating or mutating benchmark artifacts.
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
                # The executor checks this explicit confirmation against the persisted global
                # development seal before it opens holdout bytes.
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
        selected_models = ([parsed.model] if parsed.model else [family.name for family in list_model_families()])
        if parsed.model:
            config = replace(config, model_tuning_budgets={parsed.model: config.model_tuning_budgets[parsed.model]})
        tracks = (parsed.track,) if parsed.track else tuple(track.value for track in BenchmarkTrack)
        try:
            spaces = load_search_spaces(parsed.search_spaces)
            if parsed.study_directory is not None:
                if (not parsed.resume or parsed.session_count is not None
                        or parsed.artifact_root is not None or any(getattr(parsed, key) is not None
                            for key in ("normalized", "candlesticks", "database", "project_config"))):
                    raise ValueError("--study-directory requires --resume and no live sources, artifact root, or new cap.")
                snapshot = load_benchmark_snapshot(parsed.study_directory / "snapshot")
                config = resolve_benchmark_config(snapshot.session_count, config)
                study_directory = parsed.study_directory
            elif all(getattr(parsed, key) is not None for key in (
                "normalized", "candlesticks", "database", "project_config"
            )):
                app_config = load_config(parsed.project_config)
                population = discover_population(
                    normalized_path=parsed.normalized, candlestick_path=parsed.candlesticks,
                    database_path=parsed.database, app_config=app_config,
                    config=config, session_count=parsed.session_count,
                )
                config = population.config
                validate_training_support(config, population.sessions, spaces, tracks)
                fingerprint = canonical_sha256({"population": population.identity_sha256,
                                                "search_spaces": spaces, "tracks": tracks})
                study_directory = (parsed.artifact_root or config.output_root) / (
                    f"generalized_{config.expected_session_count}_{fingerprint[:16]}"
                )
                if not parsed.dry_run:
                    snapshot = freeze_population(population, study_directory,
                        timestamp_column=app_config.data.timestamp_column, resume=parsed.resume)
            elif parsed.dry_run and parsed.session_count is not None and not any(
                getattr(parsed, key) is not None for key in ("normalized", "candlesticks", "database", "project_config")
            ):
                config = resolve_benchmark_config(parsed.session_count, config)
                study_directory = None
            else:
                raise ValueError("run requires --normalized, --candlesticks, --database, and --project-config.")
            print(json.dumps(describe_protocol(config), indent=2))
            print("Models: " + ", ".join(selected_models))
            print("Tracks: " + ", ".join(tracks))
            if study_directory is not None:
                print(f"Study: {study_directory}")
            if not parsed.dry_run:
                executor = BenchmarkExecutor(snapshot=snapshot, config=config, search_spaces=spaces,
                    study_root=study_directory, device=parsed.device, declared_tracks=tracks)
                paths = execute_suite(executor, acknowledge_scaling_risk=parsed.acknowledge_scaling_risk)
                print(f"Completed {len(paths)} final runs. Report with pricesanity-benchmark-report --artifact-root {study_directory}")
                return 0
        except (ValueError, OSError, RuntimeError, sqlite3.Error) as error:
            parser.error(str(error))
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
        config = resolve_benchmark_config(parsed.session_count, config)
        plan = plan_benchmark(parsed.session_count, config)
    except ValueError as error:
        parser.error(str(error))
    print(json.dumps(describe_protocol(config), indent=2))
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
