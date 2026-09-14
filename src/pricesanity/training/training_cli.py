"""Run single or walk-forward Price Sanity training experiments."""

import argparse
from collections.abc import Callable, Sequence
from contextlib import closing
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import hashlib
from pathlib import Path
from typing import Any
from time import perf_counter

import pandas as pd
import torch

from pricesanity.annotation.store import AnnotationStore
from pricesanity.config import AppConfig, load_config
from pricesanity.training.artifacts import file_sha256, load_run_artifacts
from pricesanity.training.dataset import (
    FEATURE_COLUMNS,
    REGIME_TO_CLASS,
    AnnotatedSessionSplit,
    TensorSession,
    build_complete_annotated_sessions,
    create_training_data_loaders,
)
from pricesanity.training.model import RegimeTransformer, TransformerConfig
from pricesanity.training.trainer import (
    EpochRecord,
    TrainingLoopConfig,
    TrainingResult,
    fit_feature_standardizer,
    fit_majority_classes,
    resolve_training_device,
    save_test_predictions,
    save_training_checkpoint,
    train_regime_transformer,
)
from pricesanity.training.walk_forward import (
    WalkForwardPlan,
    WalkForwardRun,
    plan_walk_forward_runs,
)


@dataclass(frozen=True)
class RunOutputPaths:
    """Generated artifacts that together describe one completed experiment."""

    directory: Path
    checkpoint: Path
    predictions: Path
    metadata: Path


def build_argument_parser() -> argparse.ArgumentParser:
    """Build terminal options for single and chronological walk-forward training."""

    argument_parser = argparse.ArgumentParser(
        description="Train the causal Price Sanity regime Transformer."
    )
    argument_parser.add_argument(
        "--normalized",
        required=True,
        type=Path,
        help="Path to prepared normalized Parquet data.",
    )
    argument_parser.add_argument(
        "--config",
        required=True,
        type=Path,
        help="Path to the YAML project configuration.",
    )
    argument_parser.add_argument(
        "--database",
        type=Path,
        default=Path("data/annotations/pricesanity.sqlite3"),
        help="Path to the SQLite annotation database.",
    )
    argument_parser.add_argument(
        "--candlesticks",
        type=Path,
        help="Optional OHLC Parquet path recorded for the test-results GUI.",
    )
    argument_parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("data/models/run_001/model.pt"),
        help="Single-run checkpoint path (default: data/models/run_001/model.pt).",
    )
    argument_parser.add_argument(
        "--walk-forward",
        action="store_true",
        help="Train every complete expanding-history experiment in a longer annotated corpus.",
    )
    argument_parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("data/models/walk_forward"),
        help="Parent directory for walk-forward run folders.",
    )
    run_selection = argument_parser.add_mutually_exclusive_group()
    run_selection.add_argument(
        "--run-index", type=int, help="Train only this one-based walk-forward run."
    )
    run_selection.add_argument(
        "--start-run", type=int, help="Start at this one-based walk-forward run."
    )
    argument_parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip verified completed runs; rebuild interrupted runs from their first epoch.",
    )
    argument_parser.add_argument(
        "--training-sessions",
        type=int,
        default=100,
        help=(
            "Initial training session count; later walk-forward histories expand from session one."
        ),
    )
    argument_parser.add_argument(
        "--validation-sessions",
        type=int,
        default=20,
        help="Number of following sessions used only for model selection.",
    )
    argument_parser.add_argument(
        "--test-sessions",
        type=int,
        default=10,
        help="Number of final sessions untouched until official evaluation.",
    )
    argument_parser.add_argument(
        "--model-dimension",
        type=int,
        default=48,
        help="Hidden representation width, divisible by three attention heads (default: 48).",
    )
    argument_parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Maximum sessions per batch (default: 8).",
    )
    argument_parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Maximum training epochs (default: 50).",
    )
    argument_parser.add_argument(
        "--learning-rate",
        type=float,
        default=3e-4,
        help="AdamW learning rate (default: 0.0003).",
    )
    argument_parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-2,
        help="AdamW weight decay (default: 0.01).",
    )
    argument_parser.add_argument(
        "--gradient-clip",
        type=float,
        default=1.0,
        help="Maximum gradient norm (default: 1.0).",
    )
    argument_parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Epochs without validation improvement before stopping.",
    )
    argument_parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="auto",
        help="Training device (default: auto).",
    )
    argument_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Explicitly rebuild selected runs, replacing their historical artifacts.",
    )
    return argument_parser


def _print_epoch(epoch_record: EpochRecord) -> None:
    """Print compact training and validation evidence for one epoch."""

    print(
        f"  Epoch {epoch_record.epoch:03d} | "
        f"train loss {epoch_record.training.total_loss:.4f} | "
        f"validation loss {epoch_record.validation.total_loss:.4f} | "
        f"current F1 {epoch_record.validation.current.macro_f1:.3f} | "
        f"anticipated F1 {epoch_record.validation.anticipated.macro_f1:.3f}"
    )


def _run_output_paths(
    run: WalkForwardRun,
    *,
    walk_forward: bool,
    checkpoint_path: Path,
    output_directory: Path,
) -> RunOutputPaths:
    """Give every run an isolated directory with predictable artifact names."""

    run_directory = (
        output_directory / f"run_{run.run_index:03d}"
        if walk_forward
        else checkpoint_path.parent
    )
    return RunOutputPaths(
        directory=run_directory,
        checkpoint=(run_directory / "model.pt" if walk_forward else checkpoint_path),
        predictions=run_directory / "test_predictions.parquet",
        metadata=run_directory / "run_metadata.json",
    )


def _infer_candlestick_path(normalized_path: Path) -> Path | None:
    """Find the matching conventional interim artifact when it exists."""

    if normalized_path.parent.name != "processed":
        return None
    candidate_path = (
        normalized_path.parent.parent
        / "interim"
        / f"{normalized_path.stem}_ohlc.parquet"
    )
    return candidate_path if candidate_path.is_file() else None


def _experiment_metadata(
    run: WalkForwardRun,
    plan: WalkForwardPlan,
    result: TrainingResult,
    *,
    app_config: AppConfig,
    normalized_path: Path,
    candlestick_path: Path | None,
    database_path: Path,
    output_paths: RunOutputPaths,
) -> dict[str, Any]:
    """Record temporal roles and artifact identities without opaque DataFrame slices."""

    def serialize_boundary(boundary: Any) -> dict[str, Any]:
        boundary_data = asdict(boundary)
        boundary_data["start_date"] = boundary.start_date.isoformat()
        boundary_data["end_date"] = boundary.end_date.isoformat()
        return boundary_data

    return {
        "format_version": 2,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_index": run.run_index,
        "run_count": len(plan.runs),
        "training": serialize_boundary(run.training),
        "validation": serialize_boundary(run.validation),
        "test": serialize_boundary(run.test),
        "protocol": "expanding_history_v1",
        "initialization_mode": "fresh",
        "initial_training_session_count": plan.initial_training_session_count,
        "training_session_count": run.training.end_index - run.training.start_index,
        "validation_session_count": plan.validation_session_count,
        "test_session_count": plan.test_session_count,
        "training_growth_session_count": plan.training_growth_session_count,
        "trailing_session_count": plan.trailing_session_count,
        "random_seed": app_config.project.random_seed,
        "instrument": app_config.data.instrument,
        "target_interval": app_config.data.target_interval,
        "session_timezone": app_config.data.session_timezone,
        "normalized_path": str(normalized_path.resolve()),
        "candlestick_path": (
            str(candlestick_path.resolve()) if candlestick_path is not None else None
        ),
        "annotation_database_path": str(database_path.resolve()),
        "checkpoint_path": output_paths.checkpoint.name,
        "prediction_path": output_paths.predictions.name,
        "model_state_sha256": result.model_state_sha256,
        "best_epoch": result.best_epoch,
    }


def _save_json_atomically(
    output_path: Path,
    content: dict[str, Any],
    *,
    overwrite: bool,
) -> None:
    """Publish run metadata only after its complete JSON can be written."""

    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Run metadata already exists: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(".partial.json")
    try:
        with temporary_path.open("w", encoding="utf-8") as metadata_file:
            json.dump(content, metadata_file, indent=2, sort_keys=True)
            metadata_file.write("\n")
        temporary_path.replace(output_path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _train_one_run(
    run: WalkForwardRun,
    plan: WalkForwardPlan,
    split: AnnotatedSessionSplit,
    output_paths: RunOutputPaths,
    *,
    app_config: AppConfig,
    parsed_arguments: argparse.Namespace,
    device: torch.device,
    normalized_path: Path,
    candlestick_path: Path | None,
    show_epochs: bool,
    experiment_signature: str,
    tensor_session_cache: dict[int, TensorSession],
) -> TrainingResult:
    """Train one fresh model without carrying weights across temporal runs."""

    training_started_at = perf_counter()

    # Each run fits preprocessing from its own expanding historical training pool. Reusing
    # a later run's statistics would leak future market values backward in time.
    standardizer = fit_feature_standardizer(split.training)
    majority_classes = fit_majority_classes(split.training)
    data_loaders = create_training_data_loaders(
        split,
        timestamp_column=app_config.data.timestamp_column,
        batch_size=parsed_arguments.batch_size,
        random_seed=app_config.project.random_seed,
        tensor_session_cache=tensor_session_cache,
    )

    # Fresh initialization keeps the stated protocol honest: this model learns
    # from exactly its active training window, not hidden weights from an older run.
    torch.manual_seed(app_config.project.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(app_config.project.random_seed)

    model = RegimeTransformer(_model_config(parsed_arguments))
    training_config = TrainingLoopConfig(
        epochs=parsed_arguments.epochs,
        learning_rate=parsed_arguments.learning_rate,
        weight_decay=parsed_arguments.weight_decay,
        gradient_clip=parsed_arguments.gradient_clip,
        early_stopping_patience=parsed_arguments.patience,
    )
    epoch_callback: Callable[[EpochRecord], None] | None = (
        _print_epoch if show_epochs else None
    )
    result = train_regime_transformer(
        model,
        data_loaders,
        standardizer,
        config=training_config,
        device=device,
        majority_classes=majority_classes,
        run_index=run.run_index,
        epoch_callback=epoch_callback,
    )

    experiment_metadata = _experiment_metadata(
        run,
        plan,
        result,
        app_config=app_config,
        normalized_path=normalized_path,
        candlestick_path=candlestick_path,
        database_path=parsed_arguments.database,
        output_paths=output_paths,
    )
    experiment_metadata["experiment_signature"] = experiment_signature
    experiment_metadata["training_elapsed_seconds"] = perf_counter() - training_started_at
    experiment_metadata.update(_partition_metadata(split))

    # Predictions are committed first so their exact bytes can be recorded inside both the
    # checkpoint and final metadata. Metadata is the completion marker and is published last.
    save_test_predictions(
        output_paths.predictions,
        result.test_predictions,
        overwrite=parsed_arguments.overwrite or parsed_arguments.resume,
    )
    experiment_metadata["test_predictions_sha256"] = file_sha256(output_paths.predictions)
    save_training_checkpoint(
        output_paths.checkpoint,
        model,
        standardizer,
        training_config,
        result,
        experiment_metadata=experiment_metadata,
        overwrite=parsed_arguments.overwrite or parsed_arguments.resume,
    )
    experiment_metadata["checkpoint_sha256"] = file_sha256(output_paths.checkpoint)
    _save_json_atomically(
        output_paths.metadata,
        experiment_metadata,
        overwrite=parsed_arguments.overwrite,
    )
    print(f"  Training elapsed: {experiment_metadata['training_elapsed_seconds']:.2f}s")
    return result


def _partition_metadata(split: AnnotatedSessionSplit) -> dict[str, Any]:
    """Describe real candles and both human-label distributions after evaluation finishes.

    Distributions are audit evidence only. They never change loss weighting, sampling,
    standardization, or checkpoint selection, including when later history changes them.
    """

    real_candle_counts = {}
    label_distributions = {}
    for role in ("training", "validation", "test"):
        sessions = getattr(split, role)
        real_candle_counts[role] = sum(len(session) for session in sessions)
        label_distributions[role] = {}

        # Keep absent classes visible as zero counts so regime drift can be compared between
        # runs without treating a missing dictionary key as missing measurement.
        for head in ("current", "anticipated"):
            target_counts = pd.concat(
                [session[f"{head}_target"] for session in sessions], ignore_index=True
            ).value_counts()
            label_distributions[role][head] = {
                regime.value: int(target_counts.get(class_index, 0))
                for regime, class_index in REGIME_TO_CLASS.items()
            }

    return {
        "real_candle_counts": real_candle_counts,
        "total_real_candle_count": sum(real_candle_counts.values()),
        "label_distributions": label_distributions,
    }


def _model_config(arguments: argparse.Namespace) -> TransformerConfig:
    """Scale hidden capacity without changing the input features or temporal protocol."""

    if arguments.model_dimension <= 0 or arguments.model_dimension % 3:
        raise ValueError(
            "Model dimension must be positive and divisible by three attention heads."
        )

    # Keep the existing four-to-one feed-forward ratio when testing wider representations.
    # The architecture is stored in each checkpoint and included in the resume signature.
    return TransformerConfig(
        feature_count=len(FEATURE_COLUMNS),
        model_dimension=arguments.model_dimension,
        feedforward_dimension=4 * arguments.model_dimension,
    )


def _experiment_signature(
    run: WalkForwardRun,
    sessions: Sequence[pd.DataFrame],
    app_config: AppConfig,
    arguments: argparse.Namespace,
) -> str:
    """Identify inputs and settings so resume cannot silently reuse a different experiment."""

    settings = {
        "protocol": "expanding_history_v1",
        "initialization_mode": "fresh",
        "run": asdict(run),
        "config": asdict(app_config),
        "training": {
            name: getattr(arguments, name)
            for name in (
                "epochs", "batch_size", "learning_rate", "weight_decay",
                "gradient_clip", "patience",
            )
        },
        "model": asdict(_model_config(arguments)),
    }
    digest = hashlib.sha256(json.dumps(settings, sort_keys=True, default=str).encode())

    # Include both features and human labels. A correction in SQLite must invalidate reuse
    # even if the calendar boundaries still look identical. Source paths may safely move.
    for session in sessions[run.training.start_index : run.test.end_index]:
        digest.update(pd.util.hash_pandas_object(session, index=False).to_numpy().tobytes())

    return digest.hexdigest()


def _run_needs_training(
    paths: RunOutputPaths,
    *,
    resume: bool,
    overwrite: bool,
    config: AppConfig,
    experiment_signature: str,
) -> bool:
    """Protect completed bundles; only explicit resume may rebuild an interrupted run."""

    existing_outputs = any(
        path.exists() for path in (paths.checkpoint, paths.predictions, paths.metadata)
    )
    if not existing_outputs:
        return True

    if overwrite:
        print(f"Replacing selected run with --overwrite: {paths.directory}")
        return True

    if not resume:
        raise FileExistsError(
            f"Run output already exists: {paths.directory}. Use --resume or --overwrite."
        )

    # Missing metadata means publication never finished. There is no supported optimizer
    # snapshot, so resume means rebuilding this experiment, not continuing a partial epoch.
    if not paths.metadata.exists():
        print(f"Rebuilding incomplete run from epoch 1: {paths.directory}")
        return True

    try:
        metadata, _ = load_run_artifacts(paths.directory, config=config)
        if metadata.get("experiment_signature") != experiment_signature:
            raise ValueError("inputs or training settings differ, or this is an older run")
    except (ValueError, OSError) as error:
        raise ValueError(
            f"Cannot resume {paths.directory}: {error}. "
            "Inspect the bundle or use --overwrite for explicit replacement."
        ) from error

    print(f"Skipping verified completed run: {paths.directory}")
    return False


def main(arguments: Sequence[str] | None = None) -> int:
    """Train one run or every complete chronological walk-forward run."""

    argument_parser = build_argument_parser()
    parsed_arguments = argument_parser.parse_args(arguments)

    try:
        # A misspelled output suffix must fail before an expensive fit, not at publication.
        if not parsed_arguments.walk_forward and parsed_arguments.checkpoint.suffix != ".pt":
            raise ValueError("Training checkpoints must use the .pt suffix.")

        if not parsed_arguments.database.is_file():
            raise FileNotFoundError(
                f"Annotation database does not exist: {parsed_arguments.database}"
            )

        app_config = load_config(parsed_arguments.config)
        normalized_data = pd.read_parquet(parsed_arguments.normalized)
        candlestick_path = parsed_arguments.candlesticks or _infer_candlestick_path(
            parsed_arguments.normalized
        )
        if candlestick_path is not None and not candlestick_path.is_file():
            raise FileNotFoundError(
                f"OHLC candlestick artifact does not exist: {candlestick_path}"
            )

        # Read annotations once. A GUI save after this point belongs to a later
        # experiment and cannot alter roles halfway through the active run.
        with closing(AnnotationStore(parsed_arguments.database)) as store:
            annotations = store.load_all()

        sessions = build_complete_annotated_sessions(
            normalized_data,
            annotations,
            timestamp_column=app_config.data.timestamp_column,
            interval=app_config.data.target_interval,
            session_timezone=app_config.data.session_timezone,
        )
        plan = plan_walk_forward_runs(
            sessions,
            initial_training_session_count=parsed_arguments.training_sessions,
            validation_session_count=parsed_arguments.validation_sessions,
            test_session_count=parsed_arguments.test_sessions,
        )
        if not plan.runs:
            required_sessions = (
                parsed_arguments.training_sessions
                + parsed_arguments.validation_sessions
                + parsed_arguments.test_sessions
            )
            raise ValueError(
                f"Training requires {required_sessions} complete sessions, "
                f"but {len(sessions)} were supplied."
            )

        # The simple mode means exactly one declared experiment. Longer history
        # requires --walk-forward so extra sessions are never silently discarded.
        if not parsed_arguments.walk_forward and len(sessions) != (
            parsed_arguments.training_sessions
            + parsed_arguments.validation_sessions
            + parsed_arguments.test_sessions
        ):
            raise ValueError(
                "Single-run training requires an exact session count. "
                "Use --walk-forward for a longer annotated corpus."
            )

        selected_runs = plan.runs if parsed_arguments.walk_forward else plan.runs[:1]
        selected_index = parsed_arguments.run_index or parsed_arguments.start_run
        if parsed_arguments.run_index is not None or parsed_arguments.start_run is not None:
            if not parsed_arguments.walk_forward:
                raise ValueError("--run-index and --start-run require --walk-forward.")
            if selected_index is None or not 1 <= selected_index <= len(plan.runs):
                raise ValueError("Selected run index is outside the planned run range.")
            selected_runs = (
                plan.runs[selected_index - 1 : selected_index]
                if parsed_arguments.run_index is not None
                else plan.runs[selected_index - 1 :]
            )
        output_paths = [
            _run_output_paths(
                run,
                walk_forward=parsed_arguments.walk_forward,
                checkpoint_path=parsed_arguments.checkpoint,
                output_directory=parsed_arguments.output_directory,
            )
            for run in selected_runs
        ]
        # Inspect all selected outputs before fitting any model. A conflict in a later run
        # should not waste hours training earlier runs before the command can report it.
        experiment_signatures = [
            _experiment_signature(run, sessions, app_config, parsed_arguments)
            for run in selected_runs
        ]
        needs_training = [
            _run_needs_training(
                paths,
                resume=parsed_arguments.resume,
                overwrite=parsed_arguments.overwrite,
                config=app_config,
                experiment_signature=signature,
            )
            for paths, signature in zip(output_paths, experiment_signatures, strict=True)
        ]
        device = resolve_training_device(parsed_arguments.device)

        print("Price Sanity Training")
        print(f"Device: {device}")
        print(
            f"Protocol: initially {plan.initial_training_session_count} train / "
            f"{plan.validation_session_count} validation / "
            f"{plan.test_session_count} test"
        )
        if parsed_arguments.walk_forward:
            print(
                f"Runs: {len(selected_runs)} | Training growth: "
                f"{plan.training_growth_session_count} sessions | "
                f"Trailing sessions: {plan.trailing_session_count}"
            )

        # Session tensors contain only immutable CPU inputs and targets, not fitted scaling.
        # Reuse overlapping windows without reusing weights or statistics across experiments.
        tensor_session_cache: dict[int, TensorSession] = {}
        for run, run_paths, signature, should_train in zip(
            selected_runs, output_paths, experiment_signatures, needs_training, strict=True
        ):
            if not should_train:
                continue

            # Remove the old completion marker before explicit replacement starts. A crash
            # must leave an incomplete bundle, never stale metadata claiming it is complete.
            if parsed_arguments.overwrite:
                run_paths.metadata.unlink(missing_ok=True)
            print(
                f"Run {run.run_index} / {len(plan.runs)} | "
                f"train {run.training.start_date} to {run.training.end_date} | "
                f"validation {run.validation.start_date} to {run.validation.end_date} | "
                f"test {run.test.start_date} to {run.test.end_date}"
            )
            split = run.select_sessions(sessions)
            total_real_candle_count = sum(
                len(session)
                for partition in (split.training, split.validation, split.test)
                for session in partition
            )
            print(
                f"  Sessions: {len(split.training)} train / "
                f"{len(split.validation)} validation / {len(split.test)} test | "
                f"Real candles: {total_real_candle_count}"
            )
            result = _train_one_run(
                run,
                plan,
                split,
                run_paths,
                app_config=app_config,
                parsed_arguments=parsed_arguments,
                device=device,
                normalized_path=parsed_arguments.normalized,
                candlestick_path=candlestick_path,
                show_epochs=True,
                experiment_signature=signature,
                tensor_session_cache=tensor_session_cache,
            )

            print(
                f"  Best epoch {result.best_epoch} | "
                f"test current accuracy {result.test.current.accuracy:.3f}, "
                f"macro-F1 {result.test.current.macro_f1:.3f} "
                f"(majority {result.majority_current.macro_f1:.3f}) | "
                f"test anticipated accuracy {result.test.anticipated.accuracy:.3f}, "
                f"macro-F1 {result.test.anticipated.macro_f1:.3f} "
                f"(majority {result.majority_anticipated.macro_f1:.3f})"
            )
            print(f"  Checkpoint: {run_paths.checkpoint}")
            print(f"  Predictions: {run_paths.predictions}")
            if candlestick_path is not None:
                print("  View test results:")
                print(
                    "  pricesanity-test "
                    f"--run {run_paths.directory} "
                    f"--candlesticks {candlestick_path} "
                    f"--config {parsed_arguments.config}"
                )

    except (FileExistsError, FileNotFoundError, ValueError) as error:
        argument_parser.error(str(error))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
