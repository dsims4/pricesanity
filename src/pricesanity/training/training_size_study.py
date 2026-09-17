"""Run a frozen, fixed-epoch learning curve without holdout checkpoint selection."""

import argparse
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import pandas as pd
import torch

from pricesanity.training import tuning
from pricesanity.training.model import TransformerConfig
from pricesanity.training.trainer import (
    REGIME_CLASS_BY_NAME,
    _evaluate_test_once,
    _run_epoch,
    calculate_model_state_sha256,
    fit_feature_standardizer,
    fit_majority_classes,
)
from pricesanity.training.tuning_context import ContextExperimentTransformer


TRAINING_COUNTS = (100, 120, 140, 150)


def select_sessions(sessions, training_count):
    """Keep the same final 26 sessions behind every chronological training prefix."""

    # Fixed corpus size and declared prefixes make this one reproducible study rather than a general
    # split helper whose boundaries could drift between invocations.
    if len(sessions) != 176 or training_count not in TRAINING_COUNTS:
        raise ValueError("The study requires 176 sessions and a declared training prefix.")
    dates = [str(session["session_date"].iloc[0]) for session in sessions]

    # Preserve supplied chronology exactly; sorting here would hide duplicate or reordered research
    # sessions and change which dates enter each training prefix.
    if dates != sorted(set(dates)):
        raise ValueError("Study sessions must be unique and chronological.")
    return tuple(sessions[:training_count]), tuple(sessions[150:176])


def prediction_diagnostics(predictions):
    """Reuse saved rows for class/transition reports instead of repeating inference."""

    report = {}

    # Compute each label question independently from saved predictions so no inference is repeated
    # and one head cannot hide the other's class or transition behavior.
    for head in ("current", "anticipated"):
        targets = predictions[f"human_{head}_regime"].map(REGIME_CLASS_BY_NAME)
        predicted = predictions[f"predicted_{head}_regime"].map(REGIME_CLASS_BY_NAME)
        report[head] = tuning.classification_report(targets, predicted)
        session_targets, session_predictions = [], []

        # Transition reports receive session-separated lists, preventing an overnight boundary from
        # being interpreted as an intraday regime change.
        for _, session in predictions.groupby("session_date", sort=False):
            session_targets.append(
                session[f"human_{head}_regime"].map(REGIME_CLASS_BY_NAME).tolist()
            )
            session_predictions.append(
                session[f"predicted_{head}_regime"].map(REGIME_CLASS_BY_NAME).tolist()
            )
        report[head]["transitions"] = tuning.transition_report(
            session_targets, session_predictions,
        )
    return report


def run_fixed_candidate(training, holdout, tensor_cache, settings, root, snapshot_sha256):
    """Save the final fixed epoch; holdout results cannot select or change any weights.

    Each candidate starts from the same seed and fits its own train-only statistics.
    A completed bundle is reused only with identical data, recipe, and file checksums.
    """

    training_dates = [str(session["session_date"].iloc[0]) for session in training]
    holdout_dates = [str(session["session_date"].iloc[0]) for session in holdout]

    # Date-level separation is checked before artifact lookup; a resumable directory cannot make an
    # overlapping train/holdout request scientifically valid.
    if not training_dates or not holdout_dates or max(training_dates) >= min(holdout_dates):
        raise ValueError("Training must precede the fixed holdout without overlap.")
    directory = root / f"train_{len(training):03d}"

    # Identity fixes the recipe, exact dates, source snapshot, environment, and final-epoch rule for
    # this training-size point.
    identity = {
        "settings": asdict(settings), "training_dates": training_dates,
        "holdout_dates": holdout_dates, "snapshot_sha256": snapshot_sha256,
        "device": "cpu", "torch_version": str(torch.__version__),
        "checkpoint_rule": "final fixed epoch; no validation selection",
    }
    files = ("settings.json", "checkpoint.pt", "report.json", "predictions.parquet")
    completion_path = directory / "completion.json"

    # Completion is reusable only when every component still matches its recorded checksum and the
    # requested experiment identity is unchanged.
    if completion_path.exists():
        completed = tuning._read_json_mapping(completion_path, "study completion")
        checksums = {name: tuning._file_sha256(directory / name) for name in files}
        if completed != {"identity": identity, "checksums": checksums}:
            raise ValueError(f"Completed study candidate identity changed: {directory.name}")
        print(f"Reusing {directory.name}", flush=True)
        return tuning._read_json_mapping(directory / "report.json", "study report")

    directory.mkdir(exist_ok=True)
    settings_path = directory / "settings.json"

    # Preserve incompatible partial work for inspection rather than replacing its settings under
    # the same human-readable training-size directory.
    if settings_path.exists() and tuning._read_json_mapping(settings_path, "settings") != identity:
        raise ValueError("Partial candidate settings differ; preserve it under its original name.")
    tuning.write_json(settings_path, identity)
    started = perf_counter()

    # This controlled size study is CPU-only, so hardware variation cannot masquerade as a scaling
    # effect across training prefixes.
    device = torch.device("cpu")

    # Reconstruct only architecture fields from the frozen recipe; optimizer and duration remain
    # separate training settings persisted in identity.
    configuration = TransformerConfig(**{
        name: getattr(settings, name) for name in (
            "model_dimension", "layer_count", "attention_head_count",
            "feedforward_dimension", "dropout",
        )
    })

    # Fixed seed means an independent initialization, not continuation from the smaller run.
    # The extra histories may change statistics; those changes belong only to that model.
    torch.manual_seed(settings.random_seed)
    model = ContextExperimentTransformer(configuration, settings.context_length).to(device)

    # Fit preprocessing anew on this prefix only. Larger runs may learn different statistics, but
    # no holdout candle contributes to any candidate's scale.
    standardizer = fit_feature_standardizer(training).to(device)
    training_tensors = [tensor_cache[id(session)] for session in training]
    training_loader = tuning.build_loader(training_tensors, settings, shuffle=True)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=settings.learning_rate, weight_decay=settings.weight_decay,
    )
    history = []

    # Run exactly the fixed epoch count and retain every training metric for audit; there is no
    # validation-driven checkpoint choice in this learning-curve experiment.
    for epoch in range(1, settings.epochs + 1):
        metrics = _run_epoch(
            model, training_loader, standardizer, device=device, optimizer=optimizer,
            gradient_clip=1.0, loss_setup=settings.loss_setup,
        )
        history.append({"epoch": epoch, "training": asdict(metrics)})

    # No minimum-loss search or early stopping occurs here. Even training-loss selection
    # would introduce a different duration across candidates and undermine this comparison.
    model_identity = calculate_model_state_sha256(model)

    # Publish checkpoint bytes by replacement only after the final epoch, preventing a partial save
    # from occupying the canonical model path.
    temporary_checkpoint = directory / "checkpoint.partial.pt"
    torch.save({
        "model_state": model.state_dict(), "model_config": asdict(configuration),
        "settings": asdict(settings), "feature_mean": standardizer.mean,
        "feature_standard_deviation": standardizer.standard_deviation,
        "model_state_sha256": model_identity,
    }, temporary_checkpoint)
    temporary_checkpoint.replace(directory / "checkpoint.pt")
    training_metrics = _run_epoch(
        model, tuning.build_loader(training_tensors, settings, shuffle=False), standardizer,
        device=device, optimizer=None, gradient_clip=1.0, loss_setup=settings.loss_setup,
    )

    # Reuse the existing read-only prediction collector only after weights are fixed.
    # Here its supplied data is explicitly the new 151–176 research holdout.
    holdout_loader = tuning.build_loader(
        [tensor_cache[id(session)] for session in holdout], settings, shuffle=False,
    )

    # Majority references are fit from the same training prefix and passed only to evaluation
    # diagnostics; they do not alter the frozen Transformer weights.
    holdout_metrics, _, _, predictions = _evaluate_test_once(
        model, holdout_loader, standardizer, device=device,
        majority_classes=fit_majority_classes(training), run_index=1,
        model_state_sha256=model_identity,
    )
    expected_ids = [identifier for session in holdout for identifier in session["candlestick_id"]]

    # Exact ordered IDs prove that batching and padding neither dropped nor reordered holdout rows.
    if predictions["candlestick_id"].tolist() != expected_ids:
        raise ValueError("Predictions do not exactly match the fixed holdout candles.")
    temporary_predictions = directory / "predictions.partial.parquet"
    predictions.to_parquet(temporary_predictions, index=False)
    temporary_predictions.replace(directory / "predictions.parquet")

    # The report combines immutable experiment identity with observed training and holdout evidence;
    # completion is published only after all referenced files exist.
    report = {
        **identity, "training_session_count": len(training),
        "training_candle_count": sum(len(session) for session in training),
        "holdout_session_count": len(holdout), "parameter_count": sum(
            parameter.numel() for parameter in model.parameters()
        ),
        "epoch_count": settings.epochs, "history": history,
        "training": asdict(training_metrics), "holdout_loss": asdict(holdout_metrics),
        "holdout": prediction_diagnostics(predictions), "model_state_sha256": model_identity,
        "elapsed_seconds": perf_counter() - started,
    }
    tuning.write_json(directory / "report.json", report)
    tuning.write_json(completion_path, {
        "identity": identity,
        "checksums": {name: tuning._file_sha256(directory / name) for name in files},
    })
    print(f"Completed {directory.name}: {settings.epochs} fixed epochs", flush=True)
    return report


def load_study(root):
    """Read the frozen research snapshot; never consult the live annotation database."""

    # The plan points back to the completed tuning pass only as protected provenance; this study
    # writes solely beneath its separate root.
    plan = tuning._read_json_mapping(root / "study_plan.json", "study plan")
    protected = Path(plan["protected_directory"]).resolve()
    if root.resolve() == protected or protected in root.resolve().parents:
        raise ValueError("The completed tuning directory is read-only for this study.")
    identity = tuning._read_json_mapping(root / "study_identity.json", "study identity")
    recipe_path = root / "frozen_recipe.json"
    recipe = tuning._read_json_mapping(recipe_path, "frozen recipe")
    if (
        identity != {"plan_sha256": tuning._file_sha256(root / "study_plan.json")}
        or tuning._file_sha256(recipe_path) != plan["recipe_sha256"]
        or recipe["settings"] != plan["settings"]
        or plan["training_counts"] != list(TRAINING_COUNTS)
        or plan["holdout_indices"] != [151, 176]
    ):
        raise ValueError("Frozen study recipe or comparison plan changed.")

    # Verify bytes before loading the frozen snapshot, then verify semantic session identity after
    # Parquet decoding so format-level integrity and research meaning are both covered.
    snapshot = root / "study_snapshot.parquet"
    if tuning._file_sha256(snapshot) != plan["snapshot_file_sha256"]:
        raise ValueError("Study snapshot checksum changed.")
    frozen = pd.read_parquet(snapshot)
    frozen["session_date"] = pd.to_datetime(frozen["session_date"]).dt.date
    sessions = tuple(session.reset_index(drop=True) for _, session in
                     frozen.groupby("session_date", sort=False))

    # Reuse the strict split helper to validate corpus size and chronology before comparing its
    # semantic digest with the plan.
    select_sessions(sessions, 100)
    if (
        [str(session["session_date"].iloc[0]) for session in sessions] != plan["session_dates"]
        or tuning._semantic_snapshot_sha256(sessions) != plan["snapshot_semantic_sha256"]
    ):
        raise ValueError("Study session identity changed.")
    return plan, sessions


def main(arguments=None):
    """Resume four fixed research candidates from an already-frozen study plan."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study_directory", type=Path)
    root = parser.parse_args(arguments).study_directory
    plan, sessions = load_study(root)
    settings = tuning.TuningSettings(**plan["settings"])

    # Convert every frozen session once because all four candidates reuse overlapping prefixes and
    # the identical forward holdout.
    tensor_cache = {id(session): tuning.convert_session_to_tensors(
        session, timestamp_column=plan["timestamp_column"],
    ) for session in sessions}
    reports = []

    # Candidates execute in increasing prefix order only for readable progress; each fit starts from
    # the same independent seed and never continues from the previous model.
    for training_count in TRAINING_COUNTS:
        training, holdout = select_sessions(sessions, training_count)
        reports.append(run_fixed_candidate(
            training, holdout, tensor_cache, settings, root, plan["snapshot_semantic_sha256"],
        ))
    tuning.write_json(root / "learning_curve.json", {"candidates": reports})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
