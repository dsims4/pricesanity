"""Run recorded training/validation diagnostics without evaluating held-out test sessions."""

import argparse
from copy import deepcopy
from contextlib import closing
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
from pathlib import Path
import sqlite3
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
import torch

from pricesanity.annotation.schema import CandlestickAnnotation, MarketRegime
from pricesanity.config import load_config
from pricesanity.training.dataset import (
    AnnotatedSessionSplit,
    TensorSessionDataset,
    build_complete_annotated_sessions,
    collate_tensor_sessions,
    convert_session_to_tensors,
)
from pricesanity.training.model import TransformerConfig
from pricesanity.training.tuning_context import ContextExperimentTransformer
from pricesanity.training.trainer import (
    _run_epoch,
    fit_feature_standardizer,
    fit_majority_classes,
    resolve_training_device,
)
from pricesanity.training.walk_forward import plan_walk_forward_runs


SELECTION_MARGIN_MEAN_F1 = 0.005
SNAPSHOT_IDENTITY_FILENAME = "tuning_identity.json"
EXPERIMENT_IDENTITY_FILENAME = "artifact_identity.json"


@dataclass(frozen=True)
class TuningSettings:
    """Fully specified experiment; changing a setting never alters an existing run."""

    model_dimension: int = 48
    layer_count: int = 2
    attention_head_count: int = 3
    feedforward_dimension: int = 192
    dropout: float = 0.1
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    batch_size: int = 8
    epochs: int = 50
    patience: int = 10
    random_seed: int = 42
    context_length: int | None = None
    loss_setup: str = "both"


def classification_report(targets, predictions) -> dict:
    """Report every class, including absent classes, with truth as confusion-matrix rows."""

    targets = np.asarray(targets, dtype=int)
    predictions = np.asarray(predictions, dtype=int)
    confusion = np.bincount(3 * targets + predictions, minlength=9).reshape(3, 3)
    per_class = {}
    for class_index, regime in enumerate(("bull", "bear", "range")):
        true_positive = int(confusion[class_index, class_index])
        support = int(confusion[class_index].sum())
        predicted_count = int(confusion[:, class_index].sum())
        precision = true_positive / predicted_count if predicted_count else 0.0
        recall = true_positive / support if support else 0.0
        per_class[regime] = {
            "support": support,
            "precision": precision,
            "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        }

    return {
        "support": len(targets),
        "accuracy": float(np.trace(confusion) / len(targets)) if len(targets) else None,
        "macro_f1": float(np.mean([values["f1"] for values in per_class.values()])),
        "per_class": per_class,
        "confusion_matrix": confusion.tolist(),
    }


def transition_report(session_targets, session_predictions) -> dict:
    """Compare classification near human changes, clipping neighborhoods to each session.

    These are candle classification scores within a tolerance neighborhood, not a
    claim that several predicted changes successfully match one true change event.
    """

    report = {}
    for radius in (0, 1, 2):
        selected_targets, selected_predictions = [], []
        other_targets, other_predictions = [], []
        for targets, predictions in zip(session_targets, session_predictions, strict=True):
            targets = np.asarray(targets)
            predictions = np.asarray(predictions)
            change_indices = np.flatnonzero(targets[1:] != targets[:-1]) + 1
            selected = np.zeros(len(targets), dtype=bool)

            # The first candle establishes a regime; yesterday's final label cannot create
            # a transition or a tolerance neighborhood in today's independent sequence.
            for position in change_indices:
                selected[max(0, position - radius):position + radius + 1] = True
            selected_targets.extend(targets[selected])
            selected_predictions.extend(predictions[selected])
            other_targets.extend(targets[~selected])
            other_predictions.extend(predictions[~selected])

        report[str(radius)] = {
            "near_change": classification_report(selected_targets, selected_predictions),
            "other_candles": classification_report(other_targets, other_predictions),
        }

    return report


def build_loader(tensor_sessions, settings, *, shuffle):
    """Shuffle complete sessions only; keep natural early-close lengths and candle order."""

    return torch.utils.data.DataLoader(
        TensorSessionDataset(tensor_sessions),
        batch_size=settings.batch_size,
        shuffle=shuffle,
        generator=torch.Generator().manual_seed(settings.random_seed),
        collate_fn=collate_tensor_sessions,
        num_workers=0,
    )


def evaluate_labels(model, loader, standardizer, device, majority_classes) -> dict:
    """Inspect training or validation labels without invoking the official test evaluator."""

    model.eval()
    targets_by_head = {"current": [], "anticipated": []}
    predictions_by_head = {"current": [], "anticipated": []}
    with torch.inference_mode():
        for batch in loader:
            output = model(standardizer.transform(batch.features.to(device)),
                           batch.padding_mask.to(device))
            for head in targets_by_head:
                predictions = getattr(output, f"{head}_logits").argmax(dim=-1).cpu()
                targets = getattr(batch, f"{head}_targets")
                for position, length in enumerate(batch.lengths.tolist()):
                    targets_by_head[head].append(targets[position, :length].tolist())
                    predictions_by_head[head].append(predictions[position, :length].tolist())

    report = {}
    for head in targets_by_head:
        targets = np.concatenate(targets_by_head[head])
        predictions = np.concatenate(predictions_by_head[head])
        report[head] = classification_report(targets, predictions)
        report[head]["majority_baseline"] = classification_report(
            targets, np.full(len(targets), getattr(majority_classes, head))
        )
        report[head]["transitions"] = transition_report(
            targets_by_head[head], predictions_by_head[head]
        )

    return report


def write_json(path: Path, content: dict) -> None:
    """Publish a complete diagnostic report without exposing partially written JSON."""

    temporary_path = path.with_suffix(".partial.json")
    temporary_path.write_text(json.dumps(content, indent=2, allow_nan=False) + "\n")
    temporary_path.replace(path)


def _file_sha256(path: Path) -> str:
    """Hash a complete artifact without loading all of it into memory."""

    with path.open("rb") as artifact_file:
        return hashlib.file_digest(artifact_file, "sha256").hexdigest()


def _json_sha256(content: object) -> str:
    """Identify settings independently of JSON indentation or key order."""

    serialized = json.dumps(
        content,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def _read_json_mapping(path: Path, description: str) -> dict[str, Any]:
    """Load one named JSON object and turn malformed state into a clear error."""

    try:
        content = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not read {description}: {path}") from error
    if not isinstance(content, dict):
        raise ValueError(f"{description.capitalize()} must contain a named mapping: {path}")
    return content


def _semantic_snapshot_sha256(sessions) -> str:
    """Hash the ordered training/validation values rather than Parquet encoding details."""

    snapshot_hash = hashlib.sha256()
    for session in sessions:
        snapshot_hash.update(
            pd.util.hash_pandas_object(session, index=False).to_numpy().tobytes()
        )
    return snapshot_hash.hexdigest()


def _experiment_identity(
    *,
    snapshot_sha256: str,
    settings: TuningSettings,
    report_path: Path,
    weights_path: Path,
) -> dict[str, Any]:
    """Bind one report and weight file to the frozen tuning data and settings."""

    return {
        "format_version": 1,
        "snapshot_semantic_sha256": snapshot_sha256,
        "settings_sha256": _json_sha256(asdict(settings)),
        "report_sha256": _file_sha256(report_path),
        "diagnostic_weights_sha256": _file_sha256(weights_path),
    }


def _validate_completed_experiment(
    name,
    training_sessions,
    validation_sessions,
    settings,
    directory,
    device,
    *,
    snapshot_sha256,
    allow_legacy_migration,
    timestamp_column="ts_event",
) -> dict:
    """Verify a reusable result and migrate one valid legacy bundle exactly once."""

    report_path = directory / "report.json"
    weights_path = directory / "diagnostic_weights.pt"
    identity_path = directory / EXPERIMENT_IDENTITY_FILENAME
    settings_path = directory / "settings.json"

    existing_report = _read_json_mapping(report_path, "experiment report")
    expected_settings = asdict(settings)
    configuration = TransformerConfig(
        model_dimension=settings.model_dimension,
        attention_head_count=settings.attention_head_count,
        layer_count=settings.layer_count,
        feedforward_dimension=settings.feedforward_dimension,
        dropout=settings.dropout,
    )
    expected_configuration = asdict(configuration)
    required_report_fields = {
        "name",
        "settings",
        "model_config",
        "device",
        "torch_version",
        "training_session_count",
        "validation_session_count",
        "best_epoch",
        "history",
        "training",
        "validation",
        "test",
    }
    if required_report_fields.difference(existing_report):
        raise ValueError(f"Completed experiment report is malformed: {name}")
    if (
        existing_report["name"] != name
        or existing_report["settings"] != expected_settings
        or existing_report["model_config"] != expected_configuration
        or existing_report["device"] != str(device)
        or existing_report["torch_version"] != str(torch.__version__)
        or existing_report["training_session_count"] != len(training_sessions)
        or existing_report["validation_session_count"] != len(validation_sessions)
        or existing_report["test"] is not None
    ):
        raise ValueError(
            f"Existing experiment has different data, settings, or environment: {name}"
        )
    history = existing_report["history"]
    best_epoch = existing_report["best_epoch"]
    if (
        not isinstance(history, list)
        or not isinstance(best_epoch, int)
        or not 1 <= best_epoch <= len(history)
        or (validation_sessions and not isinstance(existing_report["validation"], dict))
        or (not validation_sessions and existing_report["validation"] is not None)
    ):
        raise ValueError(f"Completed experiment report is malformed: {name}")

    if validation_sessions:
        # A valid-looking score is insufficient if the saved epoch was chosen using a
        # different rule. Preserve the original validation-loss checkpoint selection.
        try:
            validation_losses = [epoch["validation"]["total_loss"] for epoch in history]
            selected_loss = existing_report["validation_loss"]
            if (
                not all(math.isfinite(loss) for loss in validation_losses)
                or selected_loss != validation_losses[best_epoch - 1]
                or best_epoch != validation_losses.index(min(validation_losses)) + 1
            ):
                raise ValueError("Inconsistent validation-selected epoch")
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"Completed experiment checkpoint selection is invalid: {name}"
            ) from error

    # A settings file is optional only for the earliest valid legacy results. If present,
    # it must agree with both the caller and report before any migration can trust the bundle.
    if settings_path.exists():
        recorded_settings = _read_json_mapping(settings_path, "experiment settings")
        if recorded_settings != expected_settings:
            raise ValueError(f"Existing experiment settings do not match: {name}")

    if not weights_path.is_file():
        raise ValueError(f"Completed experiment is missing diagnostic weights: {name}")
    try:
        weights = torch.load(weights_path, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError) as error:
        raise ValueError(f"Completed experiment weights are unreadable: {name}") from error
    required_weight_fields = {
        "model_state",
        "model_config",
        "settings",
        "feature_mean",
        "feature_standard_deviation",
    }
    if not isinstance(weights, dict) or required_weight_fields.difference(weights):
        raise ValueError(f"Completed experiment weights are malformed: {name}")
    if (
        weights["settings"] != expected_settings
        or weights["model_config"] != expected_configuration
    ):
        raise ValueError(f"Completed experiment weights do not match its report: {name}")

    feature_mean = weights["feature_mean"]
    feature_standard_deviation = weights["feature_standard_deviation"]
    if (
        not isinstance(feature_mean, torch.Tensor)
        or not isinstance(feature_standard_deviation, torch.Tensor)
        or feature_mean.shape != (configuration.feature_count,)
        or feature_standard_deviation.shape != (configuration.feature_count,)
        or not torch.isfinite(feature_mean).all()
        or not torch.isfinite(feature_standard_deviation).all()
        or not feature_standard_deviation.gt(0).all()
    ):
        raise ValueError(f"Completed experiment feature statistics are malformed: {name}")

    # Strict loading proves that every tensor has the expected name and shape. Finite checks
    # catch a completed-looking artifact that cannot safely reproduce its recorded model.
    try:
        validation_model = ContextExperimentTransformer(
            configuration,
            settings.context_length,
        )
        validation_model.load_state_dict(weights["model_state"], strict=True)
    except (RuntimeError, TypeError, ValueError) as error:
        raise ValueError(f"Completed experiment model state is incompatible: {name}") from error
    if any(not torch.isfinite(value).all() for value in validation_model.state_dict().values()):
        raise ValueError(f"Completed experiment model state is nonfinite: {name}")
    if existing_report.get("parameter_count") != sum(
        parameter.numel() for parameter in validation_model.parameters()
    ):
        raise ValueError(f"Completed experiment parameter count is inconsistent: {name}")

    expected_identity = _experiment_identity(
        snapshot_sha256=snapshot_sha256,
        settings=settings,
        report_path=report_path,
        weights_path=weights_path,
    )
    if identity_path.exists():
        recorded_identity = _read_json_mapping(identity_path, "artifact identity")
        if recorded_identity != expected_identity:
            raise ValueError(f"Completed experiment identity does not match: {name}")
    else:
        if not allow_legacy_migration:
            raise ValueError(
                f"Completed experiment has no artifact identity: {name}. "
                "Resume the original pass once to validate and migrate it."
            )

        # Old files have no historical checksum. Before establishing one, reproduce their
        # scores and train-only statistics from the frozen snapshot. Shapes alone cannot
        # distinguish two checkpoints of the same architecture from different experiments.
        standardizer = fit_feature_standardizer(training_sessions)
        if not (
            torch.equal(feature_mean, standardizer.mean)
            and torch.equal(feature_standard_deviation, standardizer.standard_deviation)
        ):
            raise ValueError(f"Completed experiment statistics differ from the snapshot: {name}")
        evaluation_sessions = validation_sessions or training_sessions
        evaluation_loader = build_loader(
            [convert_session_to_tensors(session, timestamp_column=timestamp_column)
             for session in evaluation_sessions],
            settings,
            shuffle=False,
        )
        reproduced = evaluate_labels(
            validation_model.to(device),
            evaluation_loader,
            standardizer.to(device),
            device,
            fit_majority_classes(training_sessions),
        )
        recorded = existing_report["validation" if validation_sessions else "training"]
        if reproduced != recorded:
            raise ValueError(f"Completed experiment scores do not reproduce: {name}")
        if validation_sessions:
            reproduced_loss = _run_epoch(
                validation_model,
                evaluation_loader,
                standardizer.to(device),
                device=device,
                optimizer=None,
                gradient_clip=1.0,
                loss_setup=settings.loss_setup,
            ).total_loss
            if not math.isclose(reproduced_loss, selected_loss, rel_tol=1e-5, abs_tol=1e-6):
                raise ValueError(
                    f"Completed experiment validation loss does not reproduce: {name}"
                )

        # Adding only this sidecar preserves every historical result and weight byte.
        write_json(identity_path, expected_identity)
        print(f"Validated legacy diagnostic identity: {name}", flush=True)

    print(f"Reusing completed diagnostic: {name}", flush=True)
    return existing_report


def run_experiment(
    name, training_sessions, validation_sessions, tensor_cache, settings,
    output_directory, device, *, snapshot_sha256, tiny=False,
    allow_legacy_migration=False,
) -> dict:
    """Fit fresh weights on the declared training subset and select only by validation loss.

    Tiny-set diagnostics have no validation/test access and stop at 98% training accuracy
    on both heads, or their declared epoch limit. They cannot select a generalization model.
    """

    directory = output_directory / name
    if (directory / "report.json").is_file():
        return _validate_completed_experiment(
            name,
            training_sessions,
            validation_sessions,
            settings,
            directory,
            device,
            snapshot_sha256=snapshot_sha256,
            allow_legacy_migration=allow_legacy_migration,
        )

    expected_settings = asdict(settings)
    settings_path = directory / "settings.json"
    identity_path = directory / EXPERIMENT_IDENTITY_FILENAME
    if directory.exists():
        if identity_path.exists():
            raise ValueError(
                f"Incomplete experiment unexpectedly has a completion identity: {name}"
            )
        if settings_path.exists():
            recorded_settings = _read_json_mapping(
                settings_path,
                "partial experiment settings",
            )
            if recorded_settings != expected_settings:
                raise ValueError(
                    f"Partial experiment has different settings; preserve it and use "
                    f"a new name: {name}"
                )
            print(f"Restarting incomplete diagnostic from epoch 1: {name}", flush=True)
        elif any(directory.iterdir()):
            raise ValueError(
                f"Partial experiment has no readable settings and cannot be resumed: {name}"
            )
    else:
        directory.mkdir()

    # Matching settings make an interrupted directory safe to reuse. Training deliberately
    # starts at epoch one because no optimizer/checkpoint continuation state is published.
    write_json(settings_path, expected_settings)
    started = perf_counter()
    configuration = TransformerConfig(
        model_dimension=settings.model_dimension,
        attention_head_count=settings.attention_head_count,
        layer_count=settings.layer_count,
        feedforward_dimension=settings.feedforward_dimension,
        dropout=settings.dropout,
    )

    # Every candidate receives the same initialization seed and immutable source snapshot.
    # Statistics are fitted afresh from this candidate's training membership, never validation.
    torch.manual_seed(settings.random_seed)
    model = ContextExperimentTransformer(configuration, settings.context_length).to(device)
    standardizer = fit_feature_standardizer(training_sessions).to(device)
    majority_classes = fit_majority_classes(training_sessions)
    training_loader = build_loader(
        [tensor_cache[id(session)] for session in training_sessions], settings, shuffle=True
    )
    training_evaluation_loader = build_loader(
        [tensor_cache[id(session)] for session in training_sessions], settings, shuffle=False
    )
    validation_loader = (
        build_loader([tensor_cache[id(session)] for session in validation_sessions],
                     settings, shuffle=False)
        if validation_sessions else None
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=settings.learning_rate, weight_decay=settings.weight_decay
    )
    best_loss = float("inf")
    best_state = None
    best_epoch = 0
    history = []
    tiny_passed = False
    for epoch in range(1, settings.epochs + 1):
        training = _run_epoch(
            model, training_loader, standardizer, device=device,
            optimizer=optimizer, gradient_clip=1.0, loss_setup=settings.loss_setup,
        )
        if not np.isfinite(training.total_loss):
            raise ValueError(f"Nonfinite training loss in {name} at epoch {epoch}.")
        record = {"epoch": epoch, "training": asdict(training)}
        if validation_loader is not None:
            validation = _run_epoch(
                model, validation_loader, standardizer, device=device,
                optimizer=None, gradient_clip=1.0, loss_setup=settings.loss_setup,
            )
            record["validation"] = asdict(validation)
            selection_loss = validation.total_loss
        else:
            selection_loss = training.total_loss
        history.append(record)

        if selection_loss < best_loss:
            best_loss = selection_loss
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())

        # Memorization is measured with dropout disabled on these same training candles.
        # No held-out session is repurposed as a tiny diagnostic example.
        if tiny and (epoch % 10 == 0 or epoch == settings.epochs):
            memorization = _run_epoch(
                model, training_evaluation_loader, standardizer, device=device,
                optimizer=None, gradient_clip=1.0, loss_setup=settings.loss_setup,
            )
            record["memorization"] = asdict(memorization)
            tiny_passed = min(memorization.current.accuracy,
                              memorization.anticipated.accuracy) >= 0.98
            if tiny_passed:
                best_state = deepcopy(model.state_dict())
                best_epoch = epoch
                break

        if epoch % 25 == 0:
            print(f"{name}: epoch {epoch}, training loss {training.total_loss:.4f}", flush=True)
        if not tiny and epoch - best_epoch >= settings.patience:
            break

    model.load_state_dict(best_state)
    training_evaluation = _run_epoch(
        model, training_evaluation_loader, standardizer, device=device,
        optimizer=None, gradient_clip=1.0, loss_setup=settings.loss_setup,
    )
    report = {
        "name": name, "settings": asdict(settings), "model_config": asdict(configuration),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "device": str(device), "torch_version": str(torch.__version__),
        "training_session_count": len(training_sessions),
        "validation_session_count": len(validation_sessions),
        "best_epoch": best_epoch, "history": history,
        "training_loss": asdict(training_evaluation),
        "training": evaluate_labels(model, training_evaluation_loader, standardizer,
                                    device, majority_classes),
        "validation": None,
        "test": None,
        "test_policy": "Not evaluated; no test measurements are used for tuning.",
        "tiny_passed": tiny_passed if tiny else None,
    }
    if validation_loader is not None:
        report["validation"] = evaluate_labels(
            model, validation_loader, standardizer, device, majority_classes
        )
        report["validation_loss"] = history[best_epoch - 1]["validation"]["total_loss"]
    report["elapsed_seconds"] = perf_counter() - started

    # These are diagnostic weights, not an official test-run bundle. The different filename
    # prevents a tuning artifact from being mistaken for a completed test evaluation in the GUI.
    weights_path = directory / "diagnostic_weights.pt"
    temporary_weights_path = directory / "diagnostic_weights.partial.pt"
    report_path = directory / "report.json"
    try:
        torch.save({
            "model_state": {key: value.cpu() for key, value in best_state.items()},
            "model_config": asdict(configuration), "settings": expected_settings,
            "feature_mean": standardizer.mean.cpu(),
            "feature_standard_deviation": standardizer.standard_deviation.cpu(),
        }, temporary_weights_path)
        temporary_weights_path.replace(weights_path)
    finally:
        temporary_weights_path.unlink(missing_ok=True)
    write_json(report_path, report)

    # The identity is written last. Its checksums make a completed report reusable only with
    # these exact weights, settings, and frozen training/validation snapshot.
    write_json(
        identity_path,
        _experiment_identity(
            snapshot_sha256=snapshot_sha256,
            settings=settings,
            report_path=report_path,
            weights_path=weights_path,
        ),
    )
    print(f"{name}: done, best epoch {best_epoch}, {report['elapsed_seconds']:.1f}s", flush=True)
    return report


def validation_score(report: dict) -> float:
    """Rank two-head candidates without consulting any test measurement."""

    return sum(report["validation"][head]["macro_f1"]
               for head in ("current", "anticipated")) / 2


def _select_stage_result(stage, reference, stage_reports, *, margin=SELECTION_MARGIN_MEAN_F1):
    """Keep the raw validation winner distinct from the declared practical choice."""

    if not stage_reports or reference not in stage_reports:
        raise ValueError("A tuning stage must include its reference result.")

    # Mean macro-F1 is primary. Validation loss breaks only an exact score tie and never
    # consults the reserved test sessions.
    raw_best = max(
        stage_reports,
        key=lambda report: (
            validation_score(report),
            -report["validation_loss"],
        ),
    )
    raw_score = validation_score(raw_best)
    comparable = [
        report
        for report in stage_reports
        if validation_score(report) >= raw_score - margin
    ]

    architecture_stages = {
        "model_width_matched_heads",
        "width_block_vs_previous",
        "depth",
        "attention_heads",
        "feedforward",
    }
    if stage == "context_length":
        # Context changes no parameter count, so that generic complexity measure cannot
        # distinguish 16 from 32 candles. Prefer the shortest comparable receptive field.
        def context_size(report):
            configured_length = report["settings"]["context_length"]
            return configured_length or report["model_config"]["maximum_session_length"]

        selected = min(
            comparable,
            key=lambda report: (
                context_size(report),
                report["parameter_count"],
                -validation_score(report),
                report["validation_loss"],
            ),
        )
        selection_basis = "shortest context within the mean-F1 margin"
    elif stage in architecture_stages:
        selected = min(
            comparable,
            key=lambda report: (
                report["parameter_count"],
                -validation_score(report),
                report["validation_loss"],
            ),
        )
        selection_basis = "fewest parameters within the mean-F1 margin"
    else:
        # Learning-rate, epoch, regularization, and batch changes are not inherently
        # simpler models. Keep the established reference unless a raw gain clears the margin.
        raw_gain = raw_score - validation_score(reference)
        selected = raw_best if raw_gain >= margin else reference
        selection_basis = (
            "raw gain cleared the mean-F1 margin"
            if selected is raw_best and raw_best is not reference
            else "reference retained because the raw gain did not clear the margin"
        )

    return raw_best, selected, selection_basis


def run_checklist(baseline, training_sessions, validation_sessions, tensor_cache,
                  output_directory, device, *, snapshot_sha256,
                  allow_legacy_migration=False) -> dict:
    """Compare one factor at a time, retaining the simpler model for negligible gains.

    The 0.005 mean-F1 margin is declared before sweeps. Single-head ablations are
    diagnostic only and cannot replace the selected two-head candidate implicitly.
    """

    selected = baseline
    experiments = {baseline["name"]: baseline}
    stages = []

    def persist_ledger(*, active_stage):
        """Keep interruption state separate from completed stage decisions."""

        write_json(output_directory / "comparison_ledger.json", {
            "selected": selected["name"],
            "selection_margin_mean_f1": SELECTION_MARGIN_MEAN_F1,
            "stages": stages,
            "completed_experiments": list(experiments),
            "active_stage": active_stage,
            "test_evaluated": False,
        })

    def persist_progress(*, status, current_stage=None, next_step=None):
        """Expose the current gate without leaving the original learning-rate note stale."""

        write_json(output_directory / "stage_summary.json", {
            "status": status,
            "baseline": baseline["name"],
            "tiny_passed": True,
            "selected": selected["name"],
            "completed_stage_count": len(stages),
            "last_completed_stage": stages[-1]["stage"] if stages else None,
            "current_stage": current_stage,
            "next_step": next_step,
            "test_evaluated": False,
        })

    def compare(stage, reference, candidates):
        nonlocal selected
        stage_reports = [reference]
        for name, candidate_settings in candidates:
            persist_progress(status="running", current_stage=stage)
            persist_ledger(active_stage=stage)
            existing = next((report for report in experiments.values()
                             if report["settings"] == asdict(candidate_settings)), None)
            report = existing or run_experiment(
                name, training_sessions, validation_sessions, tensor_cache,
                candidate_settings, output_directory, device,
                snapshot_sha256=snapshot_sha256,
                allow_legacy_migration=allow_legacy_migration,
            )
            experiments[report["name"]] = report
            if report not in stage_reports:
                stage_reports.append(report)

        raw_best, practical_selection, selection_basis = _select_stage_result(
            stage,
            reference,
            stage_reports,
        )
        stages.append({
            "stage": stage,
            "reference": reference["name"],
            "candidates": [
                {
                    "name": report["name"],
                    "mean_validation_macro_f1": validation_score(report),
                    "validation_loss": report["validation_loss"],
                    "parameter_count": report["parameter_count"],
                }
                for report in stage_reports
            ],
            "raw_best": raw_best["name"],
            "raw_best_mean_validation_macro_f1": validation_score(raw_best),
            "selected": practical_selection["name"],
            "selected_mean_validation_macro_f1": validation_score(practical_selection),
            "selected_distance_from_raw_best": (
                validation_score(raw_best) - validation_score(practical_selection)
            ),
            "selection_margin_mean_f1": SELECTION_MARGIN_MEAN_F1,
            "selection_basis": selection_basis,
        })
        selected = practical_selection
        persist_ledger(active_stage=None)
        persist_progress(status="running")
        print(
            f"Selected after {stage}: {selected['name']} "
            f"(mean validation F1 {validation_score(selected):.4f}; "
            f"raw best {raw_best['name']} at {validation_score(raw_best):.4f})",
            flush=True,
        )
        return practical_selection

    settings = TuningSettings(**selected["settings"])
    compare("learning_rate", selected, [
        (f"learning_rate_{rate:g}", replace(settings, learning_rate=rate))
        for rate in (1e-4, 3e-4, 1e-3, 3e-3)
    ])

    # Extend the epoch limit only when selection reaches its boundary; patience and every
    # other setting stay fixed, so this is a test of convergence time rather than architecture.
    for maximum_epochs in (100, 200):
        settings = TuningSettings(**selected["settings"])
        if selected["best_epoch"] < 0.8 * settings.epochs:
            break
        compare("epoch_limit", selected, [
            (f"epochs_{maximum_epochs}", replace(settings, epochs=maximum_epochs))
        ])

    settings = TuningSettings(**selected["settings"])
    compare("context_length", selected, [
        (f"context_{length}", replace(settings, context_length=length))
        for length in (16, 32, 64)
    ])

    # Widths 12, 32, and 64 all divide by four heads. Establish a matched four-head control
    # first so a width comparison does not quietly change attention-head count at the same time.
    settings = TuningSettings(**selected["settings"])
    persist_progress(status="running", current_stage="width_control")
    persist_ledger(active_stage="width_control")
    control = run_experiment(
        f"context_{settings.context_length}_width_control_four_heads",
        training_sessions,
        validation_sessions,
        tensor_cache,
        replace(settings, attention_head_count=4),
        output_directory,
        device,
        snapshot_sha256=snapshot_sha256,
        allow_legacy_migration=allow_legacy_migration,
    )
    experiments[control["name"]] = control
    before_width = selected
    width_winner = compare("model_width_matched_heads", control, [
        (
            f"context_{settings.context_length}_width_{width}",
            replace(settings, attention_head_count=4, model_dimension=width),
        )
        for width in (12, 32, 64)
    ])
    selected = compare("width_block_vs_previous", before_width, [
        (width_winner["name"], TuningSettings(**width_winner["settings"]))
    ])

    settings = TuningSettings(**selected["settings"])
    compare("depth", selected, [
        (f"layers_{layers}", replace(settings, layer_count=layers)) for layers in (3, 4)
    ])
    settings = TuningSettings(**selected["settings"])
    compare("attention_heads", selected, [
        (f"heads_{heads}", replace(settings, attention_head_count=heads))
        for heads in (2, 4, 8) if settings.model_dimension % heads == 0
    ])
    settings = TuningSettings(**selected["settings"])
    compare("feedforward", selected, [
        (f"feedforward_{ratio}x", replace(settings,
                                        feedforward_dimension=ratio * settings.model_dimension))
        for ratio in (2, 4)
    ])
    settings = TuningSettings(**selected["settings"])
    compare("dropout", selected, [
        (f"dropout_{dropout:g}", replace(settings, dropout=dropout)) for dropout in (0.0, 0.1, 0.2)
    ])
    settings = TuningSettings(**selected["settings"])
    compare("weight_decay", selected, [
        (f"weight_decay_{decay:g}", replace(settings, weight_decay=decay))
        for decay in (0.0, 1e-5, 1e-4, 1e-3)
    ])
    settings = TuningSettings(**selected["settings"])
    compare("batch_size", selected, [
        (f"batch_{batch_size}", replace(settings, batch_size=batch_size))
        for batch_size in (16, 32, 64, 128)
    ])

    settings = TuningSettings(**selected["settings"])
    for head in ("current", "anticipated"):
        persist_progress(status="running", current_stage=f"only_{head}")
        persist_ledger(active_stage=f"only_{head}")
        report = run_experiment(
            f"only_{head}", training_sessions, validation_sessions, tensor_cache,
            replace(settings, loss_setup=head), output_directory, device,
            snapshot_sha256=snapshot_sha256,
            allow_legacy_migration=allow_legacy_migration,
        )
        experiments[report["name"]] = report

    summary = {
        "selected": selected["name"], "settings": selected["settings"],
        "selection_margin_mean_f1": SELECTION_MARGIN_MEAN_F1, "test_evaluated": False,
        "stages": stages, "completed_experiments": list(experiments),
        "automatic_loss_reweighting": False, "automatic_class_reweighting": False,
    }
    write_json(output_directory / "comparison_ledger.json", summary)
    persist_progress(status="complete", next_step="review_tuning_results")
    return summary


def _manifest_dates(manifest, field, expected_count):
    """Validate one chronological date role before it is allowed to index the snapshot."""

    recorded_dates = manifest.get(field)
    if not isinstance(recorded_dates, list) or len(recorded_dates) != expected_count:
        raise ValueError(
            f"Tuning manifest must contain exactly {expected_count} {field.replace('_', ' ')}."
        )
    try:
        normalized_dates = [pd.Timestamp(value).date().isoformat() for value in recorded_dates]
    except (TypeError, ValueError) as error:
        raise ValueError(f"Tuning manifest contains invalid {field.replace('_', ' ')}.") from error
    if normalized_dates != recorded_dates:
        raise ValueError(f"Tuning manifest {field.replace('_', ' ')} must use ISO dates.")
    if normalized_dates != sorted(normalized_dates) or len(set(normalized_dates)) != len(
        normalized_dates
    ):
        raise ValueError(
            f"Tuning manifest {field.replace('_', ' ')} must be unique and chronological."
        )
    return normalized_dates


def _load_frozen_tuning_split(snapshot_path, manifest_path):
    """Rebuild exactly 100/20 frozen roles while proving test rows are absent."""

    if not manifest_path.is_file() or not snapshot_path.is_file():
        raise ValueError(
            "Tuning resume requires both manifest.json and tuning_snapshot.parquet."
        )
    manifest = _read_json_mapping(manifest_path, "tuning manifest")
    if (
        manifest.get("protocol") != "expanding_history_v1"
        or manifest.get("run_index") != 1
        or manifest.get("test_evaluated") is not False
    ):
        raise ValueError("Tuning manifest protocol or test policy is incompatible.")

    training_dates = _manifest_dates(manifest, "training_dates", 100)
    validation_dates = _manifest_dates(manifest, "validation_dates", 20)
    reserved_test_dates = _manifest_dates(manifest, "reserved_test_dates", 10)
    all_declared_dates = training_dates + validation_dates + reserved_test_dates
    if len(set(all_declared_dates)) != len(all_declared_dates):
        raise ValueError("Tuning manifest roles must not share session dates.")
    if all_declared_dates != sorted(all_declared_dates):
        raise ValueError("Tuning manifest roles must remain chronological.")

    try:
        frozen = pd.read_parquet(snapshot_path)
    except OSError as error:
        raise ValueError(f"Could not read frozen tuning snapshot: {snapshot_path}") from error
    if "session_date" not in frozen.columns or frozen.empty:
        raise ValueError("Frozen tuning snapshot must contain session_date rows.")
    frozen["session_date"] = pd.to_datetime(
        frozen["session_date"],
        errors="coerce",
    ).dt.date
    if frozen["session_date"].isna().any():
        raise ValueError("Frozen tuning snapshot contains invalid session dates.")

    snapshot_dates = frozen["session_date"].map(str).tolist()
    if snapshot_dates != sorted(snapshot_dates):
        raise ValueError("Frozen tuning snapshot sessions must remain chronological.")
    unique_snapshot_dates = list(dict.fromkeys(snapshot_dates))
    expected_snapshot_dates = training_dates + validation_dates
    reserved_intersection = set(unique_snapshot_dates).intersection(reserved_test_dates)
    if reserved_intersection:
        raise ValueError("Frozen tuning snapshot must not contain reserved test sessions.")
    if unique_snapshot_dates != expected_snapshot_dates:
        raise ValueError(
            "Frozen tuning snapshot must contain exactly the manifest's 100 training "
            "and 20 validation sessions."
        )

    grouped_sessions = {
        str(session_date): session.reset_index(drop=True)
        for session_date, session in frozen.groupby("session_date", sort=True)
    }
    split = AnnotatedSessionSplit(
        training=tuple(grouped_sessions[day] for day in training_dates),
        validation=tuple(grouped_sessions[day] for day in validation_dates),
        test=(),
    )
    semantic_sha256 = _semantic_snapshot_sha256(
        (*split.training, *split.validation)
    )
    if manifest.get("training_validation_snapshot_sha256") != semantic_sha256:
        raise ValueError("Frozen tuning snapshot does not match its semantic hash.")
    return split, manifest, semantic_sha256


def _ensure_snapshot_identity(
    output_directory,
    manifest,
    snapshot_path,
    semantic_sha256,
    *,
    allow_creation,
):
    """Bind the frozen Parquet bytes and declared roles before experiments may reuse it."""

    identity_path = output_directory / SNAPSHOT_IDENTITY_FILENAME
    expected_identity = {
        "format_version": 1,
        "protocol": manifest["protocol"],
        "training_dates": manifest["training_dates"],
        "validation_dates": manifest["validation_dates"],
        "reserved_test_dates": manifest["reserved_test_dates"],
        "snapshot_semantic_sha256": semantic_sha256,
        "snapshot_file_sha256": _file_sha256(snapshot_path),
        "manifest_sha256": _json_sha256(manifest),
    }
    if identity_path.exists():
        recorded_identity = _read_json_mapping(identity_path, "tuning snapshot identity")
        if recorded_identity != expected_identity:
            raise ValueError("Frozen tuning snapshot identity does not match its artifacts.")
    elif allow_creation:
        write_json(identity_path, expected_identity)
    else:
        raise ValueError("Frozen tuning snapshot is missing its identity sidecar.")
    return expected_identity


def _migrate_legacy_experiment_identities(
    output_directory,
    split,
    device,
    snapshot_sha256,
    *,
    timestamp_column="ts_event",
):
    """Validate every legacy completed result before adding its missing identity sidecar."""

    for report_path in sorted(output_directory.glob("*/report.json")):
        directory = report_path.parent
        report = _read_json_mapping(report_path, "legacy experiment report")
        try:
            settings = TuningSettings(**report["settings"])
            training_session_count = int(report["training_session_count"])
            validation_session_count = int(report["validation_session_count"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"Legacy completed experiment is malformed: {directory.name}"
            ) from error

        if (
            training_session_count == len(split.training)
            and validation_session_count == len(split.validation)
        ):
            training_sessions = split.training
            validation_sessions = split.validation
        elif training_session_count == 5 and validation_session_count == 0:
            training_sessions = split.training[:5]
            validation_sessions = ()
        else:
            raise ValueError(
                "Legacy experiment partition counts do not match the frozen tuning pass: "
                f"{directory.name}"
            )

        _validate_completed_experiment(
            directory.name,
            training_sessions,
            validation_sessions,
            settings,
            directory,
            device,
            snapshot_sha256=snapshot_sha256,
            allow_legacy_migration=True,
            timestamp_column=timestamp_column,
        )


def main(arguments=None) -> int:
    """Freeze the first 100/20/10 plan; optionally continue a validation-only tuning checklist."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--normalized", type=Path, required=True)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--full-checklist", action="store_true")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "mps", "cuda"))
    arguments = parser.parse_args(arguments)
    configuration = load_config(arguments.config)
    device = resolve_training_device(arguments.device)

    if arguments.resume:
        if not arguments.output_directory.is_dir():
            raise ValueError("Tuning resume requires an existing output directory.")
    else:
        arguments.output_directory.mkdir(parents=True, exist_ok=False)
    snapshot_path = arguments.output_directory / "tuning_snapshot.parquet"
    manifest_path = arguments.output_directory / "manifest.json"
    if arguments.resume:
        # Resumption uses the frozen snapshot, not whatever the annotation GUI now contains.
        # New or corrected labels belong to a new pass and cannot contaminate this comparison.
        split, manifest, snapshot_sha256 = _load_frozen_tuning_split(
            snapshot_path,
            manifest_path,
        )
        _ensure_snapshot_identity(
            arguments.output_directory,
            manifest,
            snapshot_path,
            snapshot_sha256,
            allow_creation=True,
        )
    else:
        # A read-only SQLite connection takes one snapshot; annotating more sessions during this
        # pass cannot change labels, split membership, or candidate comparisons halfway through.
        database_uri = arguments.database.resolve().as_uri() + "?mode=ro"
        with closing(sqlite3.connect(database_uri, uri=True)) as store:
            annotations = [
                CandlestickAnnotation(identifier, MarketRegime(current), MarketRegime(anticipated))
                for identifier, current, anticipated in store.execute(
                    "SELECT candlestick_id, current_regime, anticipated_regime FROM annotations"
                )
            ]
        sessions = build_complete_annotated_sessions(
            pd.read_parquet(arguments.normalized), annotations,
            timestamp_column=configuration.data.timestamp_column,
            interval=configuration.data.target_interval,
            session_timezone=configuration.data.session_timezone,
        )
        plan = plan_walk_forward_runs(sessions)
        if not plan.runs:
            raise ValueError("The unchanged protocol requires 130 complete annotated sessions.")
        run = plan.runs[0]
        split = run.select_sessions(sessions)

        # Only these 120 sessions are published. Test dates remain audit metadata and their
        # prices or labels never enter the tuning snapshot or diagnostic tensor cache.
        tuning_sessions = (*split.training, *split.validation)
        snapshot_sha256 = _semantic_snapshot_sha256(tuning_sessions)
        manifest = {
            "protocol": "expanding_history_v1", "run_index": 1,
            "training_dates": [
                str(session["session_date"].iloc[0]) for session in split.training
            ],
            "validation_dates": [
                str(session["session_date"].iloc[0]) for session in split.validation
            ],
            "reserved_test_dates": [
                str(session["session_date"].iloc[0]) for session in split.test
            ],
            "training_validation_snapshot_sha256": snapshot_sha256,
            "test_evaluated": False,
            "selection_rule": (
                "Mean validation macro-F1; a 0.005 practical margin prefers simpler choices."
            ),
        }
        write_json(manifest_path, manifest)
        temporary_snapshot_path = snapshot_path.with_suffix(".partial.parquet")
        try:
            pd.concat(tuning_sessions, ignore_index=True).to_parquet(
                temporary_snapshot_path,
                index=False,
            )
            temporary_snapshot_path.replace(snapshot_path)
        finally:
            temporary_snapshot_path.unlink(missing_ok=True)
        _ensure_snapshot_identity(
            arguments.output_directory,
            manifest,
            snapshot_path,
            snapshot_sha256,
            allow_creation=True,
        )

    tuning_sessions = (*split.training, *split.validation)
    if arguments.resume:
        # Finish the one-time migration before any new candidate starts. This leaves every
        # historical completed report independently bound to the same frozen snapshot.
        _migrate_legacy_experiment_identities(
            arguments.output_directory,
            split,
            device,
            snapshot_sha256,
            timestamp_column=configuration.data.timestamp_column,
        )
    tensor_cache = {id(session): convert_session_to_tensors(
        session, timestamp_column=configuration.data.timestamp_column
    ) for session in tuning_sessions}
    settings = TuningSettings(random_seed=configuration.project.random_seed)
    baseline = run_experiment(
        "baseline", split.training, split.validation, tensor_cache, settings,
        arguments.output_directory, device,
        snapshot_sha256=snapshot_sha256,
        allow_legacy_migration=arguments.resume,
    )
    tiny = run_experiment(
        "tiny_5_sessions", split.training[:5], (), tensor_cache,
        replace(settings, batch_size=5, epochs=1000), arguments.output_directory, device,
        snapshot_sha256=snapshot_sha256,
        tiny=True,
        allow_legacy_migration=arguments.resume,
    )
    write_json(arguments.output_directory / "stage_summary.json", {
        "status": "ready" if tiny["tiny_passed"] else "halted",
        "baseline": baseline["name"],
        "tiny_passed": tiny["tiny_passed"],
        "selected": baseline["name"],
        "completed_stage_count": 0,
        "last_completed_stage": None,
        "current_stage": None,
        "next_step": "learning_rate_sweep" if tiny["tiny_passed"] else "investigate_tiny_fit",
        "test_evaluated": False,
    })
    if arguments.full_checklist:
        if not tiny["tiny_passed"]:
            print("Tiny-set check failed; investigate optimization before broader tuning.")
            return 0
        run_checklist(
            baseline,
            split.training,
            split.validation,
            tensor_cache,
            arguments.output_directory,
            device,
            snapshot_sha256=snapshot_sha256,
            allow_legacy_migration=arguments.resume,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
