"""Run recorded training/validation diagnostics without evaluating held-out test sessions."""

import argparse
from copy import deepcopy
from contextlib import closing
from dataclasses import asdict, dataclass, replace
import hashlib
import json
from pathlib import Path
import sqlite3
from time import perf_counter

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


def run_experiment(
    name, training_sessions, validation_sessions, tensor_cache, settings,
    output_directory, device, *, tiny=False,
) -> dict:
    """Fit fresh weights on the declared training subset and select only by validation loss.

    Tiny-set diagnostics have no validation/test access and stop at 98% training accuracy
    on both heads, or their declared epoch limit. They cannot select a generalization model.
    """

    directory = output_directory / name
    if (directory / "report.json").is_file():
        existing_report = json.loads((directory / "report.json").read_text())
        if (
            existing_report["settings"] != asdict(settings)
            or existing_report["device"] != str(device)
            or existing_report["torch_version"] != str(torch.__version__)
        ):
            raise ValueError(f"Existing experiment has different settings or environment: {name}")
        print(f"Reusing completed diagnostic: {name}", flush=True)
        return existing_report
    directory.mkdir()
    write_json(directory / "settings.json", asdict(settings))
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
    torch.save({
        "model_state": {key: value.cpu() for key, value in best_state.items()},
        "model_config": asdict(configuration), "settings": asdict(settings),
        "feature_mean": standardizer.mean.cpu(),
        "feature_standard_deviation": standardizer.standard_deviation.cpu(),
    }, directory / "diagnostic_weights.pt")
    write_json(directory / "report.json", report)
    print(f"{name}: done, best epoch {best_epoch}, {report['elapsed_seconds']:.1f}s", flush=True)
    return report


def validation_score(report: dict) -> float:
    """Rank two-head candidates without consulting any test measurement."""

    return sum(report["validation"][head]["macro_f1"]
               for head in ("current", "anticipated")) / 2


def run_checklist(baseline, training_sessions, validation_sessions, tensor_cache,
                  output_directory, device) -> dict:
    """Compare one factor at a time, retaining the simpler model for negligible gains.

    The 0.005 mean-F1 margin is declared before sweeps. Single-head ablations are
    diagnostic only and cannot replace the selected two-head candidate implicitly.
    """

    selected = baseline
    experiments = {baseline["name"]: baseline}
    stages = []

    def compare(stage, reference, candidates):
        nonlocal selected
        stage_reports = [reference]
        for name, candidate_settings in candidates:
            existing = next((report for report in experiments.values()
                             if report["settings"] == asdict(candidate_settings)), None)
            report = existing or run_experiment(
                name, training_sessions, validation_sessions, tensor_cache,
                candidate_settings, output_directory, device,
            )
            experiments[report["name"]] = report
            stage_reports.append(report)
            write_json(output_directory / "comparison_ledger.json", {
                "selected": selected["name"], "stages": stages,
                "completed_experiments": list(experiments), "active_stage": stage,
            })

        winner = max(stage_reports, key=lambda report: (
            validation_score(report), -report["validation_loss"]
        ))
        reference_score = validation_score(reference)

        # Small validation fluctuations are not enough reason to increase complexity.
        # A smaller model within the declared margin is preferred as a comparable result.
        if validation_score(winner) < reference_score + 0.005:
            winner = reference
        comparable = [report for report in stage_reports
                      if validation_score(report) >= validation_score(winner) - 0.005]
        winner = min(comparable, key=lambda report: (report["parameter_count"],
                                                    -validation_score(report)))
        stages.append({"stage": stage, "reference": reference["name"],
                       "candidates": [report["name"] for report in stage_reports],
                       "selected": winner["name"]})
        selected = winner
        write_json(output_directory / "comparison_ledger.json", {
            "selected": selected["name"], "stages": stages,
            "completed_experiments": list(experiments), "active_stage": None,
        })
        print(f"Selected after {stage}: {selected['name']} "
              f"(mean validation F1 {validation_score(selected):.4f})", flush=True)
        return winner

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
    control = run_experiment(
        "width_control_four_heads", training_sessions, validation_sessions, tensor_cache,
        replace(settings, attention_head_count=4), output_directory, device,
    )
    experiments[control["name"]] = control
    before_width = selected
    width_winner = compare("model_width_matched_heads", control, [
        (f"width_{width}", replace(settings, attention_head_count=4, model_dimension=width))
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
        report = run_experiment(
            f"only_{head}", training_sessions, validation_sessions, tensor_cache,
            replace(settings, loss_setup=head), output_directory, device,
        )
        experiments[report["name"]] = report

    summary = {
        "selected": selected["name"], "settings": selected["settings"],
        "selection_margin_mean_f1": 0.005, "test_evaluated": False,
        "stages": stages, "completed_experiments": list(experiments),
        "automatic_loss_reweighting": False, "automatic_class_reweighting": False,
    }
    write_json(output_directory / "comparison_ledger.json", summary)
    return summary


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

    arguments.output_directory.mkdir(parents=True, exist_ok=arguments.resume)
    snapshot_path = arguments.output_directory / "tuning_snapshot.parquet"
    manifest_path = arguments.output_directory / "manifest.json"
    if arguments.resume and snapshot_path.is_file():
        # Resumption uses the frozen snapshot, not whatever the annotation GUI now contains.
        # New or corrected labels belong to a new pass and cannot contaminate this comparison.
        previous_manifest = json.loads(manifest_path.read_text())
        frozen = pd.read_parquet(snapshot_path)
        frozen["session_date"] = pd.to_datetime(frozen["session_date"]).dt.date
        grouped_sessions = {
            str(session_date): session.reset_index(drop=True)
            for session_date, session in frozen.groupby("session_date", sort=True)
        }
        split = AnnotatedSessionSplit(
            training=tuple(grouped_sessions[day] for day in previous_manifest["training_dates"]),
            validation=tuple(grouped_sessions[day]
                             for day in previous_manifest["validation_dates"]),
            test=(),
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

    # Only the 100 training and 20 validation sessions become tensors. The ten test sessions
    # are named in the manifest for auditing but never passed to any diagnostic evaluator.
    tuning_sessions = (*split.training, *split.validation)
    tensor_cache = {id(session): convert_session_to_tensors(
        session, timestamp_column=configuration.data.timestamp_column
    ) for session in tuning_sessions}
    snapshot = hashlib.sha256()
    for session in tuning_sessions:
        snapshot.update(pd.util.hash_pandas_object(session, index=False).to_numpy().tobytes())
    manifest = {
        "protocol": "expanding_history_v1", "run_index": 1,
        "training_dates": [str(session["session_date"].iloc[0]) for session in split.training],
        "validation_dates": [
            str(session["session_date"].iloc[0]) for session in split.validation
        ],
        "reserved_test_dates": [str(session["session_date"].iloc[0]) for session in split.test],
        "training_validation_snapshot_sha256": snapshot.hexdigest(),
        "test_evaluated": False,
        "selection_rule": "Mean validation macro-F1 across both heads; lower loss breaks ties.",
    }
    manifest_path = arguments.output_directory / "manifest.json"
    if manifest_path.exists():
        previous_manifest = json.loads(manifest_path.read_text())
        if previous_manifest["training_validation_snapshot_sha256"] != snapshot.hexdigest():
            raise ValueError("Tuning input snapshot changed; start a new pass directory.")
    else:
        write_json(manifest_path, manifest)
    if not snapshot_path.exists():
        pd.concat(tuning_sessions, ignore_index=True).to_parquet(snapshot_path, index=False)
    settings = TuningSettings(random_seed=configuration.project.random_seed)
    baseline = run_experiment(
        "baseline", split.training, split.validation, tensor_cache, settings,
        arguments.output_directory, device,
    )
    tiny = run_experiment(
        "tiny_5_sessions", split.training[:5], (), tensor_cache,
        replace(settings, batch_size=5, epochs=1000), arguments.output_directory, device, tiny=True,
    )
    write_json(arguments.output_directory / "stage_summary.json", {
        "baseline": baseline["name"], "tiny_passed": tiny["tiny_passed"],
        "next_step": "learning_rate_sweep" if tiny["tiny_passed"] else "investigate_tiny_fit",
    })
    if arguments.full_checklist:
        if not tiny["tiny_passed"]:
            print("Tiny-set check failed; investigate optimization before broader tuning.")
            return 0
        run_checklist(baseline, split.training, split.validation, tensor_cache,
                      arguments.output_directory, device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
