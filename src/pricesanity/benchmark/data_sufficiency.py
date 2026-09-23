"""Reproducible development learning curves, isolated from the permanent final benchmark."""

from dataclasses import asdict, replace
from pathlib import Path
from time import perf_counter
import json

import numpy as np
import pandas as pd

from pricesanity.benchmark.artifacts import (
    _publish_atomic,
    _run_lock,
    _write_json_atomic,
    build_benchmark_run_identity,
    canonical_sha256,
    checkpoint_fitted_model,
    file_sha256,
    load_benchmark_run,
    prepare_benchmark_run,
    resume_environment_fingerprint,
    save_benchmark_run,
)
from pricesanity.benchmark.learning_curve import plan_learning_curve
from pricesanity.benchmark.registry import build_model, get_model_family, load_model
from pricesanity.benchmark.runner import (
    build_benchmark_prediction_frame,
    evaluate_fitted_model,
    run_model_once,
)
from pricesanity.benchmark.snapshot import (
    freeze_benchmark_snapshot_from_sessions,
    load_benchmark_snapshot,
    load_snapshot_sessions,
)
from pricesanity.benchmark.sufficiency_analysis import (
    SCORE_NAMES,
    assess_curve,
    paired_curve_difference,
    summarize_curve_point,
)
from pricesanity.benchmark.sufficiency_data import load_development_sessions
from pricesanity.features import FEATURE_COLUMNS
from pricesanity.features.representations import fit_unique_candle_standardizer
from pricesanity.features.specification import (
    EvaluationUniverse,
    RepresentationSpec,
    assert_same_evaluation_universe,
    build_representation_corpus,
)


def plan_data_sufficiency(
    session_count: int,
    train_sizes: tuple[int, ...],
    *,
    development_limit: int,
    allow_small_evaluation: bool = False,
):
    """Use nested chronological prefixes and one strictly later whole-session block."""

    if session_count > development_limit:
        raise ValueError("Data-sufficiency population exceeds the development boundary.")
    if train_sizes and session_count <= train_sizes[-1]:
        raise ValueError(
            f"Training size {train_sizes[-1]} leaves no later evaluation sessions among "
            f"{session_count} complete sessions; annotate more or reduce --train-sizes."
        )
    points = plan_learning_curve(
        train_sizes, development_session_count=session_count,
        evaluation_range=(max(train_sizes, default=0), session_count),
    )
    evaluation_count = session_count - train_sizes[-1]
    if evaluation_count < 2:
        raise ValueError("At least two later evaluation sessions are required.")
    if evaluation_count < 20 and not allow_small_evaluation:
        raise ValueError(
            f"Only {evaluation_count} later evaluation sessions remain; need at least 20. "
            "Annotate more sessions, reduce --train-sizes, or use --allow-small-evaluation."
        )
    return points


def run_data_sufficiency(
    *,
    normalized_path: Path,
    candlestick_path: Path,
    database_path: Path,
    app_config,
    benchmark_config,
    output_directory: Path,
    model_names: tuple[str, ...],
    train_sizes: tuple[int, ...] = (100, 200, 300, 400, 500),
    seeds: tuple[int, ...] | None = None,
    device: str = "cpu",
    fixed_config: Path | None = None,
    benchmark_snapshot: Path | None = None,
    allow_small_evaluation: bool = False,
    meaningful_gain: float = 0.01,
    small_gain: float = 0.005,
    bootstrap_repetitions: int = 2000,
    resume: bool = False,
) -> dict:
    """Freeze development evidence once, resume exact runs, and report paired marginal gains.

    This command never tunes or selects epochs using evaluation data. Neural adapters use
    fixed training budgets. Resumption keeps the frozen annotations even if the live GUI has
    acquired more labels; including those labels requires a new diagnostic directory.
    """

    started = perf_counter()
    seeds = tuple(benchmark_config.final_seeds if seeds is None else seeds)
    if not seeds or len(set(seeds)) != len(seeds) or any(seed < 0 for seed in seeds):
        raise ValueError("Seeds must be distinct nonnegative integers.")
    if not model_names or len(set(model_names)) != len(model_names):
        raise ValueError("Choose distinct registered model names.")
    if bootstrap_repetitions <= 0:
        raise ValueError("Bootstrap repetitions must be positive.")
    assess_curve(
        [], evaluation_sessions=0, stochastic=False, seed_count=1, latest_seed_sd=None,
        meaningful_gain=meaningful_gain, small_gain=small_gain,
    )
    if benchmark_config.window_length != 16 or tuple(benchmark_config.feature_columns) != (
        FEATURE_COLUMNS
    ) or benchmark_config.controlled_first_scored_candle_position != 15:
        raise ValueError("This diagnostic requires the existing controlled 16 x 4 contract.")
    plan_learning_curve(
        train_sizes, development_session_count=benchmark_config.development_session_count,
        evaluation_range=(max(train_sizes, default=0), benchmark_config.development_session_count),
    )

    parameters = {name: {} for name in model_names}
    if fixed_config is not None:
        supplied = json.loads(fixed_config.read_text())
        if not isinstance(supplied, dict) or set(supplied) != set(model_names):
            raise ValueError(
                "Fixed-config JSON must map exactly the selected model names to settings."
            )
        parameters = supplied
    elif "transformer" in parameters:
        parameters["transformer"] = dict(benchmark_config.transformer_incumbent)
    for name in model_names:
        if not isinstance(parameters[name], dict):
            raise ValueError("Each fixed model configuration must be a JSON object.")
        if parameters[name].get("context_length", 16) != 16:
            raise ValueError("Controlled data sufficiency cannot change the 16-candle context.")
        # Validate through the real factory before reserving output or opening source data.
        build_model(name, random_seed=seeds[0], parameters=parameters[name], device=device)

    request = {
        "kind": "development_data_sufficiency_v1",
        "model_names": model_names, "train_sizes": train_sizes, "seeds": seeds,
        "device": device, "parameters": parameters,
        "configuration_origin": (
            {"path": str(fixed_config.resolve()), "sha256": file_sha256(fixed_config)}
            if fixed_config is not None else "registry defaults / benchmark Transformer incumbent"
        ),
        "benchmark_config": asdict(benchmark_config), "project_config": asdict(app_config),
        "sources": [str(path.resolve()) for path in (
            normalized_path, candlestick_path, database_path,
        )],
        "benchmark_snapshot": str(benchmark_snapshot.resolve()) if benchmark_snapshot else None,
        "allow_small_evaluation": allow_small_evaluation,
        "meaningful_gain": meaningful_gain, "small_gain": small_gain,
        "bootstrap_repetitions": bootstrap_repetitions,
    }
    output_directory = Path(output_directory)
    # A development diagnostic cannot masquerade as an official study or its snapshot.
    if output_directory.resolve().is_relative_to(benchmark_config.output_root.resolve()):
        raise ValueError("Use a data-sufficiency directory outside the official benchmark root.")
    manifest_path = output_directory / "manifest.json"
    existed = output_directory.exists()
    if existed and not resume:
        raise FileExistsError("Output already exists; use --resume or a new diagnostic directory.")
    # Reject inadequate source populations before reserving output. After more annotation,
    # the same command can then initialize normally instead of inheriting an empty failed run.
    if not existed:
        sessions, sources = load_development_sessions(
            normalized_path, candlestick_path, database_path,
            app_config=app_config, benchmark_config=benchmark_config,
            benchmark_snapshot=benchmark_snapshot,
        )
        plan_data_sufficiency(
            len(sessions), train_sizes,
            development_limit=benchmark_config.development_session_count,
            allow_small_evaluation=allow_small_evaluation,
        )
    environment = resume_environment_fingerprint(device=device)
    output_directory.mkdir(parents=True, exist_ok=True)
    with _run_lock(output_directory / ".diagnostic.lock"):
        if not existed and any(
            path.name != ".diagnostic.lock" for path in output_directory.iterdir()
        ):
            raise FileExistsError("Another writer initialized this diagnostic directory.")
        if existed:
            if not manifest_path.is_file():
                raise ValueError("Initialization is incomplete; use a new diagnostic directory.")
            manifest = json.loads(manifest_path.read_text())
            if manifest.get("request_sha256") != canonical_sha256(request):
                raise ValueError("Resume request differs from the frozen diagnostic identity.")
            if manifest.get("execution_environment") != environment:
                raise ValueError(
                    "Resume requires the same source, libraries, and device environment."
                )
            if canonical_sha256(manifest["request"]) != manifest["request_sha256"]:
                raise ValueError("Stored diagnostic request identity is inconsistent.")
            snapshot = load_benchmark_snapshot(output_directory / "snapshot")
            if snapshot.identity_sha256 != manifest["snapshot_sha256"]:
                raise ValueError("Frozen diagnostic snapshot identity changed.")
            print("Resuming frozen DEVELOPMENT ONLY evidence; live annotations are not reread.")
        else:
            snapshot = freeze_benchmark_snapshot_from_sessions(
                sessions, output_directory=output_directory / "snapshot",
                timestamp_column=app_config.data.timestamp_column,
                source_identities=sources,
            )
            manifest = {
                "request": request, "request_sha256": canonical_sha256(request),
                "snapshot_sha256": snapshot.identity_sha256,
                "execution_environment": environment,
            }
            _write_json_atomic(manifest_path, manifest, exclusive=True)
        sessions = load_snapshot_sessions(snapshot)
        points = plan_data_sufficiency(
            len(sessions), train_sizes,
            development_limit=benchmark_config.development_session_count,
            allow_small_evaluation=allow_small_evaluation,
        )
        evaluation_count = len(sessions) - train_sizes[-1]
        print(
            f"DEVELOPMENT ONLY: {len(sessions)} complete sessions; "
            f"{evaluation_count} fixed later sessions."
        )
        if evaluation_count < 50:
            print("PRELIMINARY: fewer than 50 evaluation sessions; uncertainty may be wide.")

        # Build causal windows once. Scaling a window's four values is identical to scaling
        # unique candles first, provided the scaler itself is fitted before window overlap.
        representation = RepresentationSpec(
            "controlled_16", 16, "sequential", "none", None, "none", FEATURE_COLUMNS,
        )
        corpus = build_representation_corpus(
            sessions, representation, EvaluationUniverse(15),
        )
        evaluation_raw = corpus.select_sessions(range(train_sizes[-1], len(sessions)))
        session_ids = [str(session.session_date.iloc[0]) for session in sessions]
        population = {
            "session_ids": session_ids,
            "training_session_ids": {str(size): session_ids[:size] for size in train_sizes},
            "evaluation_session_ids": session_ids[train_sizes[-1]:],
            "evaluation_candlestick_ids": evaluation_raw.candlestick_ids,
            "training_candle_ids": {
                str(size): [str(value) for session in sessions[:size]
                            for value in session.candlestick_id]
                for size in train_sizes
            },
            "representation": representation.to_dict(),
            "first_scored_candle_position": 15,
            "standardization": "fit four statistics on unique training candles only",
        }
        _write_json_atomic(output_directory / "population.json", population)
        run_records = []
        curves = {name: [] for name in model_names}
        gains = {name: [] for name in model_names}
        previous_predictions = {}
        for point in points:
            size = point.training_session_count
            standardizer = fit_unique_candle_standardizer(
                sessions[:size], feature_columns=FEATURE_COLUMNS,
            )
            training_raw = corpus.select_sessions(range(size))

            def scaled(raw):
                return replace(raw, features=standardizer.transform(
                    raw.features.reshape(-1, len(FEATURE_COLUMNS)),
                ).reshape(raw.features.shape))

            training, evaluation = scaled(training_raw), scaled(evaluation_raw)
            assert_same_evaluation_universe(evaluation_raw, evaluation)
            preprocessing = {
                "mean": standardizer.mean.tolist(), "scale": standardizer.scale.tolist(),
                "training_session_ids": session_ids[:size],
            }
            _write_json_atomic(output_directory / f"standardizer_{size}.json", preprocessing)
            for name in model_names:
                family = get_model_family(name)
                run_seeds = seeds if family.stochastic else seeds[:1]
                predictions_for_size = []
                for seed in run_seeds:
                    predictions, record = _execute_point(
                        name=name, seed=seed, parameters=parameters[name],
                        training=training, evaluation=evaluation,
                        output_directory=output_directory, snapshot=snapshot,
                        session_ids=session_ids, size=size, largest_size=train_sizes[-1],
                        request=request, environment=environment,
                        preprocessing=preprocessing, benchmark_config=benchmark_config,
                    )
                    predictions_for_size.append(predictions)
                    run_records.append(record)
                summary = summarize_curve_point(predictions_for_size)
                curves[name].append({"model": name, "training_sessions": size, **summary})
                if name in previous_predictions:
                    previous_size, earlier = previous_predictions[name]
                    difference = paired_curve_difference(
                        earlier, predictions_for_size,
                        repetitions=bootstrap_repetitions, random_seed=seeds[0],
                    )
                    difference.update({"from_sessions": previous_size, "to_sessions": size})
                    difference["per_100_sessions"] = {
                        metric: {key: value * 100 / (size - previous_size)
                                 for key, value in values.items()
                                 if key in {"gain", "lower", "upper"}}
                        for metric, values in difference["scores"].items()
                    }
                    gains[name].append(difference)
                # Retain only neighboring curve points for pairing, not every model's history.
                previous_predictions[name] = (size, predictions_for_size)
        assessments = {
            name: assess_curve(
                gains[name], evaluation_sessions=evaluation_count,
                stochastic=get_model_family(name).stochastic,
                seed_count=curves[name][-1]["seed_count"],
                latest_seed_sd=max(
                    curves[name][-1][metric + "_seed_sd"] or 0 for metric in SCORE_NAMES
                ),
                meaningful_gain=meaningful_gain, small_gain=small_gain,
                reference_only=name in {"majority_class", "previous_regime"},
            ) for name in model_names
        }
        summary = {
            "kind": request["kind"], "request_sha256": canonical_sha256(request),
            "snapshot_sha256": snapshot.identity_sha256,
            "evaluation_session_count": evaluation_count,
            "curves": curves, "marginal_gains": gains, "assessments": assessments,
            "runs": run_records, "execution_environment": environment,
            "invocation_seconds": perf_counter() - started,
        }
        # Small derived reports are rebuildable from immutable, checksummed seed artifacts.
        curve_path = output_directory / "curve.csv"
        temporary_curve = output_directory / ".curve.csv.partial"
        pd.DataFrame([row for rows in curves.values() for row in rows]).to_csv(
            temporary_curve, index=False,
        )
        _publish_atomic(temporary_curve, curve_path)
        _write_json_atomic(output_directory / "marginal_gains.json", gains)
        _write_json_atomic(output_directory / "assessment.json", assessments)
        _write_json_atomic(output_directory / "summary.json", summary)
        print_summary(summary)
        print(f"Artifacts: {output_directory}")
        return summary


def _execute_point(
    *, name, seed, parameters, training, evaluation, output_directory, snapshot,
    session_ids, size, largest_size, request, environment, preprocessing, benchmark_config,
):
    """Reuse the benchmark's exact checkpoint lineage and completion validation."""

    family = get_model_family(name)
    model = build_model(
        name, random_seed=seed, parameters=parameters, device=request["device"],
        cpu_worker_count=benchmark_config.cpu_worker_count, prestandardized=True,
    )
    configuration = {"parameters": parameters, "resolved_adapter": model.describe()}
    identity = build_benchmark_run_identity(
        track="controlled", model_name=name, run_name=f"development_train_{size}_seed_{seed}",
        seed=seed, model_configuration=configuration,
        representation={"name": "controlled_16", "preprocessing": preprocessing},
        feature_columns=FEATURE_COLUMNS, window_length=16,
        training_session_ids=session_ids[:size],
        validation_session_ids=session_ids[largest_size:], test_session_ids=[],
        annotation_snapshot_identity=snapshot.identity_sha256,
        normalized_dataset_identity=snapshot.source_identities,
        protocol_configuration=request, search_space={"tuning": False},
        label_mapping={"bull": 0, "bear": 1, "range": 2},
        training_session_range=(0, size),
        validation_session_ranges=((largest_size, len(session_ids)),), test_session_range=None,
    )
    paths, state = prepare_benchmark_run(
        output_directory / "runs", identity, resume=True, execution_environment=environment,
    )
    print(f"{name} / train {size} / seed {seed}: {state}", flush=True)
    if state != "complete":
        progress = json.loads(paths.progress.read_text()) if paths.progress.exists() else {}
        completed = progress.get("completed", {})
        if "model" in completed:
            if file_sha256(paths.model) != completed["model"]:
                raise ValueError("Interrupted diagnostic model checksum mismatch.")
            model = load_model(name, str(paths.model), device=request["device"])
            if model.state_fingerprint() != completed["model_state_sha256"]:
                raise ValueError("Interrupted diagnostic fitted-model identity mismatch.")
            timing = progress["training_evidence"]
            result = evaluate_fitted_model(
                model, evaluation=evaluation, representation=family.representation,
                training_seconds=timing["training_seconds"],
                training_sample_count=timing["training_sample_count"],
                device=request["device"], cpu_worker_count=benchmark_config.cpu_worker_count,
                inference_timing_repetitions=1,
            )
        else:
            result = run_model_once(
                model, training=training, evaluation=evaluation,
                representation=family.representation, device=request["device"],
                cpu_worker_count=benchmark_config.cpu_worker_count,
                inference_timing_repetitions=1,
                checkpoint_fitted=lambda fitted, timing: checkpoint_fitted_model(
                    paths, fitted, timing,
                ),
            )
        predictions = build_benchmark_prediction_frame(evaluation, result)
        save_benchmark_run(
            paths, identity, model=model, predictions=predictions, metrics=result.metrics,
            model_configuration=configuration,
            dataset_description={
                "purpose": "DEVELOPMENT ONLY; not final benchmark evidence",
                "snapshot_sha256": snapshot.identity_sha256,
                "training_session_ids": session_ids[:size],
                "evaluation_session_ids": session_ids[largest_size:],
                "preprocessing": preprocessing,
            },
        )
    metadata, predictions, metrics = load_benchmark_run(paths.directory)
    expected = pd.DataFrame({
        "candlestick_id": evaluation.candlestick_ids,
        "session_index": evaluation.session_indices,
        "human_current_regime": np.array(["bull", "bear", "range"])[evaluation.current_targets],
        "human_anticipated_regime": np.array(["bull", "bear", "range"])[
            evaluation.anticipated_targets
        ],
        "session_date": evaluation.session_dates,
        "candle_position": evaluation.candle_positions,
    })
    if not expected.equals(predictions.loc[:, expected.columns]):
        raise ValueError("Saved curve predictions do not match the fixed evaluation identities.")
    if list(pd.to_datetime(predictions.timestamp, utc=True)) != list(evaluation.timestamps):
        raise ValueError("Saved curve prediction timestamps changed.")
    return predictions, {
        "model": name, "seed": seed, "training_sessions": size,
        "run_directory": str(paths.directory.relative_to(output_directory)),
        "experiment_sha256": identity.experiment_sha256,
        "model_configuration": configuration, "metrics": metrics,
        "artifacts": metadata["artifacts"],
    }


def print_summary(summary: dict) -> None:
    """Print both heads, paired uncertainty, and cautious development interpretations."""

    for name, rows in summary["curves"].items():
        print(f"\nDATA SUFFICIENCY — {name} / controlled / DEVELOPMENT ONLY")
        print("Train sessions    Current F1    Anticipated F1    Mean F1    Mean seed SD")
        for row in rows:
            deviation = row["mean_head_macro_f1_seed_sd"]
            deviation_text = "not estimated" if deviation is None else f"{deviation:.4f}"
            print(f"{row['training_sessions']:14}    {row['current_macro_f1']:.4f}"
                  f"        {row['anticipated_macro_f1']:.4f}            "
                  f"{row['mean_head_macro_f1']:.4f}     {deviation_text}")
        print("Recent gains (observed interval; paired 95% session CI):")
        for gain in summary["marginal_gains"][name][-2:]:
            mean = gain["scores"]["mean_head_macro_f1"]
            print(f"  {gain['from_sessions']} -> {gain['to_sessions']}: {mean['gain']:+.4f} "
                  f"[{mean['lower']:+.4f}, {mean['upper']:+.4f}]")
        assessment = summary["assessments"][name]
        print(f"Assessment: {assessment['label']} — {assessment['reason']}")
        print(assessment["interpretation"])
        if name == "previous_regime":
            print("Previous regime uses prior HUMAN labels; this is a non-deployable reference.")
