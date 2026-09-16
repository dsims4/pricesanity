"""Execute resumable chronological benchmark studies from one frozen snapshot."""

from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any
from functools import wraps
from collections import OrderedDict
from dataclasses import replace
import json
import os
import tempfile

import pandas as pd
import numpy as np

from pricesanity.benchmark.artifacts import (
    build_benchmark_run_identity,
    canonical_sha256,
    prepare_benchmark_run,
    resume_environment_fingerprint,
    save_benchmark_run,
    source_identity,
    _run_lock,
    checkpoint_fitted_model,
    _component_is_complete,
    _metrics_from_prediction_frame,
    load_benchmark_summary,
    publish_checkpointed_run,
)
from pricesanity.benchmark.learning_curve import plan_learning_curve
from pricesanity.benchmark.metrics import EfficiencyMetrics
from pricesanity.benchmark.resources import hardware_fingerprint
from pricesanity.benchmark.preflight import build_scalability_preflight
from pricesanity.benchmark.protocol import (
    BenchmarkConfig,
    BenchmarkTrack,
    ChronologicalValidationFold,
    plan_benchmark,
)
from pricesanity.benchmark.registry import (
    build_model,
    get_model_family,
    load_model,
)
from pricesanity.benchmark.runner import (
    BenchmarkRunResult,
    build_benchmark_prediction_frame,
    evaluate_fitted_model,
    run_model_once,
)
from pricesanity.benchmark.search_spaces import representative_candidate, track_search_space
from pricesanity.benchmark.snapshot import (
    BenchmarkSnapshot,
    load_snapshot_sessions,
    load_sealed_holdout,
)
from pricesanity.features import (
    EvaluationUniverse,
    RepresentationSpec,
    build_representation_corpus,
    fit_unique_candle_standardizer,
    transform_sessions,
)


@dataclass(frozen=True)
class BenchmarkStudyPaths:
    """Persistent tuning and final-output locations for one frozen snapshot."""

    root: Path
    tuning: Path
    selected: Path
    runs: Path
    learning_curves: Path


def study_paths(root: str | Path) -> BenchmarkStudyPaths:
    root = Path(root)
    return BenchmarkStudyPaths(
        root=root,
        tuning=root / "tuning",
        selected=root / "selected",
        runs=root / "runs",
        learning_curves=root / "learning_curves",
    )


def _study_locked(method):
    @wraps(method)
    def locked(self, *args, **kwargs):
        with _run_lock(self.paths.root / ".study.lock"):
            return method(self, *args, **kwargs)
    return locked


class BenchmarkExecutor:
    """Coordinate tuning, frozen final fits, learning curves, and crash recovery."""

    def __init__(
        self,
        *,
        snapshot: BenchmarkSnapshot,
        config: BenchmarkConfig,
        search_spaces: dict[str, dict[str, dict[str, Any]]],
        study_root: str | Path,
        device: str = "cpu",
        declared_tracks: tuple[str, ...] = ("controlled", "best_of_family"),
    ) -> None:
        self.snapshot = snapshot
        self.config = config
        self.search_spaces = search_spaces
        self.paths = study_paths(study_root)
        self.device = device
        if device != "cpu":
            from pricesanity.models.sequence import _resolve_device
            _resolve_device(device)
        self.execution_source = source_identity()
        self._representation_cache = OrderedDict()
        self._representation_cache_bytes = 0
        self.representation_cache_limit_bytes = 512 * 1024**2
        if snapshot.development_session_count != config.development_session_count:
            raise ValueError("Benchmark execution requires a physically isolated snapshot matching the protocol.")
        self.sessions = load_snapshot_sessions(snapshot)
        self.plan = plan_benchmark(snapshot.session_count, config)
        self.paths.tuning.mkdir(parents=True, exist_ok=True)
        self.paths.selected.mkdir(parents=True, exist_ok=True)
        self.paths.runs.mkdir(parents=True, exist_ok=True)
        self.paths.learning_curves.mkdir(parents=True, exist_ok=True)
        self.scope = {
            "models": sorted(config.model_tuning_budgets),
            "tracks": sorted(declared_tracks),
            "snapshot_sha256": snapshot.identity_sha256,
        }
        scope_path = self.paths.root / "study_scope.json"
        with _run_lock(self.paths.root / ".study.lock"):
            if scope_path.exists():
                if _read_optional_json(scope_path) != self.scope:
                    raise ValueError("Declared study scope is immutable; use a new study directory.")
            else:
                _write_json_replace(scope_path, self.scope)

    def _require_source(self) -> None:
        if source_identity() != self.execution_source:
            raise ValueError("Source changed while this executor was active; restart with compatible source.")

    def _require_development(self) -> None:
        self._require_source()
        if (self.paths.root / "development_frozen.json").exists():
            raise PermissionError("Study development is frozen; tuning and diagnostics cannot mutate it.")

    def _space(self, model_name: str, track: BenchmarkTrack) -> dict:
        if model_name not in self.scope["models"] or track.value not in self.scope["tracks"]:
            raise ValueError("Model and track must belong to the declared study scope.")
        return track_search_space(model_name, self.search_spaces.get(model_name, {}), track.value)

    def _selection_identity(self, model_name: str, track: BenchmarkTrack, parameters: dict) -> dict:
        return {
            "snapshot_sha256": self.snapshot.identity_sha256,
            "protocol_sha256": canonical_sha256(asdict(self.config)),
            "search_space_sha256": canonical_sha256(self._space(model_name, track)),
            "model_name": model_name,
            "track": track.value,
            "parameters_sha256": canonical_sha256(parameters),
            "selection_source": source_identity(),
            "representation_sha256": canonical_sha256({
                "context": self.config.window_length if track is BenchmarkTrack.CONTROLLED
                else parameters.get("context_length", self.config.window_length),
                "features": self.config.feature_columns,
                "track": track.value,
                "preprocessing_version": 2,
            }),
        }

    def _freeze_contents(self) -> dict:
        self._require_source()
        selections = {}
        for track in self.scope["tracks"]:
            for model in self.scope["models"]:
                selected = self.load_selected(model, BenchmarkTrack(track))
                selections[f"{track}/{model}"] = {
                    "configuration_sha256": canonical_sha256(selected),
                    "representation_sha256": selected["representation_sha256"],
                    "search_space_sha256": selected["search_space_sha256"],
                }
        return {
            "state": "frozen", "format_version": 1,
            "scope": self.scope,
            "snapshot_sha256": self.snapshot.identity_sha256,
            "protocol_sha256": canonical_sha256(asdict(self.config)),
            "selected": selections, "source": source_identity(),
        }

    @_study_locked
    def freeze_development(self) -> dict:
        # Every declared track freezes together: inspecting one final track must never
        # influence continued selection in the other.
        contents = self._freeze_contents()
        path = self.paths.root / "development_frozen.json"
        if path.exists():
            if _read_optional_json(path) != contents:
                raise ValueError("Frozen study identity changed.")
        else:
            _write_json_replace(path, contents)
        return contents

    @_study_locked
    def tune_model(
        self,
        model_name: str,
        *,
        track: BenchmarkTrack,
        fold_indices: tuple[int, ...] | None = None,
        acknowledge_scaling_risk: bool = False,
    ) -> dict[str, Any]:
        """Tune one family using only configured chronological development folds."""

        self._require_development()
        family = get_model_family(model_name)
        folds = self._selected_folds(fold_indices)
        selected_fold_indices = (
            tuple(range(len(self.plan.chronological_validation_folds)))
            if fold_indices is None
            else fold_indices
        )
        all_development_folds = selected_fold_indices == tuple(
            range(len(self.plan.chronological_validation_folds))
        ) and len(folds) >= 2
        budget = self.config.model_tuning_budgets[model_name]
        space = self._space(model_name, track)
        self._require_scaling_acknowledgement(
            model_name, track, max(1, budget), len(folds), acknowledge_scaling_risk
        )
        if budget == 0:
            parameters: dict[str, Any] = {}
            fold_scores = self._score_candidate(model_name, parameters, track, folds)
            selected = self._selected_record(
                model_name, track, parameters, fold_scores,
                fold_indices=selected_fold_indices,
                all_development_folds=all_development_folds,
            )
            self._write_tuning_result(selected)
            if all_development_folds:
                self._write_selected(selected)
            return selected

        try:
            import optuna
        except ImportError as error:
            raise RuntimeError("Tuning requires the optional benchmark dependencies.") from error

        fold_scope = (
            "all_folds"
            if all_development_folds
            else "folds_" + "_".join(str(index + 1) for index in selected_fold_indices)
        )
        # A focused one-fold diagnostic cannot share an Optuna database with the real
        # all-fold objective: identical parameter values would then carry incomparable scores.
        tuning_directory = self.paths.tuning / track.value / model_name / fold_scope
        tuning_directory.mkdir(parents=True, exist_ok=True)
        storage_path = tuning_directory / "optuna.sqlite3"
        tuning_identity = {
            **self._selection_identity(model_name, track, {}),
            "fold_indices": list(selected_fold_indices),
            "tuning_seed": self.config.tuning_seed,
            "objective": "mean_chronological_fold_mean_head_macro_f1_v1",
            "source": source_identity(),
        }
        identity_path = tuning_directory / "study_identity.json"
        if identity_path.exists():
            if _read_optional_json(identity_path) != tuning_identity:
                raise ValueError("Optuna study identity changed; create a distinct study.")
        elif storage_path.exists():
            raise ValueError("Legacy Optuna study lacks immutable identity; create a distinct study.")
        else:
            _write_json_replace(identity_path, tuning_identity)
        study = optuna.create_study(
            study_name=(
                f"{track.value}_{model_name}_{fold_scope}_"
                f"{self.snapshot.identity_sha256[:12]}"
            ),
            storage=f"sqlite:///{storage_path}",
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=self.config.tuning_seed),
            load_if_exists=True,
        )
        if model_name == "transformer":
            # The known strong small-corpus recipe is guaranteed one trial. A stochastic search
            # should improve on accumulated evidence, not accidentally omit the incumbent.
            study.enqueue_trial(
                dict(self.config.transformer_incumbent),
                skip_if_exists=True,
            )

        terminal_states = {
            optuna.trial.TrialState.COMPLETE,
            optuna.trial.TrialState.FAIL,
            optuna.trial.TrialState.PRUNED,
        }
        terminal_count = sum(trial.state in terminal_states for trial in study.trials)
        waiting_count = sum(
            trial.state is optuna.trial.TrialState.WAITING for trial in study.trials
        )
        remaining_trials = max(0, budget - terminal_count)
        if waiting_count > remaining_trials:
            raise ValueError("Persistent tuning study contains more queued trials than its budget.")

        def objective(trial: Any) -> float:
            parameters = _suggest_parameters(trial, space)
            fold_scores = self._score_candidate(
                model_name, parameters, track, folds
            )
            trial.set_user_attr("fold_mean_head_macro_f1", fold_scores)
            trial.set_user_attr("parameters", parameters)
            return float(sum(fold_scores) / len(fold_scores))

        if remaining_trials:
            study.optimize(objective, n_trials=remaining_trials)
        if not study.best_trials:
            raise RuntimeError("Tuning did not complete a legal candidate.")
        best = study.best_trial
        selected = self._selected_record(
            model_name,
            track,
            dict(best.user_attrs["parameters"]),
            tuple(float(value) for value in best.user_attrs["fold_mean_head_macro_f1"]),
            fold_indices=selected_fold_indices,
            all_development_folds=all_development_folds,
        )
        selected["optuna_trial_number"] = best.number
        selected["tuning_seed"] = self.config.tuning_seed
        self._write_tuning_result(selected)
        if all_development_folds:
            self._write_selected(selected)
        return selected

    @_study_locked
    def run_final(
        self,
        model_name: str,
        *,
        track: BenchmarkTrack,
        confirm_final_holdout: bool,
        resume: bool = True,
    ) -> tuple[Path, ...]:
        """Fit frozen selected settings and explicitly unlock the final holdout once."""

        if not confirm_final_holdout:
            raise PermissionError(
                "Final holdout remains locked; pass the explicit confirmation only after freeze."
            )
        selected = self.load_selected(model_name, track)
        seal_path = self.paths.root / "development_frozen.json"
        if not seal_path.exists():
            raise PermissionError("Final holdout is locked until global development freeze.")
        if _read_optional_json(seal_path) != self._freeze_contents():
            raise ValueError("Frozen study identity changed; final holdout remains locked.")
        parameters = dict(selected["parameters"])
        development = tuple(self.sessions[self.plan.development.start_index:self.plan.development.end_index])
        holdout = load_sealed_holdout(
            self.snapshot, seal_path=seal_path, confirm_final_holdout=confirm_final_holdout,
        )
        representation, training, evaluation, model_representation = self._corpora(
            model_name, parameters, track, development, holdout
        )
        family = get_model_family(model_name)
        seeds = self.config.final_seeds if family.stochastic else (self.config.tuning_seed,)
        output_directories = []
        for seed in seeds:
            output_directories.append(self._execute_persisted_run(
                model_name=model_name,
                parameters=parameters,
                track=track,
                seed=seed,
                run_name=f"final_seed_{seed}",
                training_sessions=development,
                evaluation_sessions=holdout,
                training=training,
                evaluation=evaluation,
                representation=representation,
                model_representation=model_representation,
                output_root=self.paths.runs,
                resume=resume,
            ))
        return tuple(output_directories)

    @_study_locked
    def run_learning_curve(
        self,
        model_name: str,
        *,
        track: BenchmarkTrack,
        resume: bool = True,
    ) -> tuple[Path, ...]:
        """Fit growing development prefixes against one unchanged future block."""

        self._require_development()
        selected = self.load_selected(model_name, track)
        parameters = dict(selected["parameters"])
        points = plan_learning_curve(
            self.config.learning_curve_session_counts,
            development_session_count=self.config.development_session_count,
            evaluation_range=self.config.learning_curve_evaluation_range,
        )
        outputs = []
        for point in points:
            training_sessions = tuple(
                self.sessions[point.training_start_index:point.training_end_index]
            )
            evaluation_sessions = tuple(
                self.sessions[point.evaluation_start_index:point.evaluation_end_index]
            )
            representation, training, evaluation, model_representation = self._corpora(
                model_name, parameters, track, training_sessions, evaluation_sessions
            )
            outputs.append(self._execute_persisted_run(
                model_name=model_name,
                parameters=parameters,
                track=track,
                seed=self.config.tuning_seed,
                run_name=f"train_{point.training_session_count}",
                training_sessions=training_sessions,
                evaluation_sessions=evaluation_sessions,
                training=training,
                evaluation=evaluation,
                representation=representation,
                model_representation=model_representation,
                output_root=self.paths.learning_curves,
                resume=resume,
            ))
        return tuple(outputs)

    @_study_locked
    def pilot(
        self,
        model_name: str,
        *,
        track: BenchmarkTrack,
    ) -> dict[str, Any]:
        """Time one representative candidate on one fold without publishing a result."""

        self._require_development()
        parameters = (
            dict(self.config.transformer_incumbent)
            if model_name == "transformer"
            else representative_candidate(self._space(model_name, track))
        )
        if model_name in {"tcn", "gru", "transformer"}:
            parameters["epochs"] = 1
        fold = self.plan.chronological_validation_folds[0]
        training_sessions, evaluation_sessions = self._fold_sessions(fold)
        representation, training, evaluation, model_representation = self._corpora(
            model_name, parameters, track, training_sessions, evaluation_sessions
        )
        model = self._build_model(
            model_name, parameters, self.config.tuning_seed,
            prestandardized=(track is BenchmarkTrack.CONTROLLED),
        )
        started = perf_counter()
        result = run_model_once(
            model,
            training=training,
            evaluation=evaluation,
            representation=model_representation,
            device=self.device if get_model_family(model_name).representation == "sequential" else "cpu",
            cpu_worker_count=self.config.cpu_worker_count,
            inference_timing_repetitions=self.config.inference_timing_repetitions,
        )
        elapsed = perf_counter() - started
        efficiency = result.metrics.efficiency
        assert efficiency is not None
        with tempfile.TemporaryDirectory(prefix="pricesanity-pilot-") as directory:
            model_path = Path(directory) / "model.bin"
            model.save(model_path)
            serialized_model_bytes = model_path.stat().st_size
        flat_feature_dimension = int(np.prod(training.features.shape[1:]))
        preflight = build_scalability_preflight(
            model_name=model_name,
            training_samples=len(training.current_targets),
            evaluation_samples=len(evaluation.current_targets),
            feature_dimension=flat_feature_dimension,
            candidate_count=max(1, self.config.model_tuning_budgets[model_name]),
            fold_count=len(self.plan.chronological_validation_folds),
            seed_count=(
                len(self.config.final_seeds)
                if get_model_family(model_name).stochastic else 1
            ),
            polynomial_degree=2 if model_name == "polynomial_logistic" else None,
        )
        report = {
            "model_name": model_name,
            "track": track.value,
            "parameters": parameters,
            "wall_seconds": elapsed,
            "training_seconds": efficiency.training_seconds,
            "inference_seconds": efficiency.inference_seconds,
            "training_samples": efficiency.training_sample_count,
            "evaluation_samples": efficiency.inference_sample_count,
            "feature_dimension": flat_feature_dimension,
            "transformed_feature_dimension": preflight.transformed_feature_dimension,
            "parameter_count": efficiency.parameter_count,
            "serialized_model_bytes": serialized_model_bytes,
            "estimated_input_bytes": int(
                training.features.nbytes + evaluation.features.nbytes
                + training.valid_history_mask.nbytes
                + evaluation.valid_history_mask.nbytes
            ),
            "estimated_tuning_hours": (
                elapsed
                * max(1, self.config.model_tuning_budgets[model_name])
                * len(self.plan.chronological_validation_folds)
                / 3600.0
            ),
            "preflight": preflight.to_dict(),
            "hardware": efficiency.hardware_fingerprint,
            "note": "One-fold pilot; full-search time is only a rough linear extrapolation.",
        }
        pilot_path = self.paths.root / "pilots" / track.value / f"{model_name}.json"
        pilot_path.parent.mkdir(parents=True, exist_ok=True)
        _write_json_replace(pilot_path, report)
        return report

    def load_selected(self, model_name: str, track: BenchmarkTrack) -> dict[str, Any]:
        path = self._selected_path(model_name, track)
        try:
            with path.open(encoding="utf-8") as input_file:
                selected = json.load(input_file)
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"No frozen selected configuration exists: {path}") from error
        if selected.get("snapshot_sha256") != self.snapshot.identity_sha256:
            raise ValueError("Selected configuration belongs to another benchmark snapshot.")
        expected = self._selection_identity(model_name, track, selected.get("parameters", {}))
        if any(selected.get(key) != value for key, value in expected.items()):
            raise ValueError("Selected configuration identity does not match the current protocol/search space/representation.")
        if not selected.get("all_development_folds"):
            raise ValueError(
                "Final execution requires a winner selected across every development fold."
            )
        return selected

    def _score_candidate(
        self,
        model_name: str,
        parameters: dict[str, Any],
        track: BenchmarkTrack,
        folds: tuple[ChronologicalValidationFold, ...],
    ) -> tuple[float, ...]:
        scores = []
        for fold in folds:
            training_sessions, evaluation_sessions = self._fold_sessions(fold)
            representation, training, evaluation, model_representation = self._corpora(
                model_name, parameters, track, training_sessions, evaluation_sessions
            )
            directory = self._execute_persisted_run(
                model_name=model_name, parameters=parameters, track=track,
                seed=self.config.tuning_seed,
                run_name=f"candidate_{canonical_sha256(parameters)[:16]}_train_{fold.training.end_index}",
                training_sessions=training_sessions, evaluation_sessions=evaluation_sessions,
                training=training,
                evaluation=evaluation,
                representation=representation, model_representation=model_representation,
                output_root=self.paths.tuning / "fold_runs", resume=True,
            )
            _, metrics = load_benchmark_summary(directory)
            scores.append(metrics["mean_head_macro_f1"])
        return tuple(scores)

    def _corpora(
        self,
        model_name: str,
        parameters: dict[str, Any],
        track: BenchmarkTrack,
        training_sessions: tuple[pd.DataFrame, ...],
        evaluation_sessions: tuple[pd.DataFrame, ...],
    ) -> tuple[RepresentationSpec, Any, Any, str]:
        family = get_model_family(model_name)
        layout = "sequential" if family.representation == "sequential" else "tabular"
        context_length = (
            self.config.window_length
            if track is BenchmarkTrack.CONTROLLED
            else int(parameters.get("context_length", self.config.window_length))
        )
        padding_policy = (
            "none" if track is BenchmarkTrack.CONTROLLED else "left_zero_masked"
        )
        first_position = (
            self.config.controlled_first_scored_candle_position
            if track is BenchmarkTrack.CONTROLLED
            else self.config.best_of_family_first_scored_candle_position
        )
        # These frames are owned by this executor's immutable snapshot. The key includes
        # exact train/evaluation identities; no fitted state can cross a chronology boundary.
        cache_key = (
            self.snapshot.identity_sha256, track.value, self.config.feature_columns,
            tuple(_session_ids(training_sessions)), tuple(_session_ids(evaluation_sessions)),
            first_position, self.config.window_length,
        )
        cached = self._representation_cache.get(cache_key)
        build_context = context_length if track is BenchmarkTrack.CONTROLLED else max(64, context_length)
        if cached is not None and cached[0].window_length >= context_length:
            self._representation_cache.move_to_end(cache_key)
        else:
            cached = None
        if cached is None and track is BenchmarkTrack.CONTROLLED:
            standardizer = fit_unique_candle_standardizer(
                training_sessions, feature_columns=self.config.feature_columns
            )
            training_sessions = transform_sessions(
                training_sessions, standardizer,
                feature_columns=self.config.feature_columns,
            )
            evaluation_sessions = transform_sessions(
                evaluation_sessions, standardizer,
                feature_columns=self.config.feature_columns,
            )
        representation = RepresentationSpec(
            name=f"{track.value}_{layout}_{context_length}",
            window_length=context_length,
            layout=layout,
            standardization=(
                "none" if track is BenchmarkTrack.CONTROLLED else "training_only"
            ),
            polynomial_degree=(2 if model_name == "polynomial_logistic" else None),
            padding_policy=padding_policy,
            feature_columns=self.config.feature_columns,
        )
        universe = EvaluationUniverse(first_scored_candle_position=first_position)
        if cached is None:
            maximum = replace(representation, window_length=build_context)
            cached = (
                build_representation_corpus(training_sessions, maximum, universe),
                build_representation_corpus(evaluation_sessions, maximum, universe),
            )
            size = sum(corpus.features.nbytes + corpus.valid_history_mask.nbytes for corpus in cached)
            if size <= self.representation_cache_limit_bytes:
                if cache_key in self._representation_cache:
                    old = self._representation_cache.pop(cache_key)
                    self._representation_cache_bytes -= sum(c.features.nbytes + c.valid_history_mask.nbytes for c in old)
                while self._representation_cache and self._representation_cache_bytes + size > self.representation_cache_limit_bytes:
                    _, old = self._representation_cache.popitem(last=False)
                    self._representation_cache_bytes -= sum(c.features.nbytes + c.valid_history_mask.nbytes for c in old)
                self._representation_cache[cache_key] = cached
                self._representation_cache_bytes += size
        def trailing(corpus):
            if corpus.window_length == context_length:
                return corpus
            return replace(corpus, features=corpus.features[:, -context_length:],
                           valid_history_mask=corpus.valid_history_mask[:, -context_length:],
                           window_length=context_length)
        return (
            representation,
            trailing(cached[0]),
            trailing(cached[1]),
            layout,
        )

    def _execute_persisted_run(
        self,
        *,
        model_name: str,
        parameters: dict[str, Any],
        track: BenchmarkTrack,
        seed: int,
        run_name: str,
        training_sessions: tuple[pd.DataFrame, ...],
        evaluation_sessions: tuple[pd.DataFrame, ...],
        training: Any,
        evaluation: Any,
        representation: RepresentationSpec,
        model_representation: str,
        output_root: Path,
        resume: bool,
    ) -> Path:
        self._require_source()
        actual_device = self.device if get_model_family(model_name).representation == "sequential" else "cpu"
        model_configuration = {
            "model_name": model_name,
            "parameters": parameters,
            "track": track.value,
        }
        identity = build_benchmark_run_identity(
            track=track.value,
            model_name=model_name,
            run_name=run_name,
            seed=seed,
            model_configuration=model_configuration,
            representation=representation.to_dict(),
            feature_columns=self.config.feature_columns,
            window_length=representation.window_length,
            training_session_ids=_session_ids(training_sessions),
            validation_session_ids=[],
            test_session_ids=_session_ids(evaluation_sessions),
            annotation_snapshot_identity=self.snapshot.source_identities.get(
                "annotation_rows_sha256", self.snapshot.identity_sha256
            ),
            normalized_dataset_identity=self.snapshot.source_identities.get(
                "normalized_sha256", self.snapshot.identity_sha256
            ),
            protocol_configuration=asdict(self.config),
            search_space=self._space(model_name, track),
            label_mapping={"bull": 0, "bear": 1, "range": 2},
            training_session_range=(
                int(training.session_indices.min()),
                int(training.session_indices.max()) + 1,
            ),
            validation_session_ranges=(),
            test_session_range=(
                int(evaluation.session_indices.min()),
                int(evaluation.session_indices.max()) + 1,
            ),
        )
        environment = resume_environment_fingerprint(device=actual_device)
        environment["hardware"] = hardware_fingerprint(device=actual_device, cpu_worker_count=self.config.cpu_worker_count)
        paths, state = prepare_benchmark_run(
            output_root,
            identity,
            resume=resume,
            execution_environment=environment,
        )
        if state == "complete":
            return paths.directory

        progress = _read_optional_json(paths.progress)
        completed = progress.get("completed", {})
        dataset_description = {
            "snapshot_path": str(self.snapshot.data_path),
            "snapshot_sha256": self.snapshot.identity_sha256,
            **self.snapshot.source_identities,
            "declared_final_seeds": list(self.config.final_seeds) if get_model_family(model_name).stochastic and run_name.startswith("final_seed_") else [seed],
        }
        if completed.get("metrics"):
            publish_checkpointed_run(paths, identity, model_configuration, dataset_description)
            return paths.directory
        model_is_checkpointed = _component_is_complete(paths.model, completed.get("model"))
        saved_predictions = None
        if model_is_checkpointed:
            # If the fitted model survived, missing downstream work must use that exact state.
            model = load_model(model_name, str(paths.model), device=actual_device)
            if _component_is_complete(paths.predictions, completed.get("predictions")):
                # Prediction publication includes timing in the same durable checkpoint.
                # Recovering metrics must not measure a new inference pass and substitute it.
                saved_predictions = pd.read_parquet(paths.predictions)
                evidence = progress.get("efficiency")
                persisted_metrics = _metrics_from_prediction_frame(
                    saved_predictions,
                    efficiency=EfficiencyMetrics(**evidence) if evidence else None,
                )
            else:
                evidence = progress.get("training_evidence")
                if not evidence:
                    raise ValueError("Fitted checkpoint lacks original training timing evidence.")
                result = evaluate_fitted_model(
                    model,
                    evaluation=evaluation,
                    representation=model_representation,
                    training_seconds=evidence["training_seconds"],
                    training_sample_count=evidence["training_sample_count"],
                    serialized_model_bytes=paths.model.stat().st_size,
                    device=actual_device,
                    cpu_worker_count=self.config.cpu_worker_count,
                    inference_timing_repetitions=self.config.inference_timing_repetitions,
                )
        else:
            model = self._build_model(
                model_name,
                parameters,
                seed,
                prestandardized=(track is BenchmarkTrack.CONTROLLED),
            )
            result = run_model_once(
                model,
                training=training,
                evaluation=evaluation,
                representation=model_representation,
                device=actual_device,
                cpu_worker_count=self.config.cpu_worker_count,
                inference_timing_repetitions=self.config.inference_timing_repetitions,
                checkpoint_fitted=lambda fitted, evidence: checkpoint_fitted_model(paths, fitted, evidence),
            )
        self._require_source()
        save_benchmark_run(
            paths,
            identity,
            model=model,
            predictions=(saved_predictions if saved_predictions is not None else build_benchmark_prediction_frame(evaluation, result)),
            metrics=(persisted_metrics if saved_predictions is not None else result.metrics),
            model_configuration=model_configuration,
            dataset_description=dataset_description,
        )
        return paths.directory

    def _build_model(
        self,
        model_name: str,
        parameters: dict[str, Any],
        seed: int,
        *,
        prestandardized: bool,
    ) -> Any:
        return build_model(
            model_name,
            random_seed=seed,
            parameters=parameters,
            cpu_worker_count=self.config.cpu_worker_count,
            device=self.device,
            prestandardized=prestandardized,
        )

    def _selected_folds(
        self, fold_indices: tuple[int, ...] | None
    ) -> tuple[ChronologicalValidationFold, ...]:
        folds = self.plan.chronological_validation_folds
        if fold_indices is None:
            return folds
        if not fold_indices or any(index < 0 or index >= len(folds) for index in fold_indices):
            raise ValueError("Requested chronological fold index is outside the plan.")
        return tuple(folds[index] for index in fold_indices)

    def _fold_sessions(
        self, fold: ChronologicalValidationFold
    ) -> tuple[tuple[pd.DataFrame, ...], tuple[pd.DataFrame, ...]]:
        return (
            tuple(self.sessions[fold.training.start_index:fold.training.end_index]),
            tuple(self.sessions[fold.validation.start_index:fold.validation.end_index]),
        )

    def _selected_record(
        self,
        model_name: str,
        track: BenchmarkTrack,
        parameters: dict[str, Any],
        fold_scores: tuple[float, ...],
        *,
        fold_indices: tuple[int, ...],
        all_development_folds: bool,
    ) -> dict[str, Any]:
        self._require_source()
        return {
            "format_version": 1,
            "state": "frozen" if all_development_folds else "diagnostic",
            "snapshot_sha256": self.snapshot.identity_sha256,
            "protocol_sha256": canonical_sha256(asdict(self.config)),
            "search_space_sha256": canonical_sha256(
                self.search_spaces.get(model_name, {})
            ),
            "model_name": model_name,
            "track": track.value,
            "parameters": parameters,
            "fold_indices": [index + 1 for index in fold_indices],
            "fold_count": len(fold_indices),
            "all_development_folds": all_development_folds,
            "fold_mean_head_macro_f1": list(fold_scores),
            "mean_validation_macro_f1": sum(fold_scores) / len(fold_scores),
            **self._selection_identity(model_name, track, parameters),
        }

    def _write_tuning_result(self, selected: dict[str, Any]) -> None:
        """Persist diagnostics without letting a partial fold selection unlock final data."""

        fold_token = "all_folds" if selected["all_development_folds"] else (
            "folds_" + "_".join(str(value) for value in selected["fold_indices"])
        )
        path = (
            self.paths.tuning
            / str(selected["track"])
            / str(selected["model_name"])
            / fold_token
            / "selected_trial.json"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        _write_json_replace(path, selected)

    def _write_selected(self, selected: dict[str, Any]) -> None:
        path = self._selected_path(
            str(selected["model_name"]), BenchmarkTrack(str(selected["track"]))
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        _write_json_replace(path, selected)

    def _selected_path(self, model_name: str, track: BenchmarkTrack) -> Path:
        return self.paths.selected / track.value / f"{model_name}.json"

    def _require_scaling_acknowledgement(
        self,
        model_name: str,
        track: BenchmarkTrack,
        candidate_count: int,
        fold_count: int,
        acknowledged: bool,
    ) -> None:
        first_position = (self.config.controlled_first_scored_candle_position
                          if track is BenchmarkTrack.CONTROLLED
                          else self.config.best_of_family_first_scored_candle_position)
        largest_fold = max(self.plan.chronological_validation_folds, key=lambda fold: fold.training.end_index)
        train_sessions, validation_sessions = self._fold_sessions(largest_fold)
        estimated_rows = sum(max(0, len(session)-first_position) for session in train_sessions)
        validation_rows = sum(max(0, len(session)-first_position) for session in validation_sessions)
        context_space = self._space(model_name, track).get("context_length", {})
        context = (self.config.window_length if track is BenchmarkTrack.CONTROLLED else
                   max(context_space.get("values", [context_space.get("value", self.config.window_length)])))
        dimension = context * (len(self.config.feature_columns) + (1 if track is BenchmarkTrack.BEST_OF_FAMILY else 0))
        preflight = build_scalability_preflight(
            model_name=model_name,
            training_samples=estimated_rows,
            evaluation_samples=validation_rows,
            feature_dimension=dimension,
            candidate_count=candidate_count,
            fold_count=fold_count,
            seed_count=(
                len(self.config.final_seeds)
                if get_model_family(model_name).stochastic else 1
            ),
            polynomial_degree=2 if model_name == "polynomial_logistic" else None,
        )
        dense_memory_gib = max(
            preflight.training_dense_float64_bytes,
            preflight.evaluation_dense_float64_bytes,
        ) / (1024**3)
        exceeds_memory_budget = dense_memory_gib > self.config.pilot_max_dense_memory_gib
        preflight_path = self.paths.root / "preflights" / track.value / f"{model_name}.json"
        preflight_path.parent.mkdir(parents=True, exist_ok=True)
        _write_json_replace(preflight_path, preflight.to_dict())
        actual_device = self.device if get_model_family(model_name).representation == "sequential" else "cpu"
        print(
            f"Preflight {track.value}/{model_name}: {estimated_rows:,} training rows, "
            f"{validation_rows:,} validation rows, {dimension} input features, "
            f"{preflight.transformed_feature_dimension:,} transformed features, "
            f"{fold_count} folds × {candidate_count} candidates; "
            f"{preflight.seed_count} final seeds; approximately {preflight.approximate_job_count} jobs; "
            f"float64 dense matrix up to {dense_memory_gib:.2f} GiB; device={actual_device}. "
            f"Details: {preflight_path}"
        )
        # Best-of-family and controlled pilots can differ in context, so only a matching
        # track can justify a timing acknowledgement.
        pilot_path = self.paths.root / "pilots" / track.value / f"{model_name}.json"
        estimated_hours = None
        if pilot_path.is_file():
            estimated_hours = float(
                _read_optional_json(pilot_path).get("estimated_tuning_hours", 0.0)
            )
        exceeds_time_budget = (
            estimated_hours is not None
            and estimated_hours > self.config.pilot_max_estimated_hours
        )
        if (
            preflight.acknowledgement_required
            or exceeds_memory_budget
            or exceeds_time_budget
        ) and not acknowledged:
            raise PermissionError(
                f"{model_name} scalability risk requires an explicit acknowledgement after "
                "reviewing its pilot and dense-memory estimate."
            )


def _suggest_parameters(trial: Any, space: dict[str, dict[str, Any]]) -> dict[str, Any]:
    parameters = {}
    for name, specification in space.items():
        kind = specification["type"]
        if kind == "fixed":
            value = specification["value"]
        elif kind == "categorical":
            value = trial.suggest_categorical(name, specification["values"])
        elif kind == "integer":
            value = trial.suggest_int(
                name, int(specification["low"]), int(specification["high"]),
                step=int(specification.get("step", 1)),
            )
        else:
            value = trial.suggest_float(
                name, float(specification["low"]), float(specification["high"]),
                log=(kind == "log_float"),
            )
        parameters[name] = value
    return parameters


def _session_ids(sessions: tuple[pd.DataFrame, ...]) -> list[str]:
    return [str(session["session_date"].iloc[0]) for session in sessions]


def _read_optional_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8") as input_file:
        return json.load(input_file)


def _write_json_replace(path: Path, values: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    with temporary.open("x", encoding="utf-8") as output_file:
        json.dump(values, output_file, indent=2, sort_keys=True, default=str)
        output_file.write("\n")
        output_file.flush()
        os.fsync(output_file.fileno())
    os.replace(temporary, path)
