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
    """Derive stable lifecycle directories beneath one immutable study root."""

    # Every stage receives a dedicated subtree so development diagnostics, frozen selections,
    # learning curves, and final evidence cannot be mistaken for one another.
    root = Path(root)
    return BenchmarkStudyPaths(
        root=root,
        tuning=root / "tuning",
        selected=root / "selected",
        runs=root / "runs",
        learning_curves=root / "learning_curves",
    )


def _study_locked(method):
    """Serialize study-level mutations while allowing independent process launches."""

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
        """Bind one immutable snapshot, protocol, source tree, and declared study scope."""

        # Keep the snapshot, protocol, search spaces, and root together for the executor's entire
        # lifetime; allowing any one of them to drift would change the study between stages.
        self.snapshot = snapshot
        self.config = config
        self.search_spaces = search_spaces
        self.paths = study_paths(study_root)
        self.device = device

        # Validate an explicitly requested accelerator at construction time. A later silent CPU
        # fallback would make persisted device and timing evidence false.
        if device != "cpu":
            from pricesanity.models.sequence import _resolve_device

            _resolve_device(device)

        # Capture source state once so a long-running executor cannot mix code revisions
        # between candidate fitting, recovery, and final artifact publication.
        self.execution_source = source_identity()
        self._representation_cache = OrderedDict()
        self._representation_cache_bytes = 0

        # Bound reusable corpus memory independently of model memory so large searches cannot
        # accumulate one immutable representation pair per chronological fold indefinitely.
        self.representation_cache_limit_bytes = 512 * 1024**2

        # The protocol's development boundary must match the physically isolated snapshot; an
        # in-memory slice cannot repair a snapshot that already exposes different rows.
        if snapshot.development_session_count != config.development_session_count:
            raise ValueError(
                "Benchmark execution requires a physically isolated snapshot "
                "matching the protocol."
            )

        # The executor only loads development rows. Final rows remain in the sealed file until
        # the explicit final path validates the global development freeze.
        self.sessions = load_snapshot_sessions(snapshot)

        # Derive all chronological ranges once from the frozen corpus and protocol. Later stages
        # refer to this plan rather than independently recalculating split boundaries.
        self.plan = plan_benchmark(snapshot.session_count, config)

        # Create only lifecycle containers. Individual run directories remain identity-bound and
        # are reserved atomically by the artifact layer when work actually begins.
        self.paths.tuning.mkdir(parents=True, exist_ok=True)
        self.paths.selected.mkdir(parents=True, exist_ok=True)
        self.paths.runs.mkdir(parents=True, exist_ok=True)
        self.paths.learning_curves.mkdir(parents=True, exist_ok=True)

        # Scope is the declared cross-product that must finish development before the global seal
        # can unlock any final holdout evaluation.
        self.scope = {
            "models": sorted(config.model_tuning_budgets),
            "tracks": sorted(declared_tracks),
            "snapshot_sha256": snapshot.identity_sha256,
        }
        scope_path = self.paths.root / "study_scope.json"

        # Serialize scope creation with every other study mutation, then make an existing scope
        # immutable so a resumed directory cannot silently add or remove models or tracks.
        with _run_lock(self.paths.root / ".study.lock"):
            if scope_path.exists():
                if _read_optional_json(scope_path) != self.scope:
                    raise ValueError(
                        "Declared study scope is immutable; use a new study directory."
                    )
            else:
                _write_json_replace(scope_path, self.scope)

    def _require_source(self) -> None:
        # A long-lived executor may outlast an edit in the working tree. Stop at every lifecycle
        # boundary rather than mixing outputs from two source states under one study identity.
        if source_identity() != self.execution_source:
            raise ValueError(
                "Source changed while this executor was active; restart with "
                "compatible source."
            )

    def _require_development(self) -> None:
        self._require_source()

        # The global seal is a one-way transition. Once present, no additional tuning, pilots, or
        # learning-curve work may influence choices after final evaluation becomes possible.
        if (self.paths.root / "development_frozen.json").exists():
            raise PermissionError(
                "Study development is frozen; tuning and diagnostics cannot mutate it."
            )

    def _space(self, model_name: str, track: BenchmarkTrack) -> dict:
        # Reject undeclared work before resolving a search space. Ad hoc models or tracks would
        # evade the complete-scope requirement enforced by the global development freeze.
        if (
            model_name not in self.scope["models"]
            or track.value not in self.scope["tracks"]
        ):
            raise ValueError("Model and track must belong to the declared study scope.")
        return track_search_space(
            model_name,
            self.search_spaces.get(model_name, {}),
            track.value,
        )

    def _selection_identity(
        self,
        model_name: str,
        track: BenchmarkTrack,
        parameters: dict,
    ) -> dict:
        """Bind a selected configuration to its protocol, search space, and inputs."""

        # Selection identity includes both the candidate parameters and the representation rules.
        # Equal estimator settings under a different track or context remain different winners.
        return {
            "snapshot_sha256": self.snapshot.identity_sha256,
            "protocol_sha256": canonical_sha256(asdict(self.config)),
            "search_space_sha256": canonical_sha256(self._space(model_name, track)),
            "model_name": model_name,
            "track": track.value,
            "parameters_sha256": canonical_sha256(parameters),
            "selection_source": source_identity(),
            "representation_sha256": canonical_sha256(
                {
                    "context": (
                        self.config.window_length
                        if track is BenchmarkTrack.CONTROLLED
                        else parameters.get(
                            "context_length",
                            self.config.window_length,
                        )
                    ),
                    "features": self.config.feature_columns,
                    "track": track.value,
                    "preprocessing_version": 2,
                }
            ),
        }

    def _freeze_contents(self) -> dict:
        """Build the global seal only after every declared track has a frozen winner."""

        self._require_source()
        selections = {}

        # Require a verified all-fold selection for every declared model/track pair. The seal is
        # unavailable until the entire development decision surface has been fixed.
        for track in self.scope["tracks"]:
            for model in self.scope["models"]:
                selected = self.load_selected(model, BenchmarkTrack(track))
                selections[f"{track}/{model}"] = {
                    "configuration_sha256": canonical_sha256(selected),
                    "representation_sha256": selected["representation_sha256"],
                    "search_space_sha256": selected["search_space_sha256"],
                }
        return {
            "state": "frozen",
            "format_version": 1,
            "scope": self.scope,
            "snapshot_sha256": self.snapshot.identity_sha256,
            "protocol_sha256": canonical_sha256(asdict(self.config)),
            "selected": selections,
            "source": source_identity(),
        }

    @_study_locked
    def freeze_development(self) -> dict:
        # Every declared track freezes together: inspecting one final track must never
        # influence continued selection in the other.
        contents = self._freeze_contents()
        path = self.paths.root / "development_frozen.json"

        # Repeated freeze calls are idempotent only for byte-equivalent scientific contents. Any
        # changed winner or source identity invalidates the attempted reuse.
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

        # Tuning is a development-only mutation and must stop after source drift or global freeze.
        self._require_development()

        # Registry metadata determines stochastic behavior and representation needs; fold
        # selection stays within the already planned chronological development ranges.
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

        # Candidate budget and track-filtered search space form part of the tuning contract. The
        # preflight must run before either parameter-free fitting or Optuna work begins.
        budget = self.config.model_tuning_budgets[model_name]
        space = self._space(model_name, track)
        self._require_scaling_acknowledgement(
            model_name, track, max(1, budget), len(folds), acknowledge_scaling_risk
        )

        # Parameter-free baselines still run every requested fold, but do not create a meaningless
        # optimization study with an empty search space.
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

        # Optuna remains optional for artifact readers and baseline execution; import it only on a
        # search path that genuinely requires optimization.
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

        # Persist the objective's full identity beside the database. SQLite contents alone do not
        # prove which folds, source bytes, or search space produced its trial scores.
        identity_path = tuning_directory / "study_identity.json"
        if identity_path.exists():
            if _read_optional_json(identity_path) != tuning_identity:
                raise ValueError("Optuna study identity changed; create a distinct study.")
        elif storage_path.exists():
            raise ValueError(
                "Legacy Optuna study lacks immutable identity; create a distinct study."
            )
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

        # Count terminal and queued trials separately so resuming fills only the configured budget
        # and cannot double-enqueue work left waiting by an interrupted process.
        terminal_count = sum(trial.state in terminal_states for trial in study.trials)
        waiting_count = sum(
            trial.state is optuna.trial.TrialState.WAITING for trial in study.trials
        )
        remaining_trials = max(0, budget - terminal_count)
        if waiting_count > remaining_trials:
            raise ValueError(
                "Persistent tuning study contains more queued trials than its budget."
            )

        def objective(trial: Any) -> float:
            # Score one suggested configuration independently on every selected chronological fold;
            # the scalar objective is their equal mean, not the most favorable fold.
            parameters = _suggest_parameters(trial, space)
            fold_scores = self._score_candidate(
                model_name, parameters, track, folds
            )
            trial.set_user_attr("fold_mean_head_macro_f1", fold_scores)
            trial.set_user_attr("parameters", parameters)
            return float(sum(fold_scores) / len(fold_scores))

        if remaining_trials:
            # Persistent storage resumes completed trials and schedules only the outstanding count.
            study.optimize(objective, n_trials=remaining_trials)
        if not study.best_trials:
            raise RuntimeError("Tuning did not complete a legal candidate.")
        best = study.best_trial

        # Freeze the exact all-fold evidence and parameters used by the winning Optuna trial. A
        # diagnostic fold subset is recorded but never promoted to the final-selection path.
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

        # Requiring explicit confirmation makes final access a deliberate one-way research action,
        # not a side effect of completing model selection.
        if not confirm_final_holdout:
            raise PermissionError(
                "Final holdout remains locked; pass the explicit confirmation only after freeze."
            )
        selected = self.load_selected(model_name, track)
        seal_path = self.paths.root / "development_frozen.json"

        # Validate both presence and reconstructed contents of the global seal before asking the
        # snapshot layer to open physically isolated holdout bytes.
        if not seal_path.exists():
            raise PermissionError("Final holdout is locked until global development freeze.")
        if _read_optional_json(seal_path) != self._freeze_contents():
            raise ValueError("Frozen study identity changed; final holdout remains locked.")
        parameters = dict(selected["parameters"])

        # Final training uses every development session fixed by the protocol; evaluation uses the
        # separately loaded holdout and never contributes fitted preprocessing state.
        development = tuple(
            self.sessions[
                self.plan.development.start_index : self.plan.development.end_index
            ]
        )
        holdout = load_sealed_holdout(
            self.snapshot,
            seal_path=seal_path,
            confirm_final_holdout=confirm_final_holdout,
        )
        representation, training, evaluation, model_representation = self._corpora(
            model_name, parameters, track, development, holdout
        )
        family = get_model_family(model_name)

        # Stochastic families require the declared replicate set, while deterministic families use
        # one stable protocol seed solely for a common run identity.
        seeds = (
            self.config.final_seeds
            if family.stochastic
            else (self.config.tuning_seed,)
        )
        output_directories = []

        # Persist each seed as an independent identity-bound run so a failed replicate can resume
        # without invalidating completed peers or averaging a partial seed set.
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

        # Learning curves reuse the frozen all-fold winner; they measure data quantity rather than
        # retuning architecture separately at every prefix size.
        selected = self.load_selected(model_name, track)
        parameters = dict(selected["parameters"])
        points = plan_learning_curve(
            self.config.learning_curve_session_counts,
            development_session_count=self.config.development_session_count,
            evaluation_range=self.config.learning_curve_evaluation_range,
        )
        outputs = []

        # Every point changes only its chronological training prefix. The fixed future evaluation
        # block makes scores comparable across increasing amounts of training evidence.
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

        # Use the established Transformer incumbent or a deterministic representative point from
        # another family's space; a pilot estimates scale rather than selecting a winner.
        parameters = (
            dict(self.config.transformer_incumbent)
            if model_name == "transformer"
            else representative_candidate(self._space(model_name, track))
        )
        if model_name in {"tcn", "gru", "transformer"}:
            # One epoch keeps the pilot bounded while still exercising the complete training path.
            parameters["epochs"] = 1

        # The first chronological fold supplies realistic development shapes without touching the
        # sealed final population.
        fold = self.plan.chronological_validation_folds[0]
        training_sessions, evaluation_sessions = self._fold_sessions(fold)
        representation, training, evaluation, model_representation = self._corpora(
            model_name, parameters, track, training_sessions, evaluation_sessions
        )
        model = self._build_model(
            model_name, parameters, self.config.tuning_seed,
            prestandardized=(track is BenchmarkTrack.CONTROLLED),
        )

        # Time the complete runner path because setup, fit, and inference all contribute to a
        # practical search estimate on the selected machine.
        started = perf_counter()
        result = run_model_once(
            model,
            training=training,
            evaluation=evaluation,
            representation=model_representation,
            device=(
                self.device
                if get_model_family(model_name).representation == "sequential"
                else "cpu"
            ),
            cpu_worker_count=self.config.cpu_worker_count,
            inference_timing_repetitions=self.config.inference_timing_repetitions,
        )
        elapsed = perf_counter() - started
        efficiency = result.metrics.efficiency
        assert efficiency is not None

        # Measure serialized size in a disposable directory; pilots must not publish model
        # artifacts that could be confused with completed benchmark evidence.
        with tempfile.TemporaryDirectory(prefix="pricesanity-pilot-") as directory:
            model_path = Path(directory) / "model.bin"
            model.save(model_path)
            serialized_model_bytes = model_path.stat().st_size
        flat_feature_dimension = int(np.prod(training.features.shape[1:]))

        # Combine observed sample shapes with the declared search budget to expose dense expansion
        # and job-count risks before launching the full study.
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

        # Pilot reports are development diagnostics keyed by both track and family because their
        # representation size and execution device can differ.
        pilot_path = self.paths.root / "pilots" / track.value / f"{model_name}.json"
        pilot_path.parent.mkdir(parents=True, exist_ok=True)
        _write_json_replace(pilot_path, report)
        return report

    def load_selected(self, model_name: str, track: BenchmarkTrack) -> dict[str, Any]:
        """Load a winner only when every identity still matches the active study."""

        # Read only the canonical all-fold selection path; diagnostic selections live under the
        # tuning subtree and can never satisfy this lookup.
        path = self._selected_path(model_name, track)
        try:
            with path.open(encoding="utf-8") as input_file:
                selected = json.load(input_file)
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"No frozen selected configuration exists: {path}") from error
        if selected.get("snapshot_sha256") != self.snapshot.identity_sha256:
            raise ValueError("Selected configuration belongs to another benchmark snapshot.")

        # Reconstruct every selection identity field from the active executor instead of trusting
        # the stored record to describe its own compatibility.
        expected = self._selection_identity(
            model_name,
            track,
            selected.get("parameters", {}),
        )
        if any(selected.get(key) != value for key, value in expected.items()):
            raise ValueError(
                "Selected configuration identity does not match the current "
                "protocol/search space/representation."
            )
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
        """Fit one candidate independently on every selected chronological fold."""

        scores = []

        # Rebuild train-only preprocessing and model state for each fold. Reusing a fitted object
        # would leak later-fold history backward into an earlier validation score.
        for fold in folds:
            training_sessions, evaluation_sessions = self._fold_sessions(fold)
            representation, training, evaluation, model_representation = self._corpora(
                model_name, parameters, track, training_sessions, evaluation_sessions
            )
            directory = self._execute_persisted_run(
                model_name=model_name,
                parameters=parameters,
                track=track,
                seed=self.config.tuning_seed,
                run_name=(
                    f"candidate_{canonical_sha256(parameters)[:16]}_"
                    f"train_{fold.training.end_index}"
                ),
                training_sessions=training_sessions,
                evaluation_sessions=evaluation_sessions,
                training=training,
                evaluation=evaluation,
                representation=representation,
                model_representation=model_representation,
                output_root=self.paths.tuning / "fold_runs",
                resume=True,
            )

            # Load only the verified metric summary; candidate predictions remain available for
            # audit without imposing their I/O cost on the tuning objective.
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
        """Build aligned train/evaluation corpora under the selected track rules."""

        # Registry representation type determines whether windows remain sequential for neural
        # models or flatten into fixed-width rows for tabular estimators.
        family = get_model_family(model_name)
        layout = "sequential" if family.representation == "sequential" else "tabular"

        # Controlled models share the protocol context; best-of-family models may tune context as
        # part of their declared advantage.
        context_length = (
            self.config.window_length
            if track is BenchmarkTrack.CONTROLLED
            else int(parameters.get("context_length", self.config.window_length))
        )
        padding_policy = (
            "none" if track is BenchmarkTrack.CONTROLLED else "left_zero_masked"
        )

        # The first scored position defines the target population. Controlled comparisons require
        # full history, while best-of-family can score masked early-session contexts.
        first_position = (
            self.config.controlled_first_scored_candle_position
            if track is BenchmarkTrack.CONTROLLED
            else self.config.best_of_family_first_scored_candle_position
        )
        # These frames are owned by this executor's immutable snapshot. The key includes
        # exact train/evaluation identities; no fitted state can cross a chronology boundary.
        cache_key = (
            self.snapshot.identity_sha256,
            track.value,
            self.config.feature_columns,
            tuple(_session_ids(training_sessions)),
            tuple(_session_ids(evaluation_sessions)),
            first_position,
            self.config.window_length,
        )
        cached = self._representation_cache.get(cache_key)
        # Best-of-family contexts share one maximum-length corpus. Shorter candidates take a
        # trailing view, preserving target identities while avoiding repeated window builds.
        build_context = (
            context_length
            if track is BenchmarkTrack.CONTROLLED
            else max(64, context_length)
        )
        if cached is not None and cached[0].window_length >= context_length:
            # A cache hit becomes most recently used so eviction preserves actively reused folds.
            self._representation_cache.move_to_end(cache_key)
        else:
            cached = None

        # Controlled preprocessing is fit exactly once on this fold's training sessions, then
        # applied unchanged to evaluation sessions before either corpus is windowed.
        if cached is None and track is BenchmarkTrack.CONTROLLED:
            standardizer = fit_unique_candle_standardizer(
                training_sessions,
                feature_columns=self.config.feature_columns,
            )
            training_sessions = transform_sessions(
                training_sessions,
                standardizer,
                feature_columns=self.config.feature_columns,
            )
            evaluation_sessions = transform_sessions(
                evaluation_sessions,
                standardizer,
                feature_columns=self.config.feature_columns,
            )

        # The representation specification enters artifact identity, making layout, context,
        # padding, feature order, and preprocessing policy auditable for every persisted run.
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

        # Both train and evaluation corpora use one first-position rule so model comparisons score
        # the identical candle population within a track.
        universe = EvaluationUniverse(first_scored_candle_position=first_position)
        if cached is None:
            # Build at the reusable maximum context, then derive trailing views for shorter
            # candidates without recomputing features or changing target identities.
            maximum = replace(representation, window_length=build_context)
            cached = (
                build_representation_corpus(training_sessions, maximum, universe),
                build_representation_corpus(evaluation_sessions, maximum, universe),
            )
            size = sum(
                corpus.features.nbytes + corpus.valid_history_mask.nbytes
                for corpus in cached
            )

            # Cache only pairs that fit the declared ceiling; execution may proceed uncached when
            # one fold is legitimately larger than the reusable-memory budget.
            if size <= self.representation_cache_limit_bytes:
                if cache_key in self._representation_cache:
                    old = self._representation_cache.pop(cache_key)
                    self._representation_cache_bytes -= sum(
                        corpus.features.nbytes
                        + corpus.valid_history_mask.nbytes
                        for corpus in old
                    )
                # Evict the least recently used corpora until this immutable pair fits within
                # the declared memory ceiling; oversized pairs remain uncached but usable.
                while (
                    self._representation_cache
                    and self._representation_cache_bytes + size
                    > self.representation_cache_limit_bytes
                ):
                    _, old = self._representation_cache.popitem(last=False)
                    self._representation_cache_bytes -= sum(
                        corpus.features.nbytes
                        + corpus.valid_history_mask.nbytes
                        for corpus in old
                    )
                self._representation_cache[cache_key] = cached
                self._representation_cache_bytes += size

        def trailing(corpus):
            # Retain the most recent causal history because every target sits at the window's final
            # position; leading history outside the selected context is deliberately discarded.
            if corpus.window_length == context_length:
                return corpus
            return replace(
                corpus,
                features=corpus.features[:, -context_length:],
                valid_history_mask=(
                    corpus.valid_history_mask[:, -context_length:]
                ),
                window_length=context_length,
            )
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
        """Run or resume one identity-bound model fit through atomic publication."""

        self._require_source()
        # Tabular libraries execute on CPU; only the registered sequence families may use the
        # explicitly selected accelerator in timing evidence and model restoration.
        actual_device = (
            self.device
            if get_model_family(model_name).representation == "sequential"
            else "cpu"
        )

        # Model configuration includes track because identical estimator parameters under
        # controlled and best-of-family representation rules are not the same experiment.
        model_configuration = {
            "model_name": model_name,
            "parameters": parameters,
            "track": track.value,
        }

        # Bind exact sessions, representation, protocol, search space, labels, and source evidence
        # before reserving output. Resume can then prove it is continuing this precise run.
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

        # Resume compatibility records the actual execution device and hardware separately from
        # the scientific identity so binary reuse is strict without redefining the experiment.
        environment = resume_environment_fingerprint(device=actual_device)
        environment["hardware"] = hardware_fingerprint(
            device=actual_device,
            cpu_worker_count=self.config.cpu_worker_count,
        )
        paths, state = prepare_benchmark_run(
            output_root,
            identity,
            resume=resume,
            execution_environment=environment,
        )

        # A verified complete run is idempotent output; no model loading or inference is necessary.
        if state == "complete":
            return paths.directory

        # The progress manifest determines which exact stages survived. Files without a committed
        # checksum are ignored even if a crash left bytes at their canonical paths.
        progress = _read_optional_json(paths.progress)
        completed = progress.get("completed", {})

        # Persist snapshot provenance and the expected replicate set with every final run so later
        # aggregation can distinguish a complete stochastic set from selective seeds.
        dataset_description = {
            "snapshot_path": str(self.snapshot.data_path),
            "snapshot_sha256": self.snapshot.identity_sha256,
            **self.snapshot.source_identities,
            "declared_final_seeds": (
                list(self.config.final_seeds)
                if get_model_family(model_name).stochastic
                and run_name.startswith("final_seed_")
                else [seed]
            ),
        }

        # A metrics checkpoint means all scientific components are already durable; only the final
        # metadata commit marker may still be missing after a crash.
        if completed.get("metrics"):
            publish_checkpointed_run(paths, identity, model_configuration, dataset_description)
            return paths.directory
        model_is_checkpointed = _component_is_complete(
            paths.model,
            completed.get("model"),
        )
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
                # Reuse original training timing while evaluating the checkpointed fitted model.
                # Measuring a second fit would create performance evidence for a different run.
                evidence = progress.get("training_evidence")
                if not evidence:
                    raise ValueError(
                        "Fitted checkpoint lacks original training timing evidence."
                    )
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
            # No committed fitted model exists, so begin a fresh seeded fit and checkpoint it before
            # downstream inference. This is the only branch allowed to construct new model state.
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
                checkpoint_fitted=lambda fitted, evidence: checkpoint_fitted_model(
                    paths,
                    fitted,
                    evidence,
                ),
            )

        # Source bytes are checked again after expensive work. Edits made during fitting prevent a
        # mixed-source run from receiving a final completion marker.
        self._require_source()

        # Publication consumes either durable recovered predictions or new results, then recomputes
        # metrics from the exact saved rows inside the artifact layer.
        save_benchmark_run(
            paths,
            identity,
            model=model,
            predictions=(
                saved_predictions
                if saved_predictions is not None
                else build_benchmark_prediction_frame(evaluation, result)
            ),
            metrics=(
                persisted_metrics
                if saved_predictions is not None
                else result.metrics
            ),
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
        # Centralize construction so tuning, pilots, learning curves, and final runs pass identical
        # seed, worker, device, and preprocessing declarations to the model registry.
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
        """Resolve an optional diagnostic subset without changing configured folds."""

        folds = self.plan.chronological_validation_folds

        # None means the complete configured protocol. Explicit indices are diagnostics and retain
        # the original fold objects rather than constructing alternate split boundaries.
        if fold_indices is None:
            return folds
        if not fold_indices or any(index < 0 or index >= len(folds) for index in fold_indices):
            raise ValueError("Requested chronological fold index is outside the plan.")
        return tuple(folds[index] for index in fold_indices)

    def _fold_sessions(
        self, fold: ChronologicalValidationFold
    ) -> tuple[tuple[pd.DataFrame, ...], tuple[pd.DataFrame, ...]]:
        """Slice whole chronological sessions at one declared fold boundary."""

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
        """Record fold evidence and whether it is sufficient to freeze a winner."""

        self._require_source()

        # Only a complete multi-fold selection receives frozen state. Focused diagnostics retain
        # their evidence but cannot participate in the global development seal.
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

        # Encode fold scope in the path so diagnostic subsets cannot replace the all-fold tuning
        # record for the same family and track.
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
        """Publish the all-fold winner used by development freeze and final fitting."""

        # This canonical path is intentionally separate from tuning diagnostics; its presence is
        # what the freeze lifecycle treats as a model/track selection.
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
        """Persist the largest-fold cost estimate and gate risky benchmark launches."""

        # Scored-row estimates subtract the track's unscored prefix from every session so the
        # preflight reflects the actual representation population rather than raw candle count.
        first_position = (
            self.config.controlled_first_scored_candle_position
            if track is BenchmarkTrack.CONTROLLED
            else self.config.best_of_family_first_scored_candle_position
        )
        largest_fold = max(
            self.plan.chronological_validation_folds,
            key=lambda fold: fold.training.end_index,
        )
        train_sessions, validation_sessions = self._fold_sessions(largest_fold)
        # The largest fold gives a conservative dense-memory and runtime warning before any
        # candidate starts fitting; it does not inspect the sealed final holdout.
        estimated_rows = sum(
            max(0, len(session) - first_position) for session in train_sessions
        )
        validation_rows = sum(
            max(0, len(session) - first_position)
            for session in validation_sessions
        )

        # Use the largest permitted best-of-family context because dense feature expansion and
        # memory risk are governed by the most expensive legal candidate.
        context_space = self._space(model_name, track).get("context_length", {})
        context = (
            self.config.window_length
            if track is BenchmarkTrack.CONTROLLED
            else max(
                context_space.get(
                    "values",
                    [context_space.get("value", self.config.window_length)],
                )
            )
        )
        per_candle_dimension = len(self.config.feature_columns) + (
            1 if track is BenchmarkTrack.BEST_OF_FAMILY else 0
        )

        # Best-of-family adds one validity indicator per historical candle to the four market
        # features; controlled rows contain only their fully observed market geometry.
        dimension = context * per_candle_dimension
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

        # Compare the largest train or evaluation matrix with the configured safety ceiling, then
        # persist the full preflight before asking for acknowledgement.
        exceeds_memory_budget = (
            dense_memory_gib > self.config.pilot_max_dense_memory_gib
        )
        preflight_path = (
            self.paths.root
            / "preflights"
            / track.value
            / f"{model_name}.json"
        )
        preflight_path.parent.mkdir(parents=True, exist_ok=True)
        _write_json_replace(preflight_path, preflight.to_dict())

        # Tabular estimators execute on CPU even when the study device names an accelerator for
        # sequence models; report the device that this family will actually use.
        actual_device = (
            self.device
            if get_model_family(model_name).representation == "sequential"
            else "cpu"
        )
        print(
            f"Preflight {track.value}/{model_name}: {estimated_rows:,} training rows, "
            f"{validation_rows:,} validation rows, {dimension} input features, "
            f"{preflight.transformed_feature_dimension:,} transformed features, "
            f"{fold_count} folds × {candidate_count} candidates; "
            f"{preflight.seed_count} final seeds; approximately "
            f"{preflight.approximate_job_count} jobs; "
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

        # Any model-specific warning, memory excess, or pilot time excess requires a deliberate
        # acknowledgement. Absence of a pilot does not invent a time estimate.
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
    """Translate validated declarative distributions into one Optuna candidate."""

    parameters = {}

    # Interpret only the validated declarative distribution kinds used by repository search-space
    # files, leaving sampling and reproducibility under the persistent Optuna trial.
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
    """Represent ordered whole-session populations by their unique trading dates."""

    return [str(session["session_date"].iloc[0]) for session in sessions]


def _read_optional_json(path: Path) -> dict[str, Any]:
    """Treat an absent progress-style JSON file as an empty state."""

    # Absence is expected before the first checkpoint; malformed existing JSON still raises and
    # must never be treated as a fresh run.
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8") as input_file:
        return json.load(input_file)


def _write_json_replace(path: Path, values: dict[str, Any]) -> None:
    """Durably replace mutable study metadata through a same-directory temporary file."""

    # Tuning selections and preflights may be regenerated before freeze, but readers should still
    # observe either the previous complete JSON or the replacement, never partial bytes.
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    with temporary.open("x", encoding="utf-8") as output_file:
        json.dump(values, output_file, indent=2, sort_keys=True, default=str)
        output_file.write("\n")
        output_file.flush()
        os.fsync(output_file.fileno())
    os.replace(temporary, path)
