"""Define the additional full-corpus research protocol and its holdout boundary."""

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import yaml


class BenchmarkTrack(str, Enum):
    """Distinguish algorithm control from practical family-specific inputs."""

    CONTROLLED = "controlled"
    BEST_OF_FAMILY = "best_of_family"


@dataclass(frozen=True)
class SessionRange:
    """One half-open interval of chronological session positions."""

    start_index: int
    end_index: int

    @property
    def session_count(self) -> int:
        """Return the number of sessions represented by this boundary."""

        return self.end_index - self.start_index

    def indices(self) -> range:
        """Return positions without converting the half-open boundary to dates."""

        return range(self.start_index, self.end_index)


@dataclass(frozen=True)
class ChronologicalValidationFold:
    """Expanding training history followed by one future validation block."""

    training: SessionRange
    validation: SessionRange


@dataclass(frozen=True)
class BenchmarkConfig:
    """Validated defaults for a future full-corpus benchmark."""

    format_version: int
    output_root: Path
    window_length: int
    feature_columns: tuple[str, ...]
    expected_session_count: int
    development_session_count: int
    final_holdout_session_count: int
    chronological_validation_folds: tuple[tuple[int, int, int], ...]
    learning_curve_session_counts: tuple[int, ...]
    learning_curve_evaluation_range: tuple[int, int]
    controlled_first_scored_candle_position: int
    best_of_family_first_scored_candle_position: int
    final_seeds: tuple[int, ...]
    tuning_seed: int
    cpu_worker_count: int
    inference_timing_repetitions: int
    pilot_max_estimated_hours: float
    pilot_max_dense_memory_gib: float
    transformer_incumbent: dict[str, Any]
    primary_metric: str
    model_tuning_budgets: dict[str, int]


@dataclass(frozen=True)
class BenchmarkPlan:
    """Auditable development folds and untouched final history."""

    development: SessionRange
    final_holdout: SessionRange
    chronological_validation_folds: tuple[ChronologicalValidationFold, ...]
    learning_curve_session_counts: tuple[int, ...]

    def development_indices(self) -> range:
        """Expose only sessions legal for fitting and hyperparameter selection."""

        return self.development.indices()

    def holdout_indices(self, *, final_evaluation: bool = False) -> range:
        """Require an explicit final-evaluation action before revealing holdout positions."""

        if not final_evaluation:
            raise PermissionError(
                "Final holdout sessions are unavailable during tuning and development."
            )
        return self.final_holdout.indices()


def load_benchmark_config(path: str | Path) -> BenchmarkConfig:
    """Load and validate the readable benchmark research configuration."""

    config_path = Path(path)
    with config_path.open(encoding="utf-8") as config_file:
        raw_config = yaml.safe_load(config_file)
    if not isinstance(raw_config, dict):
        raise ValueError("Benchmark configuration must be a YAML mapping.")

    try:
        dataset = _require_mapping(raw_config, "dataset")
        protocol = _require_mapping(raw_config, "protocol")
        tuning = _require_mapping(raw_config, "tuning")
        fold_rows = protocol["chronological_validation_folds"]
        if not isinstance(fold_rows, list):
            raise ValueError("Chronological validation folds must be a list.")

        folds = tuple(
            (
                int(_require_mapping({"fold": fold}, "fold")["training_end"]),
                int(fold["validation_start"]),
                int(fold["validation_end"]),
            )
            for fold in fold_rows
        )
        budgets = {
            str(model_name): int(candidate_count)
            for model_name, candidate_count in _require_mapping(
                tuning, "candidate_budgets"
            ).items()
        }
        config = BenchmarkConfig(
            format_version=int(raw_config["format_version"]),
            output_root=Path(str(raw_config["output_root"])),
            window_length=int(dataset["window_length"]),
            feature_columns=tuple(str(value) for value in dataset["feature_columns"]),
            expected_session_count=int(protocol["expected_session_count"]),
            development_session_count=int(protocol["development_session_count"]),
            final_holdout_session_count=int(protocol["final_holdout_session_count"]),
            chronological_validation_folds=folds,
            learning_curve_session_counts=tuple(
                int(value) for value in protocol["learning_curve_session_counts"]
            ),
            learning_curve_evaluation_range=(
                int(protocol["learning_curve_evaluation_start"]),
                int(protocol["learning_curve_evaluation_end"]),
            ),
            controlled_first_scored_candle_position=int(
                protocol["controlled_first_scored_candle_position"]
            ),
            best_of_family_first_scored_candle_position=int(
                protocol["best_of_family_first_scored_candle_position"]
            ),
            final_seeds=tuple(int(value) for value in tuning["final_seeds"]),
            tuning_seed=int(tuning["tuning_seed"]),
            cpu_worker_count=int(tuning["cpu_worker_count"]),
            inference_timing_repetitions=int(tuning["inference_timing_repetitions"]),
            pilot_max_estimated_hours=float(tuning["pilot_max_estimated_hours"]),
            pilot_max_dense_memory_gib=float(tuning["pilot_max_dense_memory_gib"]),
            transformer_incumbent=dict(tuning["transformer_incumbent"]),
            primary_metric=str(tuning["primary_metric"]),
            model_tuning_budgets=budgets,
        )
    except (KeyError, TypeError, ValueError) as error:
        if isinstance(error, ValueError) and str(error).startswith("Benchmark"):
            raise
        raise ValueError(f"Invalid benchmark configuration: {error}") from error

    _validate_benchmark_config(config)
    return config


def plan_benchmark(
    session_count: int,
    config: BenchmarkConfig,
) -> BenchmarkPlan:
    """Build the exact configured split without silently ignoring corpus sessions."""

    # Revalidate direct dataclass callers; YAML loading is not the only public construction
    # path, and a malformed in-memory plan must not bypass the final-holdout boundary.
    _validate_benchmark_config(config)
    expected_session_count = config.expected_session_count
    if session_count != expected_session_count:
        raise ValueError(
            f"Benchmark configuration accounts for {expected_session_count} sessions, "
            f"but the corpus contains {session_count}. Update the split deliberately."
        )

    folds = tuple(
        ChronologicalValidationFold(
            training=SessionRange(0, training_end),
            validation=SessionRange(validation_start, validation_end),
        )
        for training_end, validation_start, validation_end
        in config.chronological_validation_folds
    )
    return BenchmarkPlan(
        development=SessionRange(0, config.development_session_count),
        final_holdout=SessionRange(
            config.development_session_count,
            expected_session_count,
        ),
        chronological_validation_folds=folds,
        learning_curve_session_counts=config.learning_curve_session_counts,
    )


def _require_mapping(mapping: dict[str, Any], key: str) -> dict[str, Any]:
    """Name malformed sections before dataclass construction obscures their source."""

    value = mapping[key]
    if not isinstance(value, dict):
        raise ValueError(f"Benchmark configuration section '{key}' must be a mapping.")
    return value


def _validate_benchmark_config(config: BenchmarkConfig) -> None:
    """Reject settings that could mix tuning history with the final holdout."""

    if config.format_version != 1:
        raise ValueError("Unsupported benchmark configuration format version.")
    if config.window_length <= 0:
        raise ValueError("Benchmark window length must be positive.")
    if not config.feature_columns or len(set(config.feature_columns)) != len(
        config.feature_columns
    ):
        raise ValueError("Benchmark feature columns must be unique and nonempty.")
    if (
        config.development_session_count <= 0
        or config.final_holdout_session_count <= 0
    ):
        raise ValueError("Development and final-holdout counts must be positive.")
    if config.expected_session_count != (
        config.development_session_count + config.final_holdout_session_count
    ):
        raise ValueError("Expected sessions must equal development plus final holdout.")
    if config.cpu_worker_count <= 0 or config.inference_timing_repetitions <= 0:
        raise ValueError("Worker and timing repetition counts must be positive.")
    if config.pilot_max_estimated_hours <= 0 or config.pilot_max_dense_memory_gib <= 0:
        raise ValueError("Pilot safety thresholds must be positive.")
    # The tracks deliberately have different evaluation populations: the controlled
    # experiment needs a complete 16-candle history, while best-of-family uses an explicit
    # padding mask so it can score early candles. Each boundary must be legal, but neither
    # is required to start after the other.
    if (
        config.controlled_first_scored_candle_position < 1
        or config.best_of_family_first_scored_candle_position < 1
    ):
        raise ValueError("Benchmark target-candle positions must be positive.")
    if config.primary_metric != "mean_head_macro_f1":
        raise ValueError("Benchmark selection metric must be mean_head_macro_f1.")
    if not config.final_seeds or len(set(config.final_seeds)) != len(config.final_seeds):
        raise ValueError("Final benchmark seeds must be unique and nonempty.")
    if any(candidate_count < 0 for candidate_count in config.model_tuning_budgets.values()):
        raise ValueError("Model tuning budgets cannot be negative.")

    previous_validation_end = 0
    for training_end, validation_start, validation_end in (
        config.chronological_validation_folds
    ):
        if not 0 < training_end <= validation_start < validation_end:
            raise ValueError("Each validation fold must follow its expanding training data.")
        if validation_end > config.development_session_count:
            raise ValueError("Validation folds cannot enter the final holdout.")
        if validation_start < previous_validation_end:
            raise ValueError("Chronological validation folds cannot overlap or move backward.")
        previous_validation_end = validation_end

    learning_counts = config.learning_curve_session_counts
    if (
        not learning_counts
        or tuple(sorted(set(learning_counts))) != learning_counts
        or learning_counts[-1] > config.development_session_count
        or learning_counts[0] <= 0
    ):
        raise ValueError("Learning-curve sizes must be unique development-history prefixes.")
    evaluation_start, evaluation_end = config.learning_curve_evaluation_range
    final_validation_end = max(
        validation_end
        for _, _, validation_end in config.chronological_validation_folds
    )
    if not (
        learning_counts[-1] <= evaluation_start
        and final_validation_end <= evaluation_start
        < evaluation_end
        <= config.development_session_count
    ):
        raise ValueError(
            "Learning-curve evaluation must be one fixed development block after every "
            "hyperparameter-validation fold."
        )
