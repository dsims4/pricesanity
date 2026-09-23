"""Define corpus-independent chronological research rules and immutable resolved boundaries."""

from dataclasses import dataclass, replace
from fractions import Fraction
import math
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


PROTOCOL_VERSION = "generalized_chronological_90_10_v1"


@dataclass(frozen=True)
class ProtocolRules:
    """Corpus-independent chronological population and development rules."""

    version: str = PROTOCOL_VERSION
    test_fraction: float = 0.10
    initial_training_fraction: float = 0.50
    learning_evaluation_fraction: float = 0.10
    fold_count: int = 5
    learning_curve_fractions: tuple[float, ...] = (0.10, 0.25, 0.50, 0.75, 1.0)
    minimum_training_sessions: int = 30
    minimum_validation_sessions: int = 5
    minimum_test_sessions: int = 5
    minimum_curve_sessions: int = 3


@dataclass(frozen=True)
class BenchmarkConfig:
    """Validated corpus accounting, representation, and execution policy."""

    format_version: int
    output_root: Path
    window_length: int
    feature_columns: tuple[str, ...]
    expected_session_count: int | None
    development_session_count: int | None
    final_holdout_session_count: int | None
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
    rules: ProtocolRules


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

    # Preserve the human-readable YAML as the protocol authority, then convert every value into an
    # immutable typed representation before planning any session boundary.
    config_path = Path(path)
    with config_path.open(encoding="utf-8") as config_file:
        raw_config = yaml.safe_load(config_file)
    if not isinstance(raw_config, dict):
        raise ValueError("Benchmark configuration must be a YAML mapping.")

    try:
        # Name the three methodological sections explicitly so a malformed file fails at its source
        # rather than later as an obscure dataclass or index error.
        dataset = _require_mapping(raw_config, "dataset")
        protocol = _require_mapping(raw_config, "protocol")
        tuning = _require_mapping(raw_config, "tuning")
        if int(raw_config["format_version"]) != 2:
            raise ValueError("Benchmark configuration format 2 is required.")
        rule_values = dict(_require_mapping(protocol, "rules"))
        if "learning_curve_fractions" in rule_values:
            rule_values["learning_curve_fractions"] = tuple(
                rule_values["learning_curve_fractions"]
            )
        rules = ProtocolRules(**rule_values)
        budgets = {
            str(model_name): int(candidate_count)
            for model_name, candidate_count in _require_mapping(
                tuning, "candidate_budgets"
            ).items()
        }

        # Normalize YAML scalars and sequences at this boundary so later identity hashing and
        # comparisons do not depend on loader-specific container or numeric types.
        config = BenchmarkConfig(
            format_version=int(raw_config["format_version"]),
            output_root=Path(str(raw_config["output_root"])),
            window_length=int(dataset["window_length"]),
            feature_columns=tuple(str(value) for value in dataset["feature_columns"]),
            expected_session_count=None,
            development_session_count=None,
            final_holdout_session_count=None,
            chronological_validation_folds=(),
            learning_curve_session_counts=(),
            learning_curve_evaluation_range=(0, 0),
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
            rules=rules,
        )
    except (KeyError, TypeError, ValueError) as error:
        # Preserve deliberate benchmark validation messages; wrap raw YAML shape/type errors with
        # enough context to identify configuration loading as the failing stage.
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
    config = resolve_benchmark_config(session_count, config)
    expected_session_count = config.expected_session_count

    # Every corpus session must belong to development or the final holdout. Silent surplus or
    # missing sessions would change the experiment without changing its configuration file.
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

    # Construct the final plan only after corpus accounting succeeds, preserving configured order
    # and half-open boundaries exactly as validated.
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

    # Validate format and representation basics before reasoning about dependent split boundaries.
    if config.format_version != 2:
        raise ValueError("Benchmark configuration format 2 is required.")
    if config.window_length <= 0:
        raise ValueError("Benchmark window length must be positive.")
    if not config.feature_columns or len(set(config.feature_columns)) != len(
        config.feature_columns
    ):
        raise ValueError("Benchmark feature columns must be unique and nonempty.")
    # Operational measurements require positive worker, repetition, and safety values; zero would
    # disable evidence collection or acknowledgement gates rather than represent a valid choice.
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

    _validate_rules(config.rules)
    if config.expected_session_count is None:
        if (config.development_session_count is not None
                or config.final_holdout_session_count is not None
                or config.chronological_validation_folds
                or config.learning_curve_session_counts
                or config.learning_curve_evaluation_range != (0, 0)):
            raise ValueError("Unresolved benchmark rules cannot contain instance boundaries.")
        return
    if config.development_session_count is None or config.final_holdout_session_count is None:
        raise ValueError("Resolved benchmark counts are required.")
    if (
        config.development_session_count <= 0
        or config.final_holdout_session_count <= 0
    ):
        raise ValueError("Development and final-holdout counts must be positive.")
    if config.expected_session_count != (
        config.development_session_count + config.final_holdout_session_count
    ):
        raise ValueError("Expected sessions must equal development plus final holdout.")

    if not config.chronological_validation_folds:
        raise ValueError("Benchmark requires chronological validation folds.")

    previous_validation_end = 0

    # Folds expand from the beginning and validate strictly later history. Monotone nonoverlapping
    # validation blocks protect chronology and keep every fold inside development.
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

    # Training sizes are unique expanding prefixes. Their order is methodological evidence, not a
    # convenience that should be silently sorted during loading.
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

    # One fixed future development block evaluates every learning-curve prefix, and it begins only
    # after both the largest prefix and every tuning-validation fold.
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


def _validate_rules(rules: ProtocolRules) -> None:
    if rules.version != PROTOCOL_VERSION or rules.test_fraction != 0.10:
        raise ValueError("Benchmark requires generalized_chronological_90_10_v1 and test_fraction 0.10.")
    if not (0 < rules.initial_training_fraction < 1 - rules.learning_evaluation_fraction < 1):
        raise ValueError("Benchmark development fractions must leave later validation and evaluation blocks.")
    fractions = rules.learning_curve_fractions
    if (not fractions or tuple(sorted(set(fractions))) != fractions
            or any(not 0 < value <= 1 for value in fractions) or fractions[-1] != 1):
        raise ValueError("Benchmark learning fractions must increase uniquely through 1.0.")
    for name in ("fold_count", "minimum_training_sessions", "minimum_validation_sessions",
                 "minimum_test_sessions", "minimum_curve_sessions"):
        value = getattr(rules, name)
        minimum = 2 if name == "fold_count" else 1
        if type(value) is not int or value < minimum:
            raise ValueError(f"Benchmark {name} must be at least {minimum}.")


def _generated_boundaries(count: int, rules: ProtocolRules) -> dict[str, Any]:
    # Rational arithmetic avoids binary floating-point surprises at integer boundaries.
    test = math.ceil(count * Fraction(str(rules.test_fraction)))
    development = count - test
    evaluation_start = development - math.ceil(
        development * Fraction(str(rules.learning_evaluation_fraction))
    )
    initial = math.floor(development * Fraction(str(rules.initial_training_fraction)))
    endpoints = tuple(initial + (evaluation_start - initial) * index // rules.fold_count
                      for index in range(rules.fold_count + 1))
    return {
        "expected_session_count": count,
        "development_session_count": development,
        "final_holdout_session_count": test,
        "chronological_validation_folds": tuple(
            (start, start, end) for start, end in zip(endpoints, endpoints[1:])
        ),
        "learning_curve_session_counts": tuple(sorted({
            math.floor(evaluation_start * Fraction(str(fraction)))
            for fraction in rules.learning_curve_fractions
        })),
        "learning_curve_evaluation_range": (evaluation_start, development),
    }


def _sufficient(boundaries: dict[str, Any], rules: ProtocolRules) -> bool:
    folds = boundaries["chronological_validation_folds"]
    start, end = boundaries["learning_curve_evaluation_range"]
    return (folds[0][0] >= rules.minimum_training_sessions
            and all(b - a >= rules.minimum_validation_sessions for _, a, b in folds)
            and end - start >= rules.minimum_validation_sessions
            and boundaries["final_holdout_session_count"] >= rules.minimum_test_sessions
            and boundaries["learning_curve_session_counts"][0] >= rules.minimum_curve_sessions)


def minimum_session_count(rules: ProtocolRules) -> int:
    """Find the first corpus satisfying every required session-level population size."""
    _validate_rules(rules)
    count = 1
    while not _sufficient(_generated_boundaries(count, rules), rules):
        count += 1
    return count


def resolve_benchmark_config(session_count: int, config: BenchmarkConfig) -> BenchmarkConfig:
    """Bind rules once to an immutable corpus; never rebind an existing instance."""
    _validate_benchmark_config(config)
    if type(session_count) is not int or session_count <= 0:
        raise ValueError("Benchmark session count must be a positive integer.")
    boundaries = _generated_boundaries(session_count, config.rules)
    if not _sufficient(boundaries, config.rules):
        raise ValueError(
            f"This protocol requires at least {minimum_session_count(config.rules)} complete "
            "eligible sessions: expanding training needs "
            f"{config.rules.minimum_training_sessions}, each validation/evaluation block "
            f"{config.rules.minimum_validation_sessions}, test {config.rules.minimum_test_sessions}, "
            f"and every three-class learning prefix {config.rules.minimum_curve_sessions} sessions."
        )
    if config.expected_session_count is not None and any(
        getattr(config, key) != value for key, value in boundaries.items()
    ):
        raise ValueError("Resolved benchmark protocol boundary cannot be rebound to a different population.")
    resolved = replace(config, **boundaries)
    _validate_benchmark_config(resolved)
    return resolved


def describe_protocol(config: BenchmarkConfig) -> dict[str, Any]:
    """Public population description shared by plans, snapshots, and run reports."""
    if config.expected_session_count is None:
        raise ValueError("Resolve the corpus before describing its protocol.")
    return {
        "protocol_version": config.rules.version,
        "total_session_count": config.expected_session_count,
        "development_session_count": config.development_session_count,
        "test_session_count": config.final_holdout_session_count,
        "test_fraction": config.rules.test_fraction,
        "rounding_rule": "ceil(test_fraction * N)",
        "corpus_status": "interim",
        "chronological_validation_folds": config.chronological_validation_folds,
        "learning_curve_session_counts": config.learning_curve_session_counts,
        "learning_curve_evaluation_range": config.learning_curve_evaluation_range,
    }
