"""Run one already-selected benchmark model through common timing and metrics."""

from dataclasses import dataclass
from time import perf_counter
from statistics import median
from typing import Callable, Any

import numpy as np
import pandas as pd

from pricesanity.benchmark.metrics import (
    BenchmarkMetrics,
    EfficiencyMetrics,
    evaluate_benchmark_predictions,
)
from pricesanity.benchmark.resources import controlled_thread_budget, hardware_fingerprint
from pricesanity.features.causal_window import CausalWindowCorpus
from pricesanity.features.representations import (
    sequential_representation,
    tabular_representation,
)
from pricesanity.models.protocol import (
    BenchmarkModel,
    DualRegimeOutput,
    DualRegimePredictions,
    DualRegimeProbabilities,
    PredictionContext,
)


@dataclass(frozen=True)
class BenchmarkRunResult:
    """Model outputs and measurements before durable artifact publication."""

    output: DualRegimeOutput
    metrics: BenchmarkMetrics

    @property
    def predictions(self) -> DualRegimePredictions:
        """Retain the earlier convenient access path for authoritative classes."""

        return self.output.predictions

    @property
    def probabilities(self) -> DualRegimeProbabilities | None:
        """Return probability estimates when this model has a valid probability path."""

        return self.output.probabilities


REGIME_NAMES = np.array(["bull", "bear", "range"])


def run_model_once(
    model: BenchmarkModel,
    *,
    training: CausalWindowCorpus,
    evaluation: CausalWindowCorpus,
    representation: str,
    serialized_model_bytes: int = 0,
    parameter_count: int | None = None,
    device: str = "cpu",
    cpu_worker_count: int = 1,
    inference_timing_repetitions: int = 3,
    checkpoint_fitted: Callable[[Any, dict], None] | None = None,
) -> BenchmarkRunResult:
    """Fit on one historical partition and evaluate one later partition."""

    training_features = _representation(training, representation)
    training_context = _context(training)

    with controlled_thread_budget(cpu_worker_count):
        _synchronize_model(model)
        training_started = perf_counter()
        model.fit(
            training_features,
            training.current_targets,
            training.anticipated_targets,
            context=training_context,
        )
        _synchronize_model(model)
        training_seconds = perf_counter() - training_started

    if checkpoint_fitted is not None:
        checkpoint_fitted(model, {
            "training_seconds": training_seconds,
            "training_sample_count": len(training_features),
            "training_samples_per_second": len(training_features) / training_seconds,
            "parameter_count": int(model.parameter_count()) if hasattr(model, "parameter_count") else None,
            "hardware_fingerprint": hardware_fingerprint(device=device, cpu_worker_count=cpu_worker_count),
            "device": device, "cpu_worker_count": cpu_worker_count,
        })

    return evaluate_fitted_model(
        model,
        evaluation=evaluation,
        representation=representation,
        training_seconds=training_seconds,
        training_sample_count=len(training_features),
        serialized_model_bytes=serialized_model_bytes,
        parameter_count=parameter_count,
        device=device,
        cpu_worker_count=cpu_worker_count,
        inference_timing_repetitions=inference_timing_repetitions,
    )


def evaluate_fitted_model(
    model: BenchmarkModel,
    *,
    evaluation: CausalWindowCorpus,
    representation: str,
    training_seconds: float = 0.0,
    training_sample_count: int = 0,
    serialized_model_bytes: int = 0,
    parameter_count: int | None = None,
    device: str = "cpu",
    cpu_worker_count: int = 1,
    inference_timing_repetitions: int = 3,
) -> BenchmarkRunResult:
    """Evaluate an exact persisted fitted model without another optimization pass."""

    evaluation_features = _representation(evaluation, representation)
    evaluation_context = _context(evaluation)
    if inference_timing_repetitions <= 0 or cpu_worker_count <= 0:
        raise ValueError("Timing repetitions and CPU worker count must be positive.")
    if parameter_count is None and hasattr(model, "parameter_count"):
        parameter_count = int(model.parameter_count())

    # A combined output avoids running an expensive estimator twice. Native predictions stay
    # authoritative even when a model also exposes auxiliary probability estimates or scores.
    with controlled_thread_budget(cpu_worker_count):
        model.predict_output(evaluation_features, context=evaluation_context)
        inference_timings = []
        output = None
        for _ in range(inference_timing_repetitions):
            _synchronize_model(model)
            inference_started = perf_counter()
            output = model.predict_output(evaluation_features, context=evaluation_context)
            _synchronize_model(model)
            inference_timings.append(perf_counter() - inference_started)
    assert output is not None
    inference_seconds = median(inference_timings)
    metrics = evaluate_benchmark_predictions(
        human_current=evaluation.current_targets,
        predicted_current=output.predictions.current,
        human_anticipated=evaluation.anticipated_targets,
        predicted_anticipated=output.predictions.anticipated,
        current_probabilities=(
            output.probabilities.current if output.probabilities is not None else None
        ),
        anticipated_probabilities=(
            output.probabilities.anticipated
            if output.probabilities is not None
            else None
        ),
        session_indices=evaluation.session_indices,
        efficiency=EfficiencyMetrics(
            training_seconds=training_seconds,
            inference_seconds=inference_seconds,
            serialized_model_bytes=serialized_model_bytes,
            parameter_count=parameter_count,
            training_sample_count=training_sample_count,
            inference_sample_count=len(evaluation_features),
            training_samples_per_second=(
                training_sample_count / training_seconds if training_seconds else 0.0
            ),
            inference_samples_per_second=(
                len(evaluation_features) / inference_seconds if inference_seconds else 0.0
            ),
            device=device,
            cpu_worker_count=cpu_worker_count,
            timing_repetitions=inference_timing_repetitions,
            data_loader_worker_count=0,
            hardware_fingerprint=hardware_fingerprint(
                device=device,
                cpu_worker_count=cpu_worker_count,
            ),
        ),
    )
    return BenchmarkRunResult(
        output=output,
        metrics=metrics,
    )


def build_benchmark_prediction_frame(
    corpus: CausalWindowCorpus,
    result: BenchmarkRunResult,
) -> pd.DataFrame:
    """Join model outputs back to exact causal sample identities for persistence."""

    sample_count = len(corpus.current_targets)
    if (
        result.predictions.current.shape != (sample_count,)
        or result.predictions.anticipated.shape != (sample_count,)
        or (
            result.probabilities is not None
            and result.probabilities.current.shape != (sample_count, 3)
        )
        or (
            result.probabilities is not None
            and result.probabilities.anticipated.shape != (sample_count, 3)
        )
    ):
        raise ValueError("Benchmark outputs must align with their causal sample corpus.")
    if result.probabilities is not None and (
        not np.isfinite(result.probabilities.current).all()
        or not np.isfinite(result.probabilities.anticipated).all()
    ):
        raise ValueError("Benchmark probabilities must be finite.")

    probability_values = result.probabilities
    score_values = result.output.scores
    missing_values = np.full((sample_count, 3), np.nan, dtype=np.float64)
    current_probabilities = (
        probability_values.current if probability_values is not None else missing_values
    )
    anticipated_probabilities = (
        probability_values.anticipated if probability_values is not None else missing_values
    )
    current_scores = score_values.current if score_values is not None else missing_values
    anticipated_scores = (
        score_values.anticipated if score_values is not None else missing_values
    )

    return pd.DataFrame({
        "candlestick_id": corpus.candlestick_ids,
        "timestamp": corpus.timestamps,
        "session_date": corpus.session_dates,
        "session_index": corpus.session_indices,
        "candle_position": corpus.candle_positions,
        "predicted_current_regime": REGIME_NAMES[result.predictions.current],
        "predicted_anticipated_regime": REGIME_NAMES[
            result.predictions.anticipated
        ],
        "uncertainty_kind": result.output.uncertainty_kind,
        "current_probability_bull": current_probabilities[:, 0],
        "current_probability_bear": current_probabilities[:, 1],
        "current_probability_range": current_probabilities[:, 2],
        "anticipated_probability_bull": anticipated_probabilities[:, 0],
        "anticipated_probability_bear": anticipated_probabilities[:, 1],
        "anticipated_probability_range": anticipated_probabilities[:, 2],
        "current_score_bull": current_scores[:, 0],
        "current_score_bear": current_scores[:, 1],
        "current_score_range": current_scores[:, 2],
        "anticipated_score_bull": anticipated_scores[:, 0],
        "anticipated_score_bear": anticipated_scores[:, 1],
        "anticipated_score_range": anticipated_scores[:, 2],
        "human_current_regime": REGIME_NAMES[corpus.current_targets],
        "human_anticipated_regime": REGIME_NAMES[corpus.anticipated_targets],
    })


def _representation(corpus: CausalWindowCorpus, representation: str) -> np.ndarray:
    """Keep representation selection outside model-family control flow."""

    if representation == "tabular":
        return tabular_representation(corpus)
    if representation == "sequential":
        return sequential_representation(corpus)
    if representation in ("none", "prior_label"):
        return tabular_representation(corpus)
    raise ValueError(f"Unknown benchmark feature representation: {representation}")


def _context(corpus: CausalWindowCorpus) -> PredictionContext:
    """Carry audit identities and prior labels separately from learned market inputs."""

    return PredictionContext(
        session_indices=corpus.session_indices,
        previous_current_targets=corpus.previous_current_targets,
        previous_anticipated_targets=corpus.previous_anticipated_targets,
        valid_history_mask=corpus.valid_history_mask,
    )


def _synchronize_model(model: BenchmarkModel) -> None:
    """Wait for queued accelerator work when the adapter exposes synchronization."""

    synchronize = getattr(model, "synchronize", None)
    if callable(synchronize):
        synchronize()
