"""Train and evaluate the causal market-regime Transformer."""

from collections.abc import Callable, Sequence
from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from pricesanity.training.dataset import (
    FEATURE_COLUMNS,
    IGNORED_TARGET,
    REGIME_TO_CLASS,
    TensorBatch,
    TrainingDataLoaders,
)
from pricesanity.training.model import RegimeTransformer, TransformerConfig


# Reverse the annotation encoding once so saved predictions use readable labels.
CLASS_TO_REGIME = {
    class_index: regime.value
    for regime, class_index in REGIME_TO_CLASS.items()
}
REGIME_CLASS_BY_NAME = {
    regime.value: class_index
    for regime, class_index in REGIME_TO_CLASS.items()
}


@dataclass(frozen=True)
class FeatureStandardizer:
    """Training-only location and scale for each candle feature."""

    mean: torch.Tensor
    standard_deviation: torch.Tensor

    def to(self, device: torch.device) -> "FeatureStandardizer":
        """Move the frozen statistics once for repeated batches on one device."""

        return FeatureStandardizer(
            mean=self.mean.to(device=device),
            standard_deviation=self.standard_deviation.to(device=device),
        )

    def transform(self, features: torch.Tensor) -> torch.Tensor:
        """Standardize features without changing their batch or time axes.

        Args:
            features: Floating-point candle features with the feature axis last.

        Returns:
            Features centered and scaled by training-corpus statistics.

        Raises:
            ValueError: If the feature dimension or tensor type is incompatible.
        """

        if features.ndim < 2 or features.shape[-1] != len(FEATURE_COLUMNS):
            raise ValueError("Features must end with the configured feature dimension.")
        if not torch.is_floating_point(features):
            raise ValueError("Features must use a floating-point tensor type.")

        # Training has already moved these statistics once. Tensor.to returns the same tensor
        # when device and dtype match; this fallback keeps standalone CPU/MPS callers safe.
        feature_mean = self.mean.to(device=features.device, dtype=features.dtype)
        feature_scale = self.standard_deviation.to(
            device=features.device,
            dtype=features.dtype,
        )
        return (features - feature_mean) / feature_scale


@dataclass(frozen=True)
class DualRegimeLoss:
    """Separate and equally weighted losses for the two annotation targets."""

    total: torch.Tensor
    current: torch.Tensor
    anticipated: torch.Tensor


@dataclass(frozen=True)
class ClassificationMetrics:
    """Class-balanced and overall measurements for one prediction head."""

    accuracy: float
    macro_f1: float
    support: int
    class_support: tuple[int, ...]
    confusion_matrix: tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class EpochMetrics:
    """Losses and classification measurements across one dataset pass."""

    total_loss: float
    current_loss: float
    anticipated_loss: float
    current: ClassificationMetrics
    anticipated: ClassificationMetrics


@dataclass(frozen=True)
class EpochRecord:
    """Training and validation evidence retained for one epoch."""

    epoch: int
    training: EpochMetrics
    validation: EpochMetrics


@dataclass(frozen=True)
class TrainingLoopConfig:
    """Optimization settings for one reproducible model fit."""

    epochs: int = 50
    learning_rate: float = 3e-4
    weight_decay: float = 1e-2
    gradient_clip: float = 1.0
    early_stopping_patience: int = 10


@dataclass(frozen=True)
class TrainingResult:
    """Best validation epoch, learning history, and untouched test result."""

    best_epoch: int
    history: tuple[EpochRecord, ...]
    test: EpochMetrics
    majority_current: ClassificationMetrics
    majority_anticipated: ClassificationMetrics
    test_predictions: pd.DataFrame
    model_state_sha256: str


@dataclass(frozen=True)
class MajorityClasses:
    """Training-only majority labels used by the honest comparison baseline."""

    current: int
    anticipated: int


@dataclass(frozen=True)
class LoadedCheckpoint:
    """Validated inference model, feature scaling, and saved experiment metadata."""

    model: RegimeTransformer
    standardizer: FeatureStandardizer
    metadata: dict[str, Any]


def fit_feature_standardizer(
    training_sessions: Sequence[pd.DataFrame],
) -> FeatureStandardizer:
    """Fit feature statistics using training sessions only.

    Args:
        training_sessions: Earliest annotated sessions assigned to model fitting.

    Returns:
        Per-feature mean and standard deviation learned without evaluation data.

    Raises:
        ValueError: If no finite training features are available.
    """

    if not training_sessions:
        raise ValueError("Feature standardization requires training sessions.")

    # Fit on training candles only. Even unlabeled validation or test prices would leak
    # future distribution information into every standardized training input.
    feature_arrays = []
    for session in training_sessions:
        missing_columns = set(FEATURE_COLUMNS).difference(session.columns)
        if missing_columns:
            raise ValueError(
                "Training session is missing feature columns: "
                + ", ".join(sorted(missing_columns))
            )

        # Copy each complete session into one common training-only calculation.
        feature_arrays.append(
            session.loc[:, list(FEATURE_COLUMNS)].to_numpy(
                dtype=np.float32,
                copy=True,
            )
        )

    training_features = np.concatenate(feature_arrays, axis=0)
    if training_features.size == 0 or not np.isfinite(training_features).all():
        raise ValueError("Training features must contain finite values.")

    feature_tensor = torch.from_numpy(training_features)
    feature_mean = feature_tensor.mean(dim=0)
    feature_standard_deviation = feature_tensor.std(dim=0, correction=0)

    # A constant feature contains no scale information; dividing it by one
    # preserves its centered zero value without producing infinities.
    minimum_scale = torch.finfo(feature_tensor.dtype).eps
    feature_standard_deviation = torch.where(
        feature_standard_deviation > minimum_scale,
        feature_standard_deviation,
        torch.ones_like(feature_standard_deviation),
    )

    return FeatureStandardizer(
        mean=feature_mean,
        standard_deviation=feature_standard_deviation,
    )


def fit_majority_classes(
    training_sessions: Sequence[pd.DataFrame],
) -> MajorityClasses:
    """Find the most frequent target for each head using training labels only.

    Args:
        training_sessions: Sessions allowed to define the comparison baseline.

    Returns:
        Deterministically selected current and anticipated majority classes.

    Raises:
        ValueError: If training targets are missing or invalid.
    """

    if not training_sessions:
        raise ValueError("Majority classes require training sessions.")

    valid_classes = set(REGIME_TO_CLASS.values())
    majority_classes = []
    for target_column in ("current_target", "anticipated_target"):
        target_values = []
        for session in training_sessions:
            if target_column not in session.columns:
                raise ValueError(
                    f"Training session is missing target column: {target_column}"
                )
            target_values.extend(session[target_column].tolist())

        if not target_values or not set(target_values).issubset(valid_classes):
            raise ValueError("Training targets must contain valid regime classes.")

        # Bincount returns the lowest class index when frequencies tie, making
        # the baseline stable across platforms and annotation retrieval order.
        class_counts = np.bincount(
            np.asarray(target_values, dtype=np.int64),
            minlength=len(valid_classes),
        )
        majority_classes.append(int(class_counts.argmax()))

    return MajorityClasses(
        current=majority_classes[0],
        anticipated=majority_classes[1],
    )


def calculate_dual_regime_loss(
    current_logits: torch.Tensor,
    anticipated_logits: torch.Tensor,
    current_targets: torch.Tensor,
    anticipated_targets: torch.Tensor,
) -> DualRegimeLoss:
    """Calculate equally weighted classification loss for both judgments.

    Args:
        current_logits: Current-regime class scores shaped by batch, time, and class.
        anticipated_logits: Anticipated-regime scores with the same shape.
        current_targets: Current-regime class indices shaped by batch and time.
        anticipated_targets: Anticipated-regime indices with the same shape.

    Returns:
        Combined and per-head cross-entropy losses.

    Raises:
        ValueError: If shapes differ or no real target remains after padding.
    """

    if current_logits.shape != anticipated_logits.shape:
        raise ValueError("Both prediction heads must return the same shape.")
    if current_logits.ndim != 3:
        raise ValueError("Regime logits must have batch, time, and class axes.")
    if current_targets.shape != current_logits.shape[:2]:
        raise ValueError("Current targets must match the logit batch and time axes.")
    if anticipated_targets.shape != anticipated_logits.shape[:2]:
        raise ValueError("Anticipated targets must match the logit batch and time axes.")

    real_current_targets = current_targets.ne(IGNORED_TARGET)
    real_anticipated_targets = anticipated_targets.ne(IGNORED_TARGET)
    if not real_current_targets.any() or not real_anticipated_targets.any():
        raise ValueError("Regime loss requires at least one real target per head.")

    regime_count = current_logits.shape[-1]

    # Flatten only the batch and time axes because cross-entropy expects one
    # row of class scores for every target class index.
    current_loss = torch.nn.functional.cross_entropy(
        current_logits.reshape(-1, regime_count),
        current_targets.reshape(-1),
        ignore_index=IGNORED_TARGET,
    )
    anticipated_loss = torch.nn.functional.cross_entropy(
        anticipated_logits.reshape(-1, regime_count),
        anticipated_targets.reshape(-1),
        ignore_index=IGNORED_TARGET,
    )

    # Equal averaging keeps both human judgments equally important while
    # preserving a loss scale comparable with one classification task.
    total_loss = (current_loss + anticipated_loss) / 2.0
    return DualRegimeLoss(
        total=total_loss,
        current=current_loss,
        anticipated=anticipated_loss,
    )


class _ClassificationAccumulator:
    """Accumulate an exact confusion matrix across variable-sized batches."""

    def __init__(self, regime_count: int) -> None:
        self._regime_count = regime_count
        self._confusion_matrix = torch.zeros(
            regime_count,
            regime_count,
            dtype=torch.int64,
        )

    def update(self, logits: torch.Tensor, targets: torch.Tensor) -> None:
        """Add predictions from every nonpadding target in one batch."""

        real_positions = targets.ne(IGNORED_TARGET)
        real_targets = targets[real_positions].detach().to(device="cpu")
        real_predictions = (
            logits.argmax(dim=-1)[real_positions].detach().to(device="cpu")
        )

        self.update_predictions(real_predictions, real_targets)

    def update_predictions(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> None:
        """Add already-decoded class predictions and real targets."""

        # A flattened target/prediction pair gives bincount one unique index
        # for every cell of the regime-by-regime confusion matrix.
        cell_indices = targets * self._regime_count + predictions
        batch_confusion = torch.bincount(
            cell_indices,
            minlength=self._regime_count**2,
        ).reshape(self._regime_count, self._regime_count)
        self._confusion_matrix += batch_confusion

    def calculate(self) -> ClassificationMetrics:
        """Convert accumulated counts into accuracy and macro-F1."""

        total = int(self._confusion_matrix.sum().item())
        if total == 0:
            raise ValueError("Classification metrics require at least one real target.")

        true_positives = self._confusion_matrix.diag().to(torch.float64)
        actual_counts = self._confusion_matrix.sum(dim=1).to(torch.float64)
        predicted_counts = self._confusion_matrix.sum(dim=0).to(torch.float64)

        # Compute each class independently so a frequent range label cannot
        # conceal poor bull or bear recognition in the final average.
        f1_scores = []
        for class_index in range(self._regime_count):
            numerator = 2.0 * true_positives[class_index]
            denominator = actual_counts[class_index] + predicted_counts[class_index]
            class_f1 = (
                float((numerator / denominator).item())
                if denominator.item() > 0
                else 0.0
            )
            f1_scores.append(class_f1)

        accuracy = float(true_positives.sum().item() / total)
        confusion_matrix = tuple(
            tuple(int(value) for value in row)
            for row in self._confusion_matrix.tolist()
        )
        return ClassificationMetrics(
            accuracy=accuracy,
            macro_f1=sum(f1_scores) / self._regime_count,
            support=total,
            class_support=tuple(int(value) for value in actual_counts.tolist()),
            confusion_matrix=confusion_matrix,
        )


def resolve_training_device(requested_device: str = "auto") -> torch.device:
    """Select an available training device without silently changing an explicit choice.

    Args:
        requested_device: One of auto, cpu, cuda, or mps.

    Returns:
        Available PyTorch device selected for training.

    Raises:
        ValueError: If the requested device name is unknown or unavailable.
    """

    # Keep accepted names explicit so a typo cannot silently select CPU and falsify device-specific
    # timing or reproducibility evidence.
    supported_devices = {"auto", "cpu", "cuda", "mps"}
    if requested_device not in supported_devices:
        raise ValueError(
            "Training device must be one of: " + ", ".join(sorted(supported_devices))
        )

    if requested_device == "auto":
        # Prefer accelerator backends in a deterministic order, with CPU as the portable fallback
        # only when the caller explicitly allowed automatic selection.
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    if requested_device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available.")
    if requested_device == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS was requested but is not available.")
    return torch.device(requested_device)


def _run_epoch(
    model: RegimeTransformer,
    data_loader: torch.utils.data.DataLoader,
    standardizer: FeatureStandardizer,
    *,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    gradient_clip: float,
    loss_setup: str = "both",
) -> EpochMetrics:
    """Run one training or evaluation pass over a complete partition."""

    # Single-head diagnostics isolate task interference; ordinary training keeps both heads
    # equally weighted. This choice never changes targets or makes labels into model inputs.
    if loss_setup not in {"both", "current", "anticipated"}:
        raise ValueError("Loss setup must be both, current, or anticipated.")

    # The optimizer's presence is the single mode switch: evaluation disables dropout and gradient
    # updates while preserving the identical forward and loss calculations.
    is_training = optimizer is not None
    model.train(is_training)

    # Accumulate confusion counts and candle-weighted losses across variable-length session batches
    # so padding and short days do not receive metric or loss weight.
    current_metrics = _ClassificationAccumulator(model.config.regime_count)
    anticipated_metrics = _ClassificationAccumulator(model.config.regime_count)
    weighted_total_loss = 0.0
    weighted_current_loss = 0.0
    weighted_anticipated_loss = 0.0
    real_candle_count = 0

    # Inference mode goes beyond disabling gradients: it also prevents PyTorch
    # from retaining version-tracking state that evaluation can never use.
    gradient_context = (
        torch.enable_grad() if is_training else torch.inference_mode()
    )
    with gradient_context:
        for batch in data_loader:
            if not isinstance(batch, TensorBatch):
                raise TypeError("Training DataLoaders must return TensorBatch values.")

            # Move only tensors required by the model and loss. Ragged audit metadata remains on CPU
            # and outside learned inputs.
            features = standardizer.transform(batch.features.to(device))
            padding_mask = batch.padding_mask.to(device)
            current_targets = batch.current_targets.to(device)
            anticipated_targets = batch.anticipated_targets.to(device)

            if optimizer is not None:
                # Clearing gradients to None avoids carrying stale buffers between independent
                # optimization steps.
                optimizer.zero_grad(set_to_none=True)

            output = model(features, padding_mask)
            losses = calculate_dual_regime_loss(
                output.current_logits,
                output.anticipated_logits,
                current_targets,
                anticipated_targets,
            )

            if loss_setup != "both":
                # Single-head diagnostics optimize one chosen task while retaining both losses and
                # predictions for transparent interference measurement.
                losses = DualRegimeLoss(
                    total=getattr(losses, loss_setup),
                    current=losses.current,
                    anticipated=losses.anticipated,
                )

            if optimizer is not None:
                losses.total.backward()

                # Bound unusually large updates before they can destabilize the
                # small model's shared Transformer representation.
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                optimizer.step()

            # Weight batch means by real candle count so the epoch average is invariant to how
            # sessions of different lengths happen to be grouped into batches.
            batch_real_candle_count = int(batch.lengths.sum().item())
            real_candle_count += batch_real_candle_count
            weighted_total_loss += losses.total.item() * batch_real_candle_count
            weighted_current_loss += losses.current.item() * batch_real_candle_count
            weighted_anticipated_loss += (
                losses.anticipated.item() * batch_real_candle_count
            )

            current_metrics.update(output.current_logits, current_targets)
            anticipated_metrics.update(
                output.anticipated_logits,
                anticipated_targets,
            )

    if real_candle_count == 0:
        raise ValueError("An epoch requires at least one real candlestick.")

    return EpochMetrics(
        total_loss=weighted_total_loss / real_candle_count,
        current_loss=weighted_current_loss / real_candle_count,
        anticipated_loss=weighted_anticipated_loss / real_candle_count,
        current=current_metrics.calculate(),
        anticipated=anticipated_metrics.calculate(),
    )


def calculate_model_state_sha256(model: RegimeTransformer) -> str:
    """Identify the exact learned weights that generated an evaluation artifact."""

    # Sort parameter names and hash contiguous CPU bytes so identity is independent of state-dict
    # iteration order and the device that held the fitted model.
    state_hash = hashlib.sha256()
    for parameter_name, parameter_value in sorted(model.state_dict().items()):
        state_hash.update(parameter_name.encode("utf-8"))
        state_hash.update(
            parameter_value.detach().to(device="cpu").contiguous().numpy().tobytes()
        )
    return state_hash.hexdigest()


def _evaluate_test_once(
    model: RegimeTransformer,
    data_loader: torch.utils.data.DataLoader,
    standardizer: FeatureStandardizer,
    *,
    device: torch.device,
    majority_classes: MajorityClasses,
    run_index: int,
    model_state_sha256: str,
) -> tuple[
    EpochMetrics,
    ClassificationMetrics,
    ClassificationMetrics,
    pd.DataFrame,
]:
    """Measure selected weights and retain every real causal test prediction."""

    model.eval()

    # Keep learned-model and training-only majority metrics side by side on the identical test rows.
    current_metrics = _ClassificationAccumulator(model.config.regime_count)
    anticipated_metrics = _ClassificationAccumulator(model.config.regime_count)
    majority_current_metrics = _ClassificationAccumulator(model.config.regime_count)
    majority_anticipated_metrics = _ClassificationAccumulator(
        model.config.regime_count
    )
    weighted_total_loss = 0.0
    weighted_current_loss = 0.0
    weighted_anticipated_loss = 0.0
    real_candle_count = 0
    prediction_rows: list[dict[str, Any]] = []

    # This is the only official test pass in a run. It occurs after the best
    # validation checkpoint is restored, and inference mode makes weight updates impossible.
    with torch.inference_mode():
        for batch in data_loader:
            if not isinstance(batch, TensorBatch):
                raise TypeError("Test DataLoader must return TensorBatch values.")

            features = standardizer.transform(batch.features.to(device))
            padding_mask = batch.padding_mask.to(device)
            current_targets = batch.current_targets.to(device)
            anticipated_targets = batch.anticipated_targets.to(device)
            output = model(features, padding_mask)
            losses = calculate_dual_regime_loss(
                output.current_logits,
                output.anticipated_logits,
                current_targets,
                anticipated_targets,
            )

            # Mirror epoch loss weighting so reported test loss describes real candles rather than
            # padded batch width.
            batch_real_candle_count = int(batch.lengths.sum().item())
            real_candle_count += batch_real_candle_count
            weighted_total_loss += losses.total.item() * batch_real_candle_count
            weighted_current_loss += losses.current.item() * batch_real_candle_count
            weighted_anticipated_loss += (
                losses.anticipated.item() * batch_real_candle_count
            )

            current_metrics.update(output.current_logits, current_targets)
            anticipated_metrics.update(
                output.anticipated_logits,
                anticipated_targets,
            )

            # Labels already exist on CPU in the collated batch. Reuse them for reporting
            # instead of downloading the same targets from MPS after uploading them for loss.
            real_current_targets = batch.current_targets[
                batch.current_targets.ne(IGNORED_TARGET)
            ]
            real_anticipated_targets = batch.anticipated_targets[
                batch.anticipated_targets.ne(IGNORED_TARGET)
            ]
            majority_current_metrics.update_predictions(
                torch.full_like(real_current_targets, majority_classes.current),
                real_current_targets,
            )
            majority_anticipated_metrics.update_predictions(
                torch.full_like(
                    real_anticipated_targets,
                    majority_classes.anticipated,
                ),
                real_anticipated_targets,
            )

            current_probabilities = output.current_logits.softmax(dim=-1).to(
                device="cpu"
            )
            anticipated_probabilities = output.anticipated_logits.softmax(
                dim=-1
            ).to(device="cpu")
            current_predictions = current_probabilities.argmax(dim=-1)
            anticipated_predictions = anticipated_probabilities.argmax(dim=-1)
            current_targets_cpu = batch.current_targets
            anticipated_targets_cpu = batch.anticipated_targets

            # Decode probabilities and classes on CPU before constructing audit rows; this avoids
            # repeated device transfers inside the per-candle metadata loop.
            # Metadata is ragged by design, so each stored row stops at the
            # session's real length and can never serialize a padded prediction.
            for batch_position, session_length_value in enumerate(batch.lengths):
                session_length = int(session_length_value.item())
                for candle_position in range(session_length):
                    current_class = int(
                        current_predictions[batch_position, candle_position].item()
                    )
                    anticipated_class = int(
                        anticipated_predictions[batch_position, candle_position].item()
                    )
                    human_current_class = int(
                        current_targets_cpu[batch_position, candle_position].item()
                    )
                    human_anticipated_class = int(
                        anticipated_targets_cpu[batch_position, candle_position].item()
                    )
                    prediction_rows.append(
                        {
                            "run_index": run_index,
                            "model_state_sha256": model_state_sha256,
                            "candlestick_id": batch.candlestick_ids[batch_position][
                                candle_position
                            ],
                            "timestamp": batch.timestamps[batch_position][candle_position],
                            "session_date": batch.session_dates[batch_position],
                            "predicted_current_regime": CLASS_TO_REGIME[current_class],
                            "predicted_anticipated_regime": CLASS_TO_REGIME[
                                anticipated_class
                            ],
                            "current_probability_bull": float(
                                current_probabilities[
                                    batch_position,
                                    candle_position,
                                    REGIME_CLASS_BY_NAME["bull"],
                                ].item()
                            ),
                            "current_probability_bear": float(
                                current_probabilities[
                                    batch_position,
                                    candle_position,
                                    REGIME_CLASS_BY_NAME["bear"],
                                ].item()
                            ),
                            "current_probability_range": float(
                                current_probabilities[
                                    batch_position,
                                    candle_position,
                                    REGIME_CLASS_BY_NAME["range"],
                                ].item()
                            ),
                            "anticipated_probability_bull": float(
                                anticipated_probabilities[
                                    batch_position,
                                    candle_position,
                                    REGIME_CLASS_BY_NAME["bull"],
                                ].item()
                            ),
                            "anticipated_probability_bear": float(
                                anticipated_probabilities[
                                    batch_position,
                                    candle_position,
                                    REGIME_CLASS_BY_NAME["bear"],
                                ].item()
                            ),
                            "anticipated_probability_range": float(
                                anticipated_probabilities[
                                    batch_position,
                                    candle_position,
                                    REGIME_CLASS_BY_NAME["range"],
                                ].item()
                            ),
                            "human_current_regime": CLASS_TO_REGIME[
                                human_current_class
                            ],
                            "human_anticipated_regime": CLASS_TO_REGIME[
                                human_anticipated_class
                            ],
                        }
                    )

    if real_candle_count == 0:
        raise ValueError("Test evaluation requires at least one real candlestick.")

    # The row-level artifact must contain every real test candle exactly once and in chronology;
    # aggregate metrics alone could not reveal dropped or duplicated identities.
    test_predictions = pd.DataFrame(prediction_rows)
    if len(test_predictions) != real_candle_count:
        raise RuntimeError("Test prediction rows do not match real test candles.")
    if test_predictions["candlestick_id"].duplicated().any():
        raise RuntimeError("Test predictions contain duplicate candlestick identifiers.")
    if not test_predictions["timestamp"].is_monotonic_increasing:
        raise RuntimeError("Test predictions must remain chronological.")

    test_metrics = EpochMetrics(
        total_loss=weighted_total_loss / real_candle_count,
        current_loss=weighted_current_loss / real_candle_count,
        anticipated_loss=weighted_anticipated_loss / real_candle_count,
        current=current_metrics.calculate(),
        anticipated=anticipated_metrics.calculate(),
    )
    return (
        test_metrics,
        majority_current_metrics.calculate(),
        majority_anticipated_metrics.calculate(),
        test_predictions,
    )


def train_regime_transformer(
    model: RegimeTransformer,
    data_loaders: TrainingDataLoaders,
    standardizer: FeatureStandardizer,
    *,
    config: TrainingLoopConfig,
    device: torch.device,
    majority_classes: MajorityClasses,
    run_index: int = 1,
    epoch_callback: Callable[[EpochRecord], None] | None = None,
) -> TrainingResult:
    """Fit the model, select by validation loss, then evaluate test data once.

    Args:
        model: Untrained causal Transformer.
        data_loaders: Chronological training, validation, and test loaders.
        standardizer: Feature statistics fitted from training sessions only.
        config: Optimizer and stopping settings.
        device: CPU, CUDA, or MPS device used for computation.
        majority_classes: Training-only majority labels used as test baselines.
        run_index: One-based walk-forward experiment number saved with predictions.
        epoch_callback: Optional receiver for completed epoch measurements.

    Returns:
        Best validation history and final untouched test measurements.

    Raises:
        ValueError: If optimization settings cannot define a valid training run.
    """

    # Validate the complete optimization contract before moving model state or constructing an
    # optimizer, so invalid runs leave no partially initialized training side effects.
    if config.epochs <= 0:
        raise ValueError("Training epochs must be positive.")
    if config.learning_rate <= 0.0:
        raise ValueError("Training learning rate must be positive.")
    if config.weight_decay < 0.0:
        raise ValueError("Training weight decay cannot be negative.")
    if config.gradient_clip <= 0.0:
        raise ValueError("Training gradient clip must be positive.")
    if config.early_stopping_patience <= 0:
        raise ValueError("Early-stopping patience must be positive.")
    if run_index <= 0:
        raise ValueError("Training run index must be positive.")

    # Move model and frozen train-only statistics once, then create one optimizer whose state spans
    # the complete epoch loop.
    model.to(device)
    active_standardizer = standardizer.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    # Validation loss owns checkpoint selection. The best state is copied in memory so later epochs
    # cannot mutate the weights that will receive the one official test evaluation.
    history = []
    best_epoch = 0
    best_validation_loss = float("inf")
    best_model_state: dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0

    for epoch in range(1, config.epochs + 1):
        # Training updates weights first; validation immediately measures that epoch on later
        # chronological sessions without optimizer access.
        training_metrics = _run_epoch(
            model,
            data_loaders.training,
            active_standardizer,
            device=device,
            optimizer=optimizer,
            gradient_clip=config.gradient_clip,
        )
        # Validation may choose the epoch we keep, but receives no optimizer and cannot
        # update weights. Its feature statistics remain those fitted on training sessions.
        validation_metrics = _run_epoch(
            model,
            data_loaders.validation,
            active_standardizer,
            device=device,
            optimizer=None,
            gradient_clip=config.gradient_clip,
        )
        epoch_record = EpochRecord(
            epoch=epoch,
            training=training_metrics,
            validation=validation_metrics,
        )
        history.append(epoch_record)

        if epoch_callback is not None:
            # Emit only completed epoch evidence, allowing callers to persist progress without
            # influencing model selection.
            epoch_callback(epoch_record)

        # Validation loss chooses the checkpoint without inspecting test labels.
        if validation_metrics.total_loss < best_validation_loss:
            best_validation_loss = validation_metrics.total_loss
            best_epoch = epoch
            best_model_state = deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= config.early_stopping_patience:
            # Patience depends only on validation history; test labels remain unopened.
            break

    if best_model_state is None:
        raise RuntimeError("Training did not produce a validation checkpoint.")

    # Test only the validation-selected weights after every training decision is final.
    model.load_state_dict(best_model_state)
    model_state_sha256 = calculate_model_state_sha256(model)
    (
        test_metrics,
        majority_current,
        majority_anticipated,
        test_predictions,
    ) = _evaluate_test_once(
        model,
        data_loaders.test,
        active_standardizer,
        device=device,
        majority_classes=majority_classes,
        run_index=run_index,
        model_state_sha256=model_state_sha256,
    )
    return TrainingResult(
        best_epoch=best_epoch,
        history=tuple(history),
        test=test_metrics,
        majority_current=majority_current,
        majority_anticipated=majority_anticipated,
        test_predictions=test_predictions,
        model_state_sha256=model_state_sha256,
    )


def save_training_checkpoint(
    checkpoint_path: str | Path,
    model: RegimeTransformer,
    standardizer: FeatureStandardizer,
    training_config: TrainingLoopConfig,
    result: TrainingResult,
    *,
    experiment_metadata: dict[str, Any] | None = None,
    overwrite: bool = False,
) -> None:
    """Atomically save the selected model and reproducibility metadata.

    Args:
        checkpoint_path: Destination for the ignored local PyTorch checkpoint.
        model: Validation-selected model containing the weights to save.
        standardizer: Training-only feature statistics required for inference.
        training_config: Optimization settings used for the completed run.
        result: Epoch history and final test measurements.
        experiment_metadata: Auditable session boundaries and artifact identities.
        overwrite: Whether an existing checkpoint may be replaced.

    Raises:
        FileExistsError: If the destination exists without overwrite permission.
        ValueError: If the destination does not use the .pt suffix.
    """

    # Suffix and overwrite policy distinguish deliberate checkpoint publication from accidental
    # replacement of an existing experiment artifact.
    checkpoint_path = Path(checkpoint_path)
    if checkpoint_path.suffix.lower() != ".pt":
        raise ValueError("Training checkpoints must use the .pt suffix.")
    if checkpoint_path.exists() and not overwrite:
        raise FileExistsError(
            f"Checkpoint already exists: {checkpoint_path}. Enable overwrite to replace it."
        )

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")

    # Bind the checkpoint to the exact weights used for official test predictions. Any mutation
    # after evaluation invalidates publication.
    if calculate_model_state_sha256(model) != result.model_state_sha256:
        raise ValueError("Model weights changed after the official test evaluation.")

    best_validation = result.history[result.best_epoch - 1].validation

    # Store only tensors and basic Python values so the artifact can be inspected
    # without importing private training classes during a later model load.
    checkpoint: dict[str, Any] = {
        "model_state": {
            name: value.detach().to(device="cpu")
            for name, value in model.state_dict().items()
        },
        "model_config": asdict(model.config),
        "training_config": asdict(training_config),
        "feature_columns": list(FEATURE_COLUMNS),
        "class_mapping": {
            regime.value: class_index
            for regime, class_index in REGIME_TO_CLASS.items()
        },
        "feature_mean": standardizer.mean.detach().to(device="cpu"),
        "feature_standard_deviation": standardizer.standard_deviation.detach().to(
            device="cpu"
        ),
        "best_epoch": result.best_epoch,
        "best_validation": asdict(best_validation),
        "history": [asdict(record) for record in result.history],
        "test": asdict(result.test),
        "majority_current": asdict(result.majority_current),
        "majority_anticipated": asdict(result.majority_anticipated),
        "model_state_sha256": result.model_state_sha256,
        "experiment": experiment_metadata or {},
    }

    try:
        # Serialize beside the destination and rename only after success so readers never observe a
        # partially written canonical checkpoint.
        torch.save(checkpoint, temporary_path)
        temporary_path.replace(checkpoint_path)
    finally:
        # A failed serialization cannot leave a file that resembles a complete checkpoint.
        temporary_path.unlink(missing_ok=True)


def save_test_predictions(
    prediction_path: str | Path,
    test_predictions: pd.DataFrame,
    *,
    overwrite: bool = False,
) -> None:
    """Atomically persist ordered real-candle predictions as Parquet.

    Args:
        prediction_path: Destination for one run's test predictions.
        test_predictions: Validated rows produced during official test evaluation.
        overwrite: Whether an existing prediction artifact may be replaced.
    """

    prediction_path = Path(prediction_path)
    if prediction_path.suffix.lower() != ".parquet":
        raise ValueError("Test prediction artifacts must use the .parquet suffix.")
    if prediction_path.exists() and not overwrite:
        raise FileExistsError(
            f"Predictions already exist: {prediction_path}. Enable overwrite to replace them."
        )

    # Identity, both predictions, and both human labels are the minimum auditable row contract for
    # later read-only result views.
    required_columns = {
        "run_index",
        "model_state_sha256",
        "candlestick_id",
        "timestamp",
        "session_date",
        "predicted_current_regime",
        "predicted_anticipated_regime",
        "human_current_regime",
        "human_anticipated_regime",
    }
    missing_columns = required_columns.difference(test_predictions.columns)
    if missing_columns:
        raise ValueError(
            "Test predictions are missing columns: "
            + ", ".join(sorted(missing_columns))
        )
    if test_predictions.empty:
        raise ValueError("Test prediction artifact cannot be empty.")
    if test_predictions["candlestick_id"].duplicated().any():
        raise ValueError("Test prediction candlestick identifiers must be unique.")

    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = prediction_path.with_suffix(".partial.parquet")
    try:
        # Parquet publication follows the same temporary-then-replace rule as model checkpoints.
        test_predictions.to_parquet(temporary_path, index=False)
        temporary_path.replace(prediction_path)
    finally:
        temporary_path.unlink(missing_ok=True)


def load_training_checkpoint(
    checkpoint_path: str | Path,
    *,
    device: torch.device,
    expected_feature_columns: Sequence[str] = FEATURE_COLUMNS,
) -> LoadedCheckpoint:
    """Validate and reconstruct one checkpoint for read-only inference.

    Args:
        checkpoint_path: Saved Price Sanity `.pt` artifact.
        device: Device receiving the reconstructed evaluation model.
        expected_feature_columns: Exact input order supplied by the caller.

    Returns:
        Evaluation model, frozen feature scaling, and experiment metadata.

    Raises:
        ValueError: If required metadata, features, classes, or weights are incompatible.
    """

    # Always deserialize onto CPU first. Device movement happens only after schema, architecture,
    # scaling, class mapping, and model-state identity all verify.
    checkpoint_path = Path(checkpoint_path)
    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
    except (OSError, RuntimeError) as error:
        raise ValueError(f"Could not load checkpoint: {checkpoint_path}") from error

    required_fields = {
        "model_state",
        "model_config",
        "training_config",
        "feature_columns",
        "class_mapping",
        "feature_mean",
        "feature_standard_deviation",
        "best_epoch",
        "best_validation",
        "history",
        "test",
        "model_state_sha256",
        "experiment",
    }
    # The checkpoint is self-contained: learned weights, preprocessing, architecture, selection
    # history, evaluation evidence, and experiment identity must travel together.
    if not isinstance(checkpoint, dict):
        raise ValueError("Checkpoint must contain a named metadata mapping.")
    missing_fields = required_fields.difference(checkpoint)
    if missing_fields:
        raise ValueError(
            "Checkpoint is missing required fields: "
            + ", ".join(sorted(missing_fields))
        )

    # Column order is part of the model contract because tensors no longer
    # retain DataFrame names after conversion.
    if list(checkpoint["feature_columns"]) != list(expected_feature_columns):
        raise ValueError("Checkpoint feature order does not match model input features.")

    expected_class_mapping = {
        regime.value: class_index
        for regime, class_index in REGIME_TO_CLASS.items()
    }
    # Class order fixes the meaning of both output-head columns; matching output width alone cannot
    # detect a reordered annotation vocabulary.
    if checkpoint["class_mapping"] != expected_class_mapping:
        raise ValueError("Checkpoint class mapping is incompatible with annotations.")

    try:
        model_config = TransformerConfig(**checkpoint["model_config"])
    except (TypeError, ValueError) as error:
        raise ValueError("Checkpoint model configuration is invalid.") from error
    if model_config.feature_count != len(expected_feature_columns):
        raise ValueError("Checkpoint feature count does not match its feature order.")
    if model_config.regime_count != len(expected_class_mapping):
        raise ValueError("Checkpoint output size does not match its class mapping.")

    feature_mean = checkpoint["feature_mean"]
    feature_standard_deviation = checkpoint["feature_standard_deviation"]
    expected_statistic_shape = (len(expected_feature_columns),)
    # Scaling must provide one finite positive value per ordered input feature before inference can
    # reproduce training-time preprocessing.
    if (
        not isinstance(feature_mean, torch.Tensor)
        or not isinstance(feature_standard_deviation, torch.Tensor)
        or feature_mean.shape != expected_statistic_shape
        or feature_standard_deviation.shape != expected_statistic_shape
        or not torch.isfinite(feature_mean).all()
        or not torch.isfinite(feature_standard_deviation).all()
        or not feature_standard_deviation.gt(0).all()
    ):
        raise ValueError("Checkpoint feature-standardization statistics are invalid.")

    # Strict loading rejects missing or extra parameters rather than partially initializing an
    # architecture that happens to share some tensor shapes.
    model = RegimeTransformer(model_config)
    try:
        model.load_state_dict(checkpoint["model_state"], strict=True)
    except (RuntimeError, TypeError) as error:
        raise ValueError("Checkpoint model weights are incompatible.") from error

    if calculate_model_state_sha256(model) != checkpoint["model_state_sha256"]:
        raise ValueError("Checkpoint model-weight identity does not match its metadata.")

    # Load on CPU first so a checkpoint never assumes the device that created it.
    # Only a fully validated model is moved to the caller's inference device.
    model.to(device)
    model.eval()
    standardizer = FeatureStandardizer(
        mean=feature_mean,
        standard_deviation=feature_standard_deviation,
    ).to(device)
    metadata = {
        key: value
        for key, value in checkpoint.items()
        if key != "model_state"
    }
    return LoadedCheckpoint(
        model=model,
        standardizer=standardizer,
        metadata=metadata,
    )
