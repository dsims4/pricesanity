"""Small readable PyTorch adapters for causal sequence benchmark families."""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from pricesanity.models.protocol import (
    DualRegimeOutput,
    DualRegimePredictions,
    DualRegimeProbabilities,
    PredictionContext,
)
from pricesanity.training.model import RegimeTransformer, TransformerConfig


@dataclass(frozen=True)
class SequenceTrainingConfig:
    """Small optimization contract shared by TCN, GRU, and Transformer adapters."""

    epochs: int = 20
    learning_rate: float = 1e-3
    batch_size: int = 256
    weight_decay: float = 1e-4

    def __post_init__(self) -> None:
        """Reject optimization settings that cannot produce a valid training run."""

        if self.epochs <= 0 or self.batch_size <= 0 or self.learning_rate <= 0:
            raise ValueError("Sequence training duration, batch size, and rate must be positive.")
        if self.weight_decay < 0:
            raise ValueError("Sequence weight decay cannot be negative.")


@dataclass(frozen=True)
class SequenceModelOutput:
    """Both class-score heads at every causal sequence position."""

    current_logits: torch.Tensor
    anticipated_logits: torch.Tensor


class CausalConvolutionBlock(nn.Module):
    """One residual temporal block padded only on the historical side."""

    def __init__(self, width: int, kernel_size: int, dilation: int) -> None:
        """Create one left-padded residual convolution at the requested dilation."""

        super().__init__()
        if width <= 0 or kernel_size <= 0 or dilation <= 0:
            raise ValueError("TCN block dimensions must be positive.")
        self.left_padding = dilation * (kernel_size - 1)
        self.convolution = nn.Conv1d(
            width,
            width,
            kernel_size,
            dilation=dilation,
        )
        self.activation = nn.GELU()

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        """Preserve length while making every output depend only on positions at or before it."""

        # Symmetric Conv1d padding would let the kernel read future candles. Explicitly adding
        # zeros only on the left makes the causality rule visible and testable.
        padded_values = torch.nn.functional.pad(values, (self.left_padding, 0))
        return values + self.activation(self.convolution(padded_values))


class RegimeTCN(nn.Module):
    """Classify both regimes with stacked causal dilated convolutions.

    PyTorch supplies convolutions, activation, and linear heads. Price Sanity chooses their
    residual arrangement, exponentially growing dilation, and absent-history clearing rule.
    """

    def __init__(
        self,
        *,
        feature_count: int = 4,
        channel_width: int = 32,
        kernel_size: int = 3,
        layer_count: int = 3,
        regime_count: int = 3,
    ) -> None:
        """Build a causal temporal stack and two independent regime heads."""

        super().__init__()
        if min(feature_count, channel_width, kernel_size, layer_count, regime_count) <= 0:
            raise ValueError("TCN dimensions must be positive.")
        # Save architecture choices separately from tensors so reload can reconstruct capacity.
        self.configuration = {
            "feature_count": feature_count,
            "channel_width": channel_width,
            "kernel_size": kernel_size,
            "layer_count": layer_count,
            "regime_count": regime_count,
        }

        # A one-step projection places every candle in the temporal channel space without
        # mixing positions before the explicitly causal convolution stack begins.
        self.input_projection = nn.Conv1d(feature_count, channel_width, kernel_size=1)

        # Doubling dilation expands historical reach without multiplying kernel parameters.
        self.blocks = nn.ModuleList(
            CausalConvolutionBlock(channel_width, kernel_size, dilation=2**layer_index)
            for layer_index in range(layer_count)
        )

        # The two annotation questions share learned price-action context but retain separate
        # output heads so one label distribution cannot overwrite the other.
        self.current_head = nn.Linear(channel_width, regime_count)
        self.anticipated_head = nn.Linear(channel_width, regime_count)

    @property
    def receptive_field(self) -> int:
        """Return the maximum historical candle span used by the final block."""

        return 1 + (self.configuration["kernel_size"] - 1) * sum(
            2**layer_index for layer_index in range(self.configuration["layer_count"])
        )

    def forward(
        self,
        features: torch.Tensor,
        valid_history_mask: torch.Tensor | None = None,
    ) -> SequenceModelOutput:
        """Return `[batch, time, class]` logits without reading later time positions."""

        _validate_sequence_features(features, self.configuration["feature_count"])
        if valid_history_mask is None:
            valid_history_mask = torch.ones(
                features.shape[:2], dtype=torch.bool, device=features.device
            )
        _validate_history_mask(valid_history_mask, features)
        # Conv1d consumes channel-first values; expand the same candle mask over channels.
        mask = valid_history_mask.unsqueeze(1)
        hidden = self.input_projection(features.transpose(1, 2))
        hidden = hidden.masked_fill(~mask, 0.0)
        for block in self.blocks:
            hidden = block(hidden)
            # Convolution bias can make padded positions nonzero. Clearing them after every
            # block ensures artificial history cannot propagate toward the target candle.
            hidden = hidden.masked_fill(~mask, 0.0)
        # Restore the shared [sample, time, class] output convention before the two heads.
        hidden = hidden.transpose(1, 2)
        return SequenceModelOutput(
            current_logits=self.current_head(hidden),
            anticipated_logits=self.anticipated_head(hidden),
        )


class RegimeGRU(nn.Module):
    """Carry a recurrent state forward through each causal candle sequence.

    PyTorch owns the gated recurrence. Price Sanity owns the forward-only chronology,
    shared representation, two label heads, and packed-history handling in the adapter.
    """

    def __init__(
        self,
        *,
        feature_count: int = 4,
        hidden_size: int = 32,
        layer_count: int = 1,
        regime_count: int = 3,
    ) -> None:
        """Build a forward-only recurrent encoder and two regime heads."""

        super().__init__()
        if min(feature_count, hidden_size, layer_count, regime_count) <= 0:
            raise ValueError("GRU dimensions must be positive.")
        self.configuration = {
            "feature_count": feature_count,
            "hidden_size": hidden_size,
            "layer_count": layer_count,
            "regime_count": regime_count,
        }
        # The default unidirectional recurrence cannot read later candles into earlier states.
        self.gru = nn.GRU(
            input_size=feature_count,
            hidden_size=hidden_size,
            num_layers=layer_count,
            batch_first=True,
        )
        # Both heads read the same causal recurrent state while learning independent regimes.
        self.current_head = nn.Linear(hidden_size, regime_count)
        self.anticipated_head = nn.Linear(hidden_size, regime_count)

    def forward(self, features: torch.Tensor) -> SequenceModelOutput:
        """Return per-candle logits from recurrent states that only moved forward in time."""

        _validate_sequence_features(features, self.configuration["feature_count"])
        hidden, _ = self.gru(features)
        return SequenceModelOutput(
            current_logits=self.current_head(hidden),
            anticipated_logits=self.anticipated_head(hidden),
        )


class ExistingTransformerBridge(nn.Module):
    """Expose the production Transformer through the benchmark sequence contract."""

    def __init__(self, config: TransformerConfig) -> None:
        """Wrap the project Transformer in the common benchmark output contract."""

        super().__init__()
        self.transformer = RegimeTransformer(config)

    def forward(
        self,
        features: torch.Tensor,
        valid_history_mask: torch.Tensor | None = None,
    ) -> SequenceModelOutput:
        """Reuse causal attention while masking absent history in benchmark windows."""

        if valid_history_mask is None:
            valid_history_mask = torch.ones(
                features.shape[:2], dtype=torch.bool, device=features.device
            )
        _validate_history_mask(valid_history_mask, features)
        # PyTorch uses True for exclusion, opposite to the representation's validity contract.
        padding_mask = ~valid_history_mask
        output = self.transformer(features, padding_mask)
        return SequenceModelOutput(
            current_logits=output.current_logits,
            anticipated_logits=output.anticipated_logits,
        )


class TorchSequenceAdapter:
    """Fit and persist one dual-head causal PyTorch sequence classifier."""

    def __init__(
        self,
        *,
        model_name: str,
        model: nn.Module,
        architecture_configuration: dict[str, Any],
        training_configuration: SequenceTrainingConfig,
        random_seed: int,
        device: str,
        cpu_worker_count: int,
        standardize: bool = True,
    ) -> None:
        """Retain architecture, training, device, and reproducibility choices."""

        # Keep construction metadata beside fitted state so saved adapters can rebuild the
        # exact architecture without serializing Python construction callables.
        self.model_name = model_name
        self._model = model
        self.architecture_configuration = dict(architecture_configuration)
        self.training_configuration = training_configuration
        self.random_seed = random_seed
        self.device_name = device
        self.cpu_worker_count = cpu_worker_count
        self.standardize = standardize
        self._feature_mean: np.ndarray | None = None
        self._feature_scale: np.ndarray | None = None
        self._fitted = False

    @property
    def name(self) -> str:
        """Return the stable sequence-family registry name."""

        return self.model_name

    def fit(
        self,
        features: np.ndarray,
        current_targets: np.ndarray,
        anticipated_targets: np.ndarray,
        *,
        context: PredictionContext | None = None,
    ) -> None:
        """Fit on `[sample, time, feature]` float values and two class-index vectors."""

        features, current_targets, anticipated_targets = _validate_fit_inputs(
            features,
            current_targets,
            anticipated_targets,
        )

        # Device resolution is explicit: a requested accelerator must never silently fall
        # back to CPU and invalidate benchmark timing comparisons.
        device = _resolve_device(self.device_name)
        with _scoped_training_seed(self.random_seed, device):
            self._fit_seeded(
                features,
                current_targets,
                anticipated_targets,
                context=context,
                device=device,
            )
        # Publish fitted state only after every epoch and queued device operation succeeds.
        self._fitted = True

    def _fit_seeded(
        self,
        features: np.ndarray,
        current_targets: np.ndarray,
        anticipated_targets: np.ndarray,
        *,
        context: PredictionContext | None,
        device: torch.device,
    ) -> None:
        """Run deterministic CPU optimization inside an isolated RNG scope."""

        # These statistics belong only to this training partition. Reusing them across folds
        # would leak later price distributions into an earlier validation experiment.
        valid_history_mask = _context_history_mask(context, features.shape[:2])

        # Fit normalization on real candles only. Left padding represents unavailable history,
        # not a market observation at price zero.
        real_features = features[valid_history_mask]
        self._feature_mean = (
            real_features.mean(axis=0, dtype=np.float64)
            if self.standardize
            else np.zeros(features.shape[2], dtype=np.float64)
        )
        self._feature_scale = (
            real_features.std(axis=0, dtype=np.float64)
            if self.standardize
            else np.ones(features.shape[2], dtype=np.float64)
        )
        # A constant training feature has no informative scale; one avoids division by zero
        # while retaining centered values without learning anything from evaluation history.
        self._feature_scale = np.where(
            self._feature_scale > np.finfo(np.float64).eps,
            self._feature_scale,
            1.0,
        )
        standardized = self._standardize(features)

        # Padding is restored to zero after standardization. Otherwise `(0 - mean) / scale`
        # would turn absence into a plausible nonzero candle.
        standardized[~valid_history_mask] = 0.0

        # Build aligned host tensors once; each batch transfers only its own rows to the device.
        feature_tensor = torch.from_numpy(standardized)
        mask_tensor = torch.from_numpy(valid_history_mask)
        current_tensor = torch.from_numpy(current_targets)
        anticipated_tensor = torch.from_numpy(anticipated_targets)
        # A private shuffle generator makes batch order part of the declared experiment seed
        # instead of ambient global RNG state.
        generator = torch.Generator().manual_seed(self.random_seed)
        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(
                feature_tensor,
                mask_tensor,
                current_tensor,
                anticipated_tensor,
            ),
            batch_size=self.training_configuration.batch_size,
            shuffle=True,
            generator=generator,
        )

        self._model.to(device)
        self._model.train()

        # AdamW applies the declared weight decay independently of the gradient update; every
        # sequence family uses this same optimizer contract for a controlled training path.
        optimizer = torch.optim.AdamW(
            self._model.parameters(),
            lr=self.training_configuration.learning_rate,
            weight_decay=self.training_configuration.weight_decay,
        )
        # Epoch count is selected outside this fit by chronological folds. No internal random
        # validation split or evaluation-dependent stopping rule is introduced here.
        for _ in range(self.training_configuration.epochs):
            for batch_features, batch_mask, batch_current, batch_anticipated in loader:
                batch_features = batch_features.to(device)
                batch_current = batch_current.to(device)
                batch_anticipated = batch_anticipated.to(device)
                batch_mask = batch_mask.to(device)
                current_logits, anticipated_logits = self._final_logits(
                    batch_features, batch_mask
                )
                # Equal head weights preserve the benchmark's symmetric two-task objective.
                loss = (
                    torch.nn.functional.cross_entropy(current_logits, batch_current)
                    + torch.nn.functional.cross_entropy(
                        anticipated_logits, batch_anticipated
                    )
                ) / 2.0

                # Clear prior-batch gradients so each update corresponds to this batch alone.
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        # Complete queued accelerator work before the enclosing runner stops its fit timer.
        _synchronize_device(device)

    def predict_output(
        self,
        features: np.ndarray,
        *,
        context: PredictionContext | None = None,
    ) -> DualRegimeOutput:
        """Return native softmax class estimates from one batched forward pass."""

        self._require_fitted()
        features = _validate_sequence_array(features)
        # Apply the frozen training statistics before constructing inference batches; no
        # evaluation observation may update the fitted distribution.
        standardized = torch.from_numpy(self._standardize(features))
        valid_history_mask = _context_history_mask(context, features.shape[:2])
        standardized[~torch.from_numpy(valid_history_mask)] = 0.0
        history_mask = torch.from_numpy(valid_history_mask)
        device = _resolve_device(self.device_name)
        self._model.to(device)
        self._model.eval()

        # Keep both output streams in batch order for exact identity alignment in artifacts.
        current_probabilities = []
        anticipated_probabilities = []

        # Inference batches bound device memory without changing the order of persisted rows.
        with torch.inference_mode():
            for batch_start in range(
                0, len(standardized), self.training_configuration.batch_size
            ):
                batch = standardized[
                    batch_start : batch_start + self.training_configuration.batch_size
                ].to(device)
                batch_mask = history_mask[
                    batch_start : batch_start + self.training_configuration.batch_size
                ].to(device)
                current_logits, anticipated_logits = self._final_logits(
                    batch, batch_mask
                )
                current_probabilities.append(
                    torch.softmax(current_logits, dim=-1).cpu()
                )
                anticipated_probabilities.append(
                    torch.softmax(anticipated_logits, dim=-1).cpu()
                )
        _synchronize_device(device)
        current = torch.cat(current_probabilities).numpy().astype(np.float64)
        anticipated = torch.cat(anticipated_probabilities).numpy().astype(np.float64)
        # Argmax classes and softmax estimates originate from the same forward pass, keeping
        # saved predictions and displayed uncertainty internally consistent.
        return DualRegimeOutput(
            predictions=DualRegimePredictions(
                current=current.argmax(axis=1).astype(np.int64),
                anticipated=anticipated.argmax(axis=1).astype(np.int64),
            ),
            probabilities=DualRegimeProbabilities(
                current=current,
                anticipated=anticipated,
            ),
            scores=None,
            uncertainty_kind="softmax_probability_estimate",
        )

    def predict(
        self,
        features: np.ndarray,
        *,
        context: PredictionContext | None = None,
    ) -> DualRegimePredictions:
        """Return both class vectors through the combined inference path."""

        return self.predict_output(features, context=context).predictions

    def predict_proba(
        self,
        features: np.ndarray,
        *,
        context: PredictionContext | None = None,
    ) -> DualRegimeProbabilities:
        """Return both softmax probability-estimate matrices."""

        probabilities = self.predict_output(features, context=context).probabilities
        assert probabilities is not None
        return probabilities

    def describe(self) -> dict[str, Any]:
        """Describe architecture, training, resource, and uncertainty semantics."""

        return {
            "model_name": self.model_name,
            "adapter": "torch_sequence_v1",
            "architecture": dict(self.architecture_configuration),
            "training": asdict(self.training_configuration),
            "random_seed": self.random_seed,
            "device": self.device_name,
            "cpu_worker_count": self.cpu_worker_count,
            "standardize": self.standardize,
            "uncertainty_kind": "softmax_probability_estimate",
        }

    def save(self, path: str | Path) -> None:
        """Persist fitted state without changing the production Transformer format."""

        self._require_fitted()
        # CPU tensors make the saved state reloadable on a different available device without
        # changing the separate compatibility checks required for partial-run continuation.
        torch.save(
            {
                "format_version": 1,
                "description": self.describe(),
                "model_state": {
                    name: tensor.detach().cpu()
                    for name, tensor in self._model.state_dict().items()
                },
                # Tensors keep weights-only loading strict; serialized NumPy objects would
                # require relaxing PyTorch's safer unpickling boundary.
                "feature_mean": torch.as_tensor(self._feature_mean),
                "feature_scale": torch.as_tensor(self._feature_scale),
            },
            Path(path),
        )

    def state_fingerprint(self) -> str:
        """Hash fitted tensor values and training-only scaling state."""

        import hashlib

        self._require_fitted()
        # Hash names, dtype, shape, values, and preprocessing, not configuration alone: two
        # stochastic fits with identical settings must still have distinct fitted identities.
        digest = hashlib.sha256()
        for name, tensor in sorted(self._model.state_dict().items()):
            values = tensor.detach().cpu().contiguous()
            digest.update(name.encode("utf-8"))
            digest.update(str(values.dtype).encode("ascii"))
            digest.update(str(tuple(values.shape)).encode("ascii"))
            digest.update(values.numpy().tobytes())
        digest.update(np.asarray(self._feature_mean).tobytes())
        digest.update(np.asarray(self._feature_scale).tobytes())
        return digest.hexdigest()

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        device: str = "cpu",
    ) -> "TorchSequenceAdapter":
        """Restore one benchmark sequence model without reading production checkpoints."""

        # Decode onto CPU before selecting the requested runtime device. The state uses only
        # primitive metadata and tensors, so unrestricted pickle loading is unnecessary.
        state = torch.load(Path(path), map_location="cpu", weights_only=True)
        # Rebuild through the normal factory before loading weights so architecture validation
        # is shared between fresh training and artifact restoration.
        if not isinstance(state, dict) or state.get("format_version") != 1:
            raise ValueError("Unsupported sequence benchmark artifact.")
        description = state["description"]
        adapter = build_sequence_adapter(
            str(description["model_name"]),
            parameters={
                **description["architecture"],
                **description["training"],
                "standardize": description.get("standardize", True),
            },
            random_seed=int(description["random_seed"]),
            device=device,
            cpu_worker_count=int(description["cpu_worker_count"]),
        )
        adapter._model.load_state_dict(state["model_state"])
        adapter._feature_mean = state["feature_mean"].numpy().astype(np.float64)
        adapter._feature_scale = state["feature_scale"].numpy().astype(np.float64)
        adapter._fitted = True
        return adapter

    def parameter_count(self) -> int:
        """Count trainable architecture parameters for efficiency reporting."""

        return sum(
            parameter.numel()
            for parameter in self._model.parameters()
            if parameter.requires_grad
        )

    def synchronize(self) -> None:
        """Wait for queued accelerator work so wall-clock timing includes computation."""

        _synchronize_device(_resolve_device(self.device_name))

    def _final_logits(
        self,
        features: torch.Tensor,
        valid_history_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return target-candle logits while excluding artificial history by architecture."""

        if isinstance(self._model, RegimeGRU):
            # Packing needs host lengths and omits padded timesteps from recurrent updates.
            lengths = valid_history_mask.sum(dim=1).cpu()
            # pack_padded_sequence expects real values on the left. Move each right-aligned
            # suffix temporarily; only the final recurrent state is used for classification.
            compact = torch.zeros_like(features)
            for row_index, length in enumerate(lengths.tolist()):
                compact[row_index, :length] = features[row_index, -length:]
            packed = torch.nn.utils.rnn.pack_padded_sequence(
                compact, lengths, batch_first=True, enforce_sorted=False
            )
            _, hidden = self._model.gru(packed)
            # Use the top recurrent layer after each row's last real candle, not after padding.
            final_hidden = hidden[-1]
            return (
                self._model.current_head(final_hidden),
                self._model.anticipated_head(final_hidden),
            )
        if isinstance(self._model, RegimeTCN):
            output = self._model(features, valid_history_mask)
        elif isinstance(self._model, ExistingTransformerBridge):
            output = self._model(features, valid_history_mask)
        else:
            output = self._model(features)
        # Every input window ends at its target candle, so the final temporal position is the
        # only position evaluated by the shared candle-level benchmark.
        return output.current_logits[:, -1], output.anticipated_logits[:, -1]

    def _standardize(self, features: np.ndarray) -> np.ndarray:
        """Apply frozen per-feature statistics without changing sequence shape."""

        if self._feature_mean is None or self._feature_scale is None:
            raise RuntimeError("Sequence standardizer has not been fitted.")
        return ((features - self._feature_mean) / self._feature_scale).astype(
            np.float32,
            copy=False,
        )

    def _require_fitted(self) -> None:
        """Reject inference and persistence before model fitting finishes."""

        if not self._fitted:
            raise RuntimeError("Sequence benchmark adapter must be fitted first.")


def build_sequence_adapter(
    name: str,
    *,
    parameters: dict[str, Any],
    random_seed: int,
    device: str,
    cpu_worker_count: int,
) -> TorchSequenceAdapter:
    """Translate conceptual sequence settings into one readable PyTorch architecture."""

    # Consume a private copy; removing constructor fields must not alter selection identity.
    parameters = dict(parameters)

    # Remove optimization controls before architecture construction so unused settings fail
    # explicitly instead of being silently ignored by one model family.
    standardize = bool(parameters.pop("standardize", True))
    training_configuration = SequenceTrainingConfig(
        epochs=int(parameters.pop("epochs", 20)),
        learning_rate=float(parameters.pop("learning_rate", 1e-3)),
        batch_size=int(parameters.pop("batch_size", 256)),
        weight_decay=float(parameters.pop("weight_decay", 1e-4)),
    )
    feature_count = int(parameters.pop("feature_count", 4))
    regime_count = int(parameters.pop("regime_count", 3))

    # Construction happens inside an isolated RNG scope. Equal declared seeds therefore yield
    # equal initial weights without perturbing random state owned by the benchmark executor.
    if name == "tcn":
        architecture = {
            "feature_count": feature_count,
            "channel_width": int(parameters.pop("channel_width", 32)),
            "kernel_size": int(parameters.pop("kernel_size", 3)),
            "layer_count": int(parameters.pop("layer_count", 3)),
            "regime_count": regime_count,
        }
        with _scoped_torch_seed(random_seed):
            model: nn.Module = RegimeTCN(**architecture)
    elif name == "gru":
        architecture = {
            "feature_count": feature_count,
            "hidden_size": int(parameters.pop("hidden_size", 32)),
            "layer_count": int(parameters.pop("layer_count", 1)),
            "regime_count": regime_count,
        }
        with _scoped_torch_seed(random_seed):
            model = RegimeGRU(**architecture)
    elif name == "transformer":
        architecture = {
            "feature_count": feature_count,
            "model_dimension": int(parameters.pop("model_dimension", 48)),
            "attention_head_count": int(parameters.pop("attention_head_count", 3)),
            "layer_count": int(parameters.pop("layer_count", 2)),
            "feedforward_dimension": int(parameters.pop("feedforward_dimension", 96)),
            "dropout": float(parameters.pop("dropout", 0.1)),
            "regime_count": regime_count,
            "maximum_session_length": int(
                parameters.pop("maximum_session_length", 81)
            ),
        }
        with _scoped_torch_seed(random_seed):
            model = ExistingTransformerBridge(TransformerConfig(**architecture))
    else:
        raise ValueError(f"Unknown sequence benchmark model: {name}")
    if parameters:
        # Rejecting leftovers catches misspelled or family-inappropriate profile settings.
        raise ValueError(
            f"Unused {name} sequence parameters: " + ", ".join(sorted(parameters))
        )
    return TorchSequenceAdapter(
        model_name=name,
        model=model,
        architecture_configuration=architecture,
        training_configuration=training_configuration,
        random_seed=random_seed,
        device=device,
        cpu_worker_count=cpu_worker_count,
        standardize=standardize,
    )


def _validate_sequence_features(features: torch.Tensor, feature_count: int) -> None:
    """Require the fixed `[batch, time, feature]` sequence contract."""

    if (
        features.ndim != 3
        or features.shape[0] == 0
        or features.shape[1] == 0
        or features.shape[2] != feature_count
        or features.dtype is not torch.float32
    ):
        raise ValueError("Sequence features have an invalid shape or dtype.")


def _validate_history_mask(mask: torch.Tensor, features: torch.Tensor) -> None:
    """Require at least one real candle and a right-aligned validity suffix."""

    if mask.shape != features.shape[:2] or mask.dtype is not torch.bool:
        raise ValueError("Sequence validity mask must match batch and time axes.")
    if (~mask.any(dim=1)).any():
        raise ValueError("Every sequence needs at least one real candle.")
    if ((mask[:, :-1]) & (~mask[:, 1:])).any():
        raise ValueError("Real sequence history must be right aligned.")


def _context_history_mask(
    context: PredictionContext | None,
    shape: tuple[int, int],
) -> np.ndarray:
    """Resolve real history explicitly instead of treating a numeric zero as padding."""

    if context is None or context.valid_history_mask is None:
        return np.ones(shape, dtype=bool)
    mask = np.asarray(context.valid_history_mask, dtype=bool)
    if mask.shape != shape or not mask.any(axis=1).all():
        raise ValueError("Prediction history mask does not match sequence features.")
    if np.any(mask[:, :-1] & ~mask[:, 1:]):
        raise ValueError("Prediction history mask must be right aligned.")
    # Callers may clear artificial values; do not expose the representation's shared mask.
    return mask.copy()


class _scoped_torch_seed:
    """Seed construction while restoring the caller's CPU RNG state afterward."""

    def __init__(self, seed: int) -> None:
        """Prepare an isolated CPU random-number context for one seed."""

        self.seed = seed
        self._context = torch.random.fork_rng(devices=[])

    def __enter__(self) -> None:
        """Enter the isolated context and activate the declared seed."""

        self._context.__enter__()
        torch.manual_seed(self.seed)

    def __exit__(self, exception_type, exception, traceback) -> None:
        """Restore the CPU random-number state owned by the caller."""

        self._context.__exit__(exception_type, exception, traceback)


class _scoped_training_seed:
    """Isolate model-training RNG changes from the surrounding benchmark process."""

    def __init__(self, seed: int, device: torch.device) -> None:
        """Capture NumPy and device RNG settings for one isolated training run."""

        self.seed = seed
        self.device = device
        cuda_devices = []
        if device.type == "cuda":
            cuda_devices = [
                device.index if device.index is not None else torch.cuda.current_device()
            ]
        self._torch_context = torch.random.fork_rng(devices=cuda_devices)
        self._numpy_state: tuple[Any, ...] | None = None
        self._deterministic_algorithms = torch.are_deterministic_algorithms_enabled()

    def __enter__(self) -> None:
        """Activate deterministic seeded behavior supported by the selected device."""

        self._torch_context.__enter__()
        self._numpy_state = np.random.get_state()
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        # CPU benchmark repeats must represent a declared seed rather than ambient process
        # state. Accelerator kernels may still have backend-specific limitations, which are
        # reported as a reproducibility caveat instead of being described as bitwise portable.
        if self.device.type == "cpu":
            torch.use_deterministic_algorithms(True)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(self.seed)
        elif self.device.type == "mps" and hasattr(torch.mps, "manual_seed"):
            torch.mps.manual_seed(self.seed)

    def __exit__(self, exception_type, exception, traceback) -> None:
        """Restore every random-number and determinism setting changed on entry."""

        if self._numpy_state is not None:
            np.random.set_state(self._numpy_state)
        if self.device.type == "cpu":
            torch.use_deterministic_algorithms(self._deterministic_algorithms)
        self._torch_context.__exit__(exception_type, exception, traceback)


def _validate_sequence_array(features: np.ndarray) -> np.ndarray:
    """Return a contiguous float32 sequence array, reusing compatible input storage."""

    features = np.asarray(features, dtype=np.float32)
    if features.ndim != 3 or features.shape[0] == 0 or not np.isfinite(features).all():
        raise ValueError("Sequence adapters require finite nonempty three-axis features.")
    return np.ascontiguousarray(features)


def _validate_fit_inputs(
    features: np.ndarray,
    current_targets: np.ndarray,
    anticipated_targets: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Align float sequence inputs with two integer regime target vectors."""

    features = _validate_sequence_array(features)
    current_targets = np.asarray(current_targets, dtype=np.int64)
    anticipated_targets = np.asarray(anticipated_targets, dtype=np.int64)
    if current_targets.shape != (len(features),) or anticipated_targets.shape != (
        len(features),
    ):
        raise ValueError("Sequence targets must align with feature samples.")
    if not np.isin(current_targets, (0, 1, 2)).all() or not np.isin(
        anticipated_targets, (0, 1, 2)
    ).all():
        raise ValueError("Sequence targets contain an unknown regime class.")
    return features, current_targets, anticipated_targets


def _resolve_device(device_name: str) -> torch.device:
    """Resolve explicit benchmark devices without silently moving an experiment."""

    if device_name == "cpu":
        return torch.device("cpu")
    if device_name == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if device_name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    raise ValueError(f"Requested benchmark device is unavailable: {device_name}")


def _synchronize_device(device: torch.device) -> None:
    """Include queued accelerator work inside timing boundaries."""

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()
