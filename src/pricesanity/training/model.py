"""Define the causal Transformer used to classify market regimes."""

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class TransformerConfig:
    """Control the size and regularization of the regime Transformer."""

    feature_count: int = 4
    model_dimension: int = 12
    attention_head_count: int = 3
    layer_count: int = 2
    feedforward_dimension: int = 48
    dropout: float = 0.1
    regime_count: int = 3
    maximum_session_length: int = 81


@dataclass(frozen=True)
class RegimeTransformerOutput:
    """Unnormalized class scores produced for every candlestick."""

    current_logits: torch.Tensor
    anticipated_logits: torch.Tensor


class RegimeTransformer(nn.Module):
    """Classify both regimes at every candle without seeing future candles."""

    def __init__(self, config: TransformerConfig) -> None:
        """Create the input representation layers.

        Args:
            config: Dimensions and regularization used by the Transformer.

        Raises:
            ValueError: If the model dimension cannot be divided among the attention heads.
        """

        super().__init__()

        # Reject invalid sizes here so an experiment fails before training begins.
        positive_dimensions = (
            config.feature_count,
            config.model_dimension,
            config.attention_head_count,
            config.layer_count,
            config.feedforward_dimension,
            config.regime_count,
            config.maximum_session_length,
        )
        if any(dimension <= 0 for dimension in positive_dimensions):
            raise ValueError("Transformer dimensions must all be positive.")
        if not 0.0 <= config.dropout < 1.0:
            raise ValueError("Transformer dropout must be at least zero and less than one.")

        # Each attention head must receive an equal part of the internal representation.
        if config.model_dimension % config.attention_head_count != 0:
            raise ValueError(
                "The model dimension must be divisible by the attention head count."
            )

        # Retain the architecture settings for validation and saved model metadata.
        self.config = config

        # Project four candle measurements into the configured learned representation width.
        self.feature_projection = nn.Linear(
            config.feature_count,
            config.model_dimension,
        )

        # Position indices describe time within the supplied sequence: absolute session
        # position for standalone training, relative window position in the benchmark bridge.
        self.position_embedding = nn.Embedding(
            config.maximum_session_length,
            config.model_dimension,
        )

        # Every run uses the same upper-triangular rule. Build its maximum size
        # once and slice it per batch instead of allocating it every forward pass.
        causal_attention_mask = torch.triu(
            torch.ones(
                config.maximum_session_length,
                config.maximum_session_length,
                dtype=torch.bool,
            ),
            diagonal=1,
        )
        self.register_buffer(
            "_causal_attention_mask",
            causal_attention_mask,
            persistent=False,
        )

        # Drop part of the combined representation during training to discourage
        # the small model from memorizing the currently limited session corpus.
        self.input_dropout = nn.Dropout(config.dropout)

        # Reuse PyTorch's tested attention and feed-forward implementation while
        # keeping sessions as the first batch dimension used by our DataLoader.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.model_dimension,
            nhead=config.attention_head_count,
            dim_feedforward=config.feedforward_dimension,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        # Stacking encoder layers lets later attention operate on price-action
        # relationships already formed by earlier attention.
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.layer_count,
            norm=nn.LayerNorm(config.model_dimension),
            enable_nested_tensor=False,
        )

        # Keep the two human judgments as separate tasks over the same learned
        # candle representation rather than allowing either label into the input.
        self.current_regime_head = nn.Linear(
            config.model_dimension,
            config.regime_count,
        )
        self.anticipated_regime_head = nn.Linear(
            config.model_dimension,
            config.regime_count,
        )

    def _validate_inputs(
        self,
        features: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> tuple[int, int, int]:
        """Validate the common tensor contract before any attention path runs."""

        # Experimental subclasses use a different attention path, but malformed tensors
        # should still fail with the same useful errors as the production Transformer.
        if features.ndim != 3:
            raise ValueError("Transformer features must have batch, time, and feature axes.")

        batch_size, session_length, feature_count = features.shape
        if batch_size == 0 or session_length == 0:
            raise ValueError("Transformer features cannot contain an empty axis.")
        if feature_count != self.config.feature_count:
            raise ValueError(
                f"Transformer expected {self.config.feature_count} features, "
                f"but received {feature_count}."
            )
        if session_length > self.config.maximum_session_length:
            raise ValueError(
                "Transformer session length exceeds its configured maximum."
            )
        if features.dtype is not torch.float32:
            raise ValueError("Transformer features must use torch.float32.")

        if padding_mask.shape != (batch_size, session_length):
            raise ValueError(
                "Transformer padding mask must match the batch and time dimensions."
            )
        if padding_mask.dtype is not torch.bool:
            raise ValueError("Transformer padding mask must use torch.bool.")
        if padding_mask.device != features.device:
            raise ValueError("Transformer features and padding mask must share a device.")

        return batch_size, session_length, feature_count

    def forward(
        self,
        features: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> RegimeTransformerOutput:
        """Classify every real candle using only its available history.

        Args:
            features: Candle features shaped as batch, time, and feature dimensions.
            padding_mask: Boolean batch-by-time mask with True at artificial positions.

        Returns:
            Current and anticipated regime scores for each candle and class.

        Raises:
            ValueError: If the tensors do not match the configured model dimensions.
        """

        # Require the exact tensor layout produced by the session collator.
        _, session_length, _ = self._validate_inputs(features, padding_mask)

        # Number from the supplied sequence's start. Complete-session training shares
        # time-of-session positions; benchmark contexts restart their relative indices.
        candle_positions = torch.arange(
            session_length,
            device=features.device,
        ).unsqueeze(0)

        # Combine what each candle contains with where it occurs before attention
        # begins comparing it with earlier price action.
        hidden_candles = self.feature_projection(features)
        hidden_candles = hidden_candles + self.position_embedding(candle_positions)
        hidden_candles = self.input_dropout(hidden_candles)

        # True values above the diagonal hide every future candle from each query.
        causal_attention_mask = self._causal_attention_mask[
            :session_length,
            :session_length,
        ]

        # The causal mask blocks future information while the padding mask removes
        # artificial candles added only to make a rectangular batch.
        encoded_candles = self.encoder(
            hidden_candles,
            mask=causal_attention_mask,
            src_key_padding_mask=padding_mask,
        )

        # Return raw logits because cross-entropy applies the stable probability
        # conversion internally during training.
        return RegimeTransformerOutput(
            current_logits=self.current_regime_head(encoded_candles),
            anticipated_logits=self.anticipated_regime_head(encoded_candles),
        )
