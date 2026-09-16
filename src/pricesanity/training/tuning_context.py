"""Strict trailing-window context experiments without changing the production Transformer."""

import torch

from pricesanity.training.model import (
    RegimeTransformer,
    RegimeTransformerOutput,
    TransformerConfig,
)


class ContextExperimentTransformer(RegimeTransformer):
    """Predict each candle from at most L candles, retaining absolute session positions.

    A banded per-layer attention mask alone would allow older information to travel
    through intermediate candles in deeper layers. Separate trailing windows enforce
    the actual end-to-end context limit, without allowing sequences to cross sessions.
    """

    def __init__(self, config: TransformerConfig, context_length: int | None) -> None:
        super().__init__(config)
        if context_length is not None and not 1 <= context_length <= config.maximum_session_length:
            raise ValueError("Context length must fit within the configured session length.")
        self.context_length = context_length
        window_length = context_length or config.maximum_session_length
        candle_positions = torch.arange(config.maximum_session_length)
        window_starts = (candle_positions - window_length + 1).clamp(min=0)
        window_positions = window_starts[:, None] + torch.arange(window_length)[None, :]

        # These indices follow model.to(device), just like the existing causal-mask buffer.
        # Keeping them nonpersistent avoids bloating checkpoints with deterministic geometry.
        self.register_buffer("_window_positions", window_positions, persistent=False)
        self.register_buffer(
            "_window_future_mask", window_positions > candle_positions[:, None], persistent=False
        )
        self.register_buffer("_window_last_positions", candle_positions - window_starts,
                             persistent=False)

    def forward(self, features, padding_mask) -> RegimeTransformerOutput:
        if self.context_length is None:
            return super().forward(features, padding_mask)

        batch_size, session_length, feature_count = self._validate_inputs(
            features,
            padding_mask,
        )
        if session_length <= self.context_length:
            return super().forward(features, padding_mask)

        window_length = self.context_length
        window_positions = self._window_positions[:session_length]
        windows = features[:, window_positions].reshape(-1, window_length, feature_count)
        window_padding = (
            padding_mask[:, window_positions] | self._window_future_mask[:session_length]
        ).reshape(-1, window_length)

        # Padded target windows contribute no loss. Give such windows one dummy key so their
        # attention softmax is defined; otherwise all-masked attention can produce NaN gradients.
        window_padding = window_padding.clone()
        window_padding[:, 0] &= ~window_padding.all(dim=1)
        hidden = self.feature_projection(windows)
        positions = self.position_embedding(window_positions).repeat(batch_size, 1, 1)
        hidden = self.input_dropout(hidden + positions)
        encoded = self.encoder(
            hidden,
            mask=self._causal_attention_mask[:window_length, :window_length],
            src_key_padding_mask=window_padding,
        )

        # Keep only each target candle's representation. Earlier candles in its window supply
        # context but do not become duplicated training examples or alter the loss weighting.
        last_positions = self._window_last_positions[:session_length].repeat(batch_size)
        row_positions = torch.arange(len(last_positions), device=features.device)
        target_encodings = encoded[row_positions, last_positions].reshape(
            batch_size, session_length, self.config.model_dimension
        )
        return RegimeTransformerOutput(
            self.current_regime_head(target_encodings),
            self.anticipated_regime_head(target_encodings),
        )
