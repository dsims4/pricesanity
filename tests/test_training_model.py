"""Tests for causal market-regime classification."""

import pytest
import torch

from pricesanity.training.model import RegimeTransformer, TransformerConfig
from pricesanity.training.tuning_context import ContextExperimentTransformer


def test_regime_transformer_returns_two_scores_per_candle() -> None:
    """Both classification tasks retain the input batch and session dimensions."""

    model = RegimeTransformer(TransformerConfig(dropout=0.0))
    features = torch.randn(2, 5, 4, dtype=torch.float32)
    padding_mask = torch.tensor(
        [
            [False, False, False, False, False],
            [False, False, False, True, True],
        ],
        dtype=torch.bool,
    )

    output = model(features, padding_mask)

    assert output.current_logits.shape == (2, 5, 3)
    assert output.anticipated_logits.shape == (2, 5, 3)


def test_regime_transformer_blocks_future_candles() -> None:
    """Changing later candles cannot alter an earlier prediction."""

    torch.manual_seed(42)
    model = RegimeTransformer(TransformerConfig(dropout=0.0))
    model.eval()

    original_features = torch.randn(2, 6, 4, dtype=torch.float32)
    changed_features = original_features.clone()

    # Replace only candles after position three to expose any future-data leak.
    changed_features[:, 4:, :] = 100.0
    padding_mask = torch.zeros(2, 6, dtype=torch.bool)

    with torch.no_grad():
        original_output = model(original_features, padding_mask)
        changed_output = model(changed_features, padding_mask)

    torch.testing.assert_close(
        original_output.current_logits[:, :4],
        changed_output.current_logits[:, :4],
    )
    torch.testing.assert_close(
        original_output.anticipated_logits[:, :4],
        changed_output.anticipated_logits[:, :4],
    )


def test_short_session_logits_match_the_same_session_inside_a_padded_batch() -> None:
    """Artificial early-close padding cannot change any real candle prediction."""

    torch.manual_seed(42)
    model = RegimeTransformer(TransformerConfig(dropout=0.0))
    model.eval()
    short_session = torch.randn(1, 3, 4, dtype=torch.float32)

    # The second representation contains the same real candles followed by
    # extreme values that exist only to occupy a longer batch rectangle.
    padded_features = torch.cat(
        [short_session, torch.full((1, 2, 4), 1_000.0)],
        dim=1,
    )
    short_mask = torch.zeros(1, 3, dtype=torch.bool)
    padded_mask = torch.tensor([[False, False, False, True, True]])

    with torch.inference_mode():
        short_output = model(short_session, short_mask)
        padded_output = model(padded_features, padded_mask)

    torch.testing.assert_close(
        short_output.current_logits,
        padded_output.current_logits[:, :3],
    )
    torch.testing.assert_close(
        short_output.anticipated_logits,
        padded_output.anticipated_logits[:, :3],
    )


def test_regime_transformer_rejects_incompatible_attention_heads() -> None:
    """Attention heads must divide the internal representation evenly."""

    with pytest.raises(ValueError, match="must be divisible"):
        RegimeTransformer(
            TransformerConfig(
                model_dimension=12,
                attention_head_count=5,
            )
        )


@pytest.mark.parametrize(
    ("features", "padding_mask", "message"),
    [
        (
            torch.zeros(2, 4, dtype=torch.float32),
            torch.zeros(2, 4, dtype=torch.bool),
            "batch, time, and feature axes",
        ),
        (
            torch.zeros(2, 5, 3, dtype=torch.float32),
            torch.zeros(2, 5, dtype=torch.bool),
            "expected 4 features",
        ),
        (
            torch.zeros(2, 5, 4, dtype=torch.float32),
            torch.zeros(2, 4, dtype=torch.bool),
            "must match the batch and time dimensions",
        ),
        (
            torch.zeros(2, 5, 4, dtype=torch.float32),
            torch.zeros(2, 5, dtype=torch.int64),
            "must use torch.bool",
        ),
    ],
)
def test_regime_transformer_rejects_invalid_batch_shapes(
    features: torch.Tensor,
    padding_mask: torch.Tensor,
    message: str,
) -> None:
    """Malformed batches fail clearly before entering attention."""

    model = RegimeTransformer(TransformerConfig())

    with pytest.raises(ValueError, match=message):
        model(features, padding_mask)


@pytest.mark.parametrize(
    ("features", "padding_mask", "message"),
    [
        (
            torch.zeros(2, 4, dtype=torch.float32),
            torch.zeros(2, 4, dtype=torch.bool),
            "batch, time, and feature axes",
        ),
        (
            torch.zeros(2, 5, 3, dtype=torch.float32),
            torch.zeros(2, 5, dtype=torch.bool),
            "expected 4 features",
        ),
        (
            torch.zeros(2, 5, 4, dtype=torch.float64),
            torch.zeros(2, 5, dtype=torch.bool),
            "must use torch.float32",
        ),
        (
            torch.zeros(2, 5, 4, dtype=torch.float32),
            torch.zeros(2, 4, dtype=torch.bool),
            "must match the batch and time dimensions",
        ),
        (
            torch.zeros(2, 5, 4, dtype=torch.float32),
            torch.zeros(2, 5, dtype=torch.int64),
            "must use torch.bool",
        ),
    ],
)
def test_strict_context_model_uses_the_common_input_validation(
    features: torch.Tensor,
    padding_mask: torch.Tensor,
    message: str,
) -> None:
    """Strict trailing windows reject malformed tensors before reshaping them."""

    model = ContextExperimentTransformer(
        TransformerConfig(dropout=0.0),
        context_length=2,
    )

    with pytest.raises(ValueError, match=message):
        model(features, padding_mask)


def test_strict_context_model_keeps_batch_sessions_independent() -> None:
    """Changing one complete session cannot alter another session's logits."""

    torch.manual_seed(42)
    model = ContextExperimentTransformer(
        TransformerConfig(dropout=0.0, layer_count=3),
        context_length=4,
    )
    model.eval()
    original_features = torch.randn(2, 7, 4, dtype=torch.float32)
    changed_features = original_features.clone()

    # Replace every candle in only the second session. The explicit window reshape must
    # preserve the batch boundary rather than mixing either session's values into the other.
    changed_features[1] = 1_000.0
    padding_mask = torch.zeros(2, 7, dtype=torch.bool)

    with torch.inference_mode():
        original_output = model(original_features, padding_mask)
        changed_output = model(changed_features, padding_mask)

    torch.testing.assert_close(
        original_output.current_logits[0],
        changed_output.current_logits[0],
    )
    torch.testing.assert_close(
        original_output.anticipated_logits[0],
        changed_output.anticipated_logits[0],
    )
