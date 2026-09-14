"""Tests for fitting and evaluating the regime Transformer."""

import pandas as pd
import pytest
import torch

from pricesanity.training.dataset import (
    AnnotatedSessionSplit,
    IGNORED_TARGET,
    create_training_data_loaders,
)
from pricesanity.training.model import RegimeTransformer, TransformerConfig
from pricesanity.training.trainer import (
    TrainingLoopConfig,
    calculate_dual_regime_loss,
    fit_feature_standardizer,
    fit_majority_classes,
    load_training_checkpoint,
    resolve_training_device,
    save_training_checkpoint,
    train_regime_transformer,
)


def _build_training_session(session_date: str, target: int) -> pd.DataFrame:
    """Build one completed three-candle session for training tests."""

    timestamps = pd.date_range(
        f"{session_date}T14:30:00Z",
        periods=3,
        freq="5min",
    )
    direction = 1.0 if target == 0 else -1.0
    return pd.DataFrame(
        {
            "session_date": [timestamps[0].date()] * 3,
            "candlestick_id": [
                f"{session_date}-candle-{position}" for position in range(3)
            ],
            "ts_event": timestamps,
            "open_gap": [0.01 * direction, 0.0, 0.0],
            "body": [0.02 * direction] * 3,
            "high_from_close": [0.01] * 3,
            "low_from_close": [-0.01] * 3,
            "current_target": [target] * 3,
            "anticipated_target": [target] * 3,
        }
    )


def test_fit_feature_standardizer_uses_only_supplied_training_sessions() -> None:
    """Evaluation values cannot influence the stored feature statistics."""

    training_session = _build_training_session("2016-01-04", 0)
    standardizer = fit_feature_standardizer([training_session])

    expected_features = torch.tensor(
        training_session[
            ["open_gap", "body", "high_from_close", "low_from_close"]
        ].to_numpy(),
        dtype=torch.float32,
    )
    transformed_features = standardizer.transform(expected_features)

    torch.testing.assert_close(
        transformed_features.mean(dim=0),
        torch.zeros(4),
        atol=1e-6,
        rtol=0.0,
    )
    assert torch.isfinite(transformed_features).all()


def test_calculate_dual_regime_loss_ignores_padding() -> None:
    """Artificial target positions make no contribution to either loss."""

    current_logits = torch.tensor(
        [[[4.0, 0.0, 0.0], [100.0, -100.0, -100.0]]],
        dtype=torch.float32,
    )
    anticipated_logits = torch.tensor(
        [[[0.0, 4.0, 0.0], [-100.0, 100.0, -100.0]]],
        dtype=torch.float32,
    )
    current_targets = torch.tensor([[0, IGNORED_TARGET]])
    anticipated_targets = torch.tensor([[1, IGNORED_TARGET]])

    losses = calculate_dual_regime_loss(
        current_logits,
        anticipated_logits,
        current_targets,
        anticipated_targets,
    )

    expected_loss = torch.nn.functional.cross_entropy(
        current_logits[:, :1].reshape(-1, 3),
        torch.tensor([0]),
    )
    torch.testing.assert_close(losses.current, expected_loss)
    torch.testing.assert_close(losses.anticipated, expected_loss)
    torch.testing.assert_close(losses.total, expected_loss)


def test_train_regime_transformer_keeps_test_until_after_selection() -> None:
    """A complete smoke run reports all splits and retains the best epoch."""

    sessions = tuple(
        _build_training_session(
            session_date.strftime("%Y-%m-%d"),
            target=index % 2,
        )
        for index, session_date in enumerate(
            pd.date_range("2016-01-04", periods=8)
        )
    )
    split = AnnotatedSessionSplit(
        training=sessions[:4],
        validation=sessions[4:6],
        test=sessions[6:],
    )
    data_loaders = create_training_data_loaders(
        split,
        timestamp_column="ts_event",
        batch_size=2,
        random_seed=42,
    )
    standardizer = fit_feature_standardizer(split.training)
    majority_classes = fit_majority_classes(split.training)
    model = RegimeTransformer(TransformerConfig(dropout=0.0))
    completed_epochs = []

    result = train_regime_transformer(
        model,
        data_loaders,
        standardizer,
        config=TrainingLoopConfig(
            epochs=2,
            learning_rate=1e-3,
            early_stopping_patience=2,
        ),
        device=torch.device("cpu"),
        majority_classes=majority_classes,
        epoch_callback=completed_epochs.append,
    )

    assert len(result.history) == 2
    assert completed_epochs == list(result.history)
    assert result.best_epoch in (1, 2)
    assert result.history[0].training.current.support == 12
    assert result.history[0].validation.current.support == 6
    assert result.test.current.support == 6
    assert len(result.test_predictions) == 6
    assert result.test_predictions["candlestick_id"].is_unique
    assert 0.0 <= result.test.current.accuracy <= 1.0
    assert 0.0 <= result.test.anticipated.macro_f1 <= 1.0


def test_save_training_checkpoint_preserves_reproducibility_data(tmp_path) -> None:
    """A saved model includes its architecture, scaling, history, and test result."""

    sessions = tuple(
        _build_training_session(
            session_date.strftime("%Y-%m-%d"),
            target=index % 2,
        )
        for index, session_date in enumerate(
            pd.date_range("2016-01-04", periods=3)
        )
    )
    split = AnnotatedSessionSplit(
        training=(sessions[0],),
        validation=(sessions[1],),
        test=(sessions[2],),
    )
    loaders = create_training_data_loaders(
        split,
        timestamp_column="ts_event",
        batch_size=1,
        random_seed=42,
    )
    standardizer = fit_feature_standardizer(split.training)
    majority_classes = fit_majority_classes(split.training)
    model = RegimeTransformer(TransformerConfig(dropout=0.0))
    training_config = TrainingLoopConfig(epochs=1, early_stopping_patience=1)
    result = train_regime_transformer(
        model,
        loaders,
        standardizer,
        config=training_config,
        device=torch.device("cpu"),
        majority_classes=majority_classes,
    )
    checkpoint_path = tmp_path / "regime-transformer.pt"

    save_training_checkpoint(
        checkpoint_path,
        model,
        standardizer,
        training_config,
        result,
    )
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

    assert checkpoint["model_config"]["model_dimension"] == 12
    assert checkpoint["feature_columns"] == [
        "open_gap",
        "body",
        "high_from_close",
        "low_from_close",
    ]
    assert checkpoint["best_epoch"] == result.best_epoch
    assert len(checkpoint["history"]) == 1
    assert "model_state" in checkpoint
    torch.testing.assert_close(checkpoint["feature_mean"], standardizer.mean)

    loaded_checkpoint = load_training_checkpoint(
        checkpoint_path,
        device=torch.device("cpu"),
    )
    assert not loaded_checkpoint.model.training
    assert (
        loaded_checkpoint.metadata["model_state_sha256"]
        == result.model_state_sha256
    )

    with pytest.raises(FileExistsError, match="already exists"):
        save_training_checkpoint(
            checkpoint_path,
            model,
            standardizer,
            training_config,
            result,
        )


def test_resolve_training_device_rejects_unknown_name() -> None:
    """A misspelled device cannot silently fall back to CPU."""

    assert resolve_training_device("cpu") == torch.device("cpu")
    with pytest.raises(ValueError, match="must be one of"):
        resolve_training_device("gpu")


@pytest.mark.parametrize("device_name", ["cpu", "mps"])
def test_device_training_inference_and_checkpoint_round_trip(device_name, tmp_path) -> None:
    """Exercise AdamW, variable lengths, inference, and reload on each available target device."""

    if device_name == "mps" and not torch.backends.mps.is_available():
        pytest.skip("Apple MPS is unavailable in this execution environment")

    sessions = []
    for session_date, length in zip(
        ("2016-01-04", "2016-01-05", "2016-01-06", "2016-01-07"),
        (81, 42, 81, 42), strict=True,
    ):
        # Exercise the full RTH limit and a natural early close, not just a tiny GPU tensor.
        session = pd.concat([_build_training_session(session_date, 0)] * 27, ignore_index=True)
        session = session.iloc[:length].copy()
        session["ts_event"] = pd.date_range(
            f"{session_date}T14:30:00Z", periods=length, freq="5min"
        )
        session["candlestick_id"] = session["ts_event"].astype(str)
        sessions.append(session)

    split = AnnotatedSessionSplit(sessions[:2], sessions[2:3], sessions[3:])
    loaders = create_training_data_loaders(
        split, timestamp_column="ts_event", batch_size=2, random_seed=42
    )
    standardizer = fit_feature_standardizer(split.training)
    model = RegimeTransformer(TransformerConfig(dropout=0.0))
    training_config = TrainingLoopConfig(epochs=1)
    device = resolve_training_device(device_name)
    result = train_regime_transformer(
        model, loaders, standardizer, config=training_config, device=device,
        majority_classes=fit_majority_classes(split.training),
    )
    assert result.test.current.support == 42
    assert next(model.parameters()).device.type == device_name
    path = tmp_path / "model.pt"
    save_training_checkpoint(path, model, standardizer, training_config, result)
    loaded = load_training_checkpoint(path, device=device)
    batch = next(iter(loaders.test))
    with torch.inference_mode():
        original = model(standardizer.to(device).transform(batch.features.to(device)),
                         batch.padding_mask.to(device))
        restored = loaded.model(loaded.standardizer.transform(batch.features.to(device)),
                                batch.padding_mask.to(device))
    torch.testing.assert_close(original.current_logits, restored.current_logits)
    assert not loaded.model.training


def test_test_pass_restores_validation_selected_weights(monkeypatch) -> None:
    """A worse second epoch cannot replace the first epoch before test evaluation."""

    from dataclasses import replace
    import pricesanity.training.trainer as trainer

    sessions = tuple(
        _build_training_session(session_date, 0)
        for session_date in ("2016-01-04", "2016-01-05", "2016-01-06")
    )
    split = AnnotatedSessionSplit(sessions[:1], sessions[1:2], sessions[2:])
    loaders = create_training_data_loaders(
        split, timestamp_column="ts_event", batch_size=1, random_seed=42
    )
    standardizer = fit_feature_standardizer(split.training)
    original_epoch = trainer._run_epoch
    original_test = trainer._evaluate_test_once
    validation_weight_hashes = []
    test_calls = []

    def controlled_epoch(model, data_loader, active_standardizer, **arguments):
        """Make selection deterministic while retaining real training and validation passes."""

        metrics = original_epoch(model, data_loader, active_standardizer, **arguments)
        if arguments["optimizer"] is None:
            assert data_loader is loaders.validation
            validation_weight_hashes.append(trainer.calculate_model_state_sha256(model))
            return replace(metrics, total_loss=float(len(validation_weight_hashes)))

        assert data_loader is loaders.training
        return metrics

    def verified_test(model, data_loader, active_standardizer, **arguments):
        """The untouched loader becomes accessible only after checkpoint restoration."""

        assert len(validation_weight_hashes) == 2
        assert trainer.calculate_model_state_sha256(model) == validation_weight_hashes[0]
        assert data_loader is loaders.test
        torch.testing.assert_close(active_standardizer.mean, standardizer.mean)
        test_calls.append(True)
        return original_test(model, data_loader, active_standardizer, **arguments)

    monkeypatch.setattr(trainer, "_run_epoch", controlled_epoch)
    monkeypatch.setattr(trainer, "_evaluate_test_once", verified_test)
    result = train_regime_transformer(
        RegimeTransformer(TransformerConfig(dropout=0.0)), loaders, standardizer,
        config=TrainingLoopConfig(epochs=2), device=torch.device("cpu"),
        majority_classes=fit_majority_classes(split.training),
    )
    assert result.best_epoch == 1
    assert len(test_calls) == 1


def test_tensor_reuse_keeps_features_raw_and_partitions_separate(monkeypatch) -> None:
    """Overlapping runs reuse conversion without mutating inputs or reusing fitted statistics."""

    import pricesanity.training.dataset as dataset

    sessions = tuple(
        _build_training_session(session_date, 0)
        for session_date in ("2016-01-04", "2016-01-05", "2016-01-06", "2016-01-07")
    )
    conversion_calls = []
    original_conversion = dataset.convert_session_to_tensors

    def count_conversion(session, **arguments):
        conversion_calls.append(id(session))
        return original_conversion(session, **arguments)

    monkeypatch.setattr(dataset, "convert_session_to_tensors", count_conversion)
    cache = {}
    first = create_training_data_loaders(
        AnnotatedSessionSplit(sessions[:1], sessions[1:2], sessions[2:3]),
        timestamp_column="ts_event", batch_size=1, random_seed=42, tensor_session_cache=cache,
    )
    second = create_training_data_loaders(
        AnnotatedSessionSplit(sessions[1:2], sessions[2:3], sessions[3:]),
        timestamp_column="ts_event", batch_size=1, random_seed=42, tensor_session_cache=cache,
    )
    assert len(conversion_calls) == 4
    assert first.validation.dataset[0] is second.training.dataset[0]
    original_features = second.training.dataset[0].features.clone()
    fit_feature_standardizer(sessions[1:2]).transform(original_features)
    torch.testing.assert_close(second.training.dataset[0].features, original_features)
    assert first.training.num_workers == 0
