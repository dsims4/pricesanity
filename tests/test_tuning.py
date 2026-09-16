"""Verify tuning diagnostics keep test data out and count transition neighborhoods honestly."""

from dataclasses import replace
import json

import numpy as np
import pandas as pd
import pytest
import torch

from pricesanity.training.dataset import convert_session_to_tensors
from pricesanity.training.tuning import (
    EXPERIMENT_IDENTITY_FILENAME,
    TuningSettings,
    _load_frozen_tuning_split,
    _select_stage_result,
    _semantic_snapshot_sha256,
    classification_report,
    run_experiment,
    transition_report,
)


def test_class_reports_use_truth_rows_and_expose_minority_failure() -> None:
    """A frequent class cannot hide zero recall in an unpredicted minority class."""

    report = classification_report([0, 0, 1, 2], [0, 2, 2, 2])
    assert report["confusion_matrix"] == [[1, 0, 1], [0, 0, 1], [0, 0, 1]]
    assert report["per_class"]["bear"]["recall"] == 0
    assert report["per_class"]["range"]["precision"] == 1 / 3
    assert report["accuracy"] == 0.5


def test_transition_regions_never_cross_sessions_or_invent_opening_changes() -> None:
    """Stop tolerance regions at session boundaries; a day's first label is not a change."""

    report = transition_report(
        [[0, 0, 1], [2, 2, 2]], [[0, 0, 0], [2, 2, 2]]
    )
    assert report["0"]["near_change"]["support"] == 1
    assert report["0"]["near_change"]["accuracy"] == 0
    assert report["1"]["near_change"]["support"] == 2
    assert report["2"]["near_change"]["support"] == 3
    assert report["2"]["other_candles"]["support"] == 3
    no_changes = transition_report([[0, 0], [1, 1]], [[0, 0], [1, 1]])
    assert no_changes["0"]["near_change"]["support"] == 0
    assert no_changes["0"]["near_change"]["accuracy"] is None


def test_tuning_run_uses_no_official_test_evaluator(tmp_path, monkeypatch) -> None:
    """Candidate fitting records training/validation only and preserves both input frames."""

    import pricesanity.training.trainer as trainer

    def forbidden_test(*args, **kwargs):
        raise AssertionError("Tuning reached the official test evaluator")

    monkeypatch.setattr(trainer, "_evaluate_test_once", forbidden_test)
    sessions = []
    for session_date in ("2016-01-04", "2016-01-05"):
        timestamps = pd.date_range(f"{session_date}T14:30:00Z", periods=3, freq="5min")
        sessions.append(pd.DataFrame({
            "session_date": timestamps.date, "ts_event": timestamps,
            "candlestick_id": timestamps.astype(str),
            "open_gap": [0.0, 0.01, -0.01], "body": [0.01, -0.01, 0.0],
            "high_from_close": 0.01, "low_from_close": -0.01,
            "current_target": [0, 1, 2], "anticipated_target": [1, 2, 0],
        }))
    originals = [session.copy(deep=True) for session in sessions]
    cache = {id(session): convert_session_to_tensors(session, timestamp_column="ts_event")
             for session in sessions}
    settings = replace(TuningSettings(), epochs=1)
    report = run_experiment("test", sessions[:1], sessions[1:], cache, settings,
                            tmp_path, torch.device("cpu"), snapshot_sha256="a" * 64)
    assert report["test"] is None
    assert report["validation"]["current"]["support"] == 3
    assert report["training"]["current"]["support"] == 3
    assert np.isfinite(report["training_loss"]["total_loss"])
    for session, original in zip(sessions, originals, strict=True):
        pd.testing.assert_frame_equal(session, original)


def test_context_limit_blocks_old_and_future_candles_across_multiple_layers() -> None:
    """A strict four-candle context cannot expand indirectly through a deeper encoder."""

    from pricesanity.training.model import TransformerConfig
    from pricesanity.training.tuning_context import ContextExperimentTransformer

    torch.manual_seed(42)
    model = ContextExperimentTransformer(TransformerConfig(dropout=0.0, layer_count=3), 4)
    model.eval()
    features = torch.randn(1, 10, 4)
    mask = torch.zeros(1, 10, dtype=torch.bool)
    with torch.inference_mode():
        original = model(features, mask).current_logits
        changed = features.clone()
        changed[:, 0] += 100
        changed[:, 8:] -= 100
        modified = model(changed, mask).current_logits

    # Candle seven may use only candles four through seven, regardless of encoder depth.
    torch.testing.assert_close(original[:, 7], modified[:, 7])


def test_context_padding_has_finite_gradients_and_preserves_absolute_positions() -> None:
    """Early-close padding cannot create all-masked NaNs or reset the time-of-day embedding."""

    from pricesanity.training.model import RegimeTransformer, TransformerConfig
    from pricesanity.training.tuning_context import ContextExperimentTransformer

    configuration = TransformerConfig(dropout=0.0)
    model = ContextExperimentTransformer(configuration, 4)
    features = torch.randn(2, 10, 4)
    mask = torch.zeros(2, 10, dtype=torch.bool)
    mask[1, 3:] = True
    output = model(features, mask)
    output.current_logits[~mask].sum().backward()
    assert all(torch.isfinite(parameter.grad).all() for parameter in model.parameters()
               if parameter.grad is not None)

    # Full context and sessions shorter than the limit retain the original forward path.
    original = RegimeTransformer(configuration)
    original.load_state_dict(model.state_dict())
    model.eval()
    original.eval()
    with torch.inference_mode():
        torch.testing.assert_close(model(features[:, :3], mask[:, :3]).current_logits,
                                   original(features[:, :3], mask[:, :3]).current_logits)


@pytest.mark.parametrize(
    ("selected_head", "unselected_head"),
    (("current", "anticipated"), ("anticipated", "current")),
)
def test_single_head_diagnostic_does_not_update_other_head(
    selected_head,
    unselected_head,
) -> None:
    """Ablating one loss leaves that output head unchanged while training the selected task."""

    from pricesanity.training.dataset import TensorSession, TensorSessionDataset
    from pricesanity.training.model import RegimeTransformer, TransformerConfig
    from pricesanity.training.trainer import FeatureStandardizer, _run_epoch
    from pricesanity.training.tuning import build_loader

    timestamps = tuple(pd.date_range("2016-01-04T14:30:00Z", periods=3, freq="5min"))
    session = TensorSession(timestamps[0].date(), tuple(map(str, timestamps)), timestamps,
                            torch.randn(3, 4), torch.tensor([0, 1, 2]), torch.tensor([2, 1, 0]))
    model = RegimeTransformer(TransformerConfig(dropout=0.0))
    previous_parameters = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
    }
    _run_epoch(
        model, build_loader([session], TuningSettings(), shuffle=False),
        FeatureStandardizer(torch.zeros(4), torch.ones(4)), device=torch.device("cpu"),
        optimizer=torch.optim.AdamW(model.parameters(), lr=0.01), gradient_clip=1.0,
        loss_setup=selected_head,
    )
    current_parameters = dict(model.named_parameters())
    selected_prefix = f"{selected_head}_regime_head."
    unselected_prefix = f"{unselected_head}_regime_head."
    assert all(
        not torch.equal(previous_parameters[name], parameter)
        for name, parameter in current_parameters.items()
        if name.startswith(selected_prefix)
    )
    assert all(
        torch.equal(previous_parameters[name], parameter)
        for name, parameter in current_parameters.items()
        if name.startswith(unselected_prefix)
    )
    assert any(
        not torch.equal(previous_parameters[name], parameter)
        for name, parameter in current_parameters.items()
        if name.startswith("encoder.")
    )


def test_resume_uses_frozen_snapshot_without_reading_live_annotations(
    tmp_path,
    monkeypatch,
) -> None:
    """Additional annotation work must not change an in-progress tuning comparison."""

    import hashlib
    import sqlite3
    import pricesanity.training.tuning as tuning

    sessions = []
    all_timestamps = pd.date_range("2016-01-04T14:30:00Z", periods=130, freq="D")
    for timestamp in all_timestamps[:120]:
        sessions.append(pd.DataFrame({
            "session_date": [timestamp.date()], "ts_event": [timestamp],
            "candlestick_id": [str(timestamp)], "open_gap": [0.0], "body": [0.01],
            "high_from_close": [0.01], "low_from_close": [-0.01],
            "current_target": [0], "anticipated_target": [1],
        }))
    snapshot = hashlib.sha256()
    for session in sessions:
        snapshot.update(pd.util.hash_pandas_object(session, index=False).to_numpy().tobytes())
    pd.concat(sessions, ignore_index=True).to_parquet(tmp_path / "tuning_snapshot.parquet")
    manifest = {
        "protocol": "expanding_history_v1",
        "run_index": 1,
        "training_dates": [str(session["session_date"].iloc[0]) for session in sessions[:100]],
        "validation_dates": [str(session["session_date"].iloc[0]) for session in sessions[100:]],
        "reserved_test_dates": [str(timestamp.date()) for timestamp in all_timestamps[120:]],
        "training_validation_snapshot_sha256": snapshot.hexdigest(),
        "test_evaluated": False,
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))

    def forbidden_database(*args, **kwargs):
        raise AssertionError("Resuming tuning accessed the live annotation database")

    calls = []

    def record_experiment(name, training, validation, *args, **kwargs):
        calls.append((name, len(training), len(validation)))
        return {"name": name, "tiny_passed": True}

    monkeypatch.setattr(sqlite3, "connect", forbidden_database)
    monkeypatch.setattr(tuning, "run_experiment", record_experiment)
    assert tuning.main([
        "--normalized", str(tmp_path / "not-read.parquet"),
        "--database", str(tmp_path / "not-read.db"), "--config", "configs/default.yaml",
        "--output-directory", str(tmp_path), "--resume", "--device", "cpu",
    ]) == 0
    assert calls == [("baseline", 100, 20), ("tiny_5_sessions", 5, 0)]
    assert json.loads((tmp_path / "manifest.json").read_text()) == manifest


def _selection_candidate(name, score, *, context=16, parameters=100):
    """Keep synthetic selection scores independent of stochastic model fitting."""

    return {
        "name": name,
        "settings": {"context_length": context},
        "model_config": {"maximum_session_length": 81},
        "parameter_count": parameters,
        "validation_loss": 1.0,
        "validation": {
            "current": {"macro_f1": score},
            "anticipated": {"macro_f1": score},
        },
    }


@pytest.mark.parametrize(
    ("stage", "large_score", "expected"),
    [
        ("context_length", 0.590212, "small"),
        ("context_length", 0.596, "large"),
        ("model_width_matched_heads", 0.590212, "small"),
        ("model_width_matched_heads", 0.596, "large"),
        ("depth", 0.590212, "small"),
        ("feedforward", 0.590212, "small"),
    ],
)
def test_practical_selection_uses_stage_specific_simplicity(stage, large_score, expected):
    """Negligible gains favor simplicity; material gains still justify a larger model."""

    small = _selection_candidate("small", 0.589932)
    large = _selection_candidate(
        "large", large_score, context=32,
        parameters=100 if stage == "context_length" else 200,
    )
    raw, practical, _ = _select_stage_result(stage, small, [small, large])
    assert raw is large
    assert practical["name"] == expected


def test_equivalence_margin_is_measured_from_raw_winner():
    """Several individually small losses must not chain into a materially worse choice."""

    smallest = _selection_candidate("smallest", 0.58, context=8)
    middle = _selection_candidate("middle", 0.584, context=16)
    best = _selection_candidate("best", 0.588, context=32)
    raw, selected, _ = _select_stage_result(
        "context_length", middle, [smallest, middle, best]
    )
    assert raw is best
    assert selected is middle


@pytest.fixture
def small_tuning_experiment(tmp_path):
    """Publish a real one-epoch bundle so integrity tests exercise saved tensors too."""

    sessions = []
    for day in ("2016-01-04", "2016-01-05"):
        timestamps = pd.date_range(f"{day}T14:30:00Z", periods=3, freq="5min")
        sessions.append(pd.DataFrame({
            "session_date": timestamps.date, "ts_event": timestamps,
            "candlestick_id": timestamps.astype(str),
            "open_gap": [0.0, 0.01, -0.01], "body": [0.01, -0.01, 0.0],
            "high_from_close": 0.01, "low_from_close": -0.01,
            "current_target": [0, 1, 2], "anticipated_target": [1, 2, 0],
        }))
    cache = {
        id(session): convert_session_to_tensors(session, timestamp_column="ts_event")
        for session in sessions
    }
    settings = replace(TuningSettings(), epochs=1)
    arguments = (
        "candidate", sessions[:1], sessions[1:], cache, settings,
        tmp_path, torch.device("cpu"),
    )
    checksum = _semantic_snapshot_sha256(sessions)
    report = run_experiment(*arguments, snapshot_sha256=checksum)
    return arguments, checksum, json.loads(json.dumps(report))


def test_completed_experiment_reuses_only_matching_identity(small_tuning_experiment, monkeypatch):
    """A completed candidate must never train again just to resume its checklist."""

    import pricesanity.training.tuning as tuning

    arguments, checksum, original = small_tuning_experiment

    def forbidden_epoch(*args, **kwargs):
        raise AssertionError("A completed candidate was retrained")

    monkeypatch.setattr(tuning, "_run_epoch", forbidden_epoch)
    assert run_experiment(*arguments, snapshot_sha256=checksum) == original
    with pytest.raises(ValueError, match="identity does not match"):
        run_experiment(*arguments, snapshot_sha256="different snapshot")
    changed = list(arguments)
    changed[4] = replace(arguments[4], learning_rate=0.01)
    with pytest.raises(ValueError, match="different data, settings, or environment"):
        run_experiment(*changed, snapshot_sha256=checksum)


@pytest.mark.parametrize("artifact", ["report.json", "diagnostic_weights.pt"])
def test_completed_artifact_tampering_is_rejected(small_tuning_experiment, artifact):
    """Readable changes are still changes: checksums cover metrics as well as weights."""

    arguments, checksum, _ = small_tuning_experiment
    path = arguments[5] / arguments[0] / artifact
    if artifact.endswith("json"):
        report = json.loads(path.read_text())
        report["elapsed_seconds"] += 1
        path.write_text(json.dumps(report))
    else:
        weights = torch.load(path, weights_only=True)
        next(iter(weights["model_state"].values())).add_(0.01)
        torch.save(weights, path)
    with pytest.raises(ValueError, match="identity does not match"):
        run_experiment(*arguments, snapshot_sha256=checksum)


def test_partial_experiment_restarts_without_touching_completed_neighbor(
    small_tuning_experiment,
):
    """An interrupted candidate restarts at epoch one and preserves adjacent complete runs."""

    from dataclasses import asdict

    arguments, checksum, original = small_tuning_experiment
    partial_arguments = list(arguments)
    partial_arguments[0] = "interrupted"
    directory = arguments[5] / "interrupted"
    directory.mkdir()
    (directory / "settings.json").write_text(json.dumps(asdict(arguments[4])))
    completed = run_experiment(*partial_arguments, snapshot_sha256=checksum)
    assert completed["best_epoch"] == 1
    assert (directory / EXPERIMENT_IDENTITY_FILENAME).is_file()
    assert run_experiment(*arguments, snapshot_sha256=checksum) == original


def test_legacy_weights_must_reproduce_report_before_identity_is_created(
    small_tuning_experiment,
):
    """Migration cannot certify a plausible but unrelated same-shaped checkpoint."""

    arguments, checksum, _ = small_tuning_experiment
    directory = arguments[5] / arguments[0]
    (directory / EXPERIMENT_IDENTITY_FILENAME).unlink()
    weights_path = directory / "diagnostic_weights.pt"
    weights = torch.load(weights_path, weights_only=True)
    weights["model_state"]["current_regime_head.weight"].zero_()
    weights["model_state"]["current_regime_head.bias"] = torch.tensor([100., 0., 0.])
    torch.save(weights, weights_path)
    with pytest.raises(ValueError, match="scores do not reproduce"):
        run_experiment(*arguments, snapshot_sha256=checksum, allow_legacy_migration=True)
    assert not (directory / EXPERIMENT_IDENTITY_FILENAME).exists()


@pytest.mark.parametrize("change", ["labels", "roles", "test_rows", "manifest"])
def test_frozen_snapshot_or_manifest_changes_are_rejected(tmp_path, change):
    """Resume cannot quietly reinterpret historical labels or move the validation boundary."""

    import pricesanity.training.tuning as tuning

    dates = pd.date_range("2016-01-04", periods=130, freq="D")
    sessions = [pd.DataFrame({"session_date": [day.date()], "value": [1.]})
                for day in dates[:120]]
    snapshot_path = tmp_path / "tuning_snapshot.parquet"
    manifest_path = tmp_path / "manifest.json"
    frame = pd.concat(sessions, ignore_index=True)
    frame.to_parquet(snapshot_path)
    checksum = _semantic_snapshot_sha256(sessions)
    manifest = {
        "protocol": "expanding_history_v1", "run_index": 1, "test_evaluated": False,
        "training_dates": [str(day.date()) for day in dates[:100]],
        "validation_dates": [str(day.date()) for day in dates[100:120]],
        "reserved_test_dates": [str(day.date()) for day in dates[120:]],
        "training_validation_snapshot_sha256": checksum,
    }
    tuning.write_json(manifest_path, manifest)
    tuning._ensure_snapshot_identity(
        tmp_path, manifest, snapshot_path, checksum, allow_creation=True,
    )
    if change == "labels":
        frame.loc[0, "value"] = 2.
        frame.to_parquet(snapshot_path)
    elif change == "roles":
        manifest["validation_dates"].insert(0, manifest["training_dates"].pop())
    elif change == "test_rows":
        frame.loc[len(frame)] = [dates[120].date(), 1.]
        frame.to_parquet(snapshot_path)
    else:
        manifest["selection_rule"] = "tampered selection policy"
    tuning.write_json(manifest_path, manifest)
    with pytest.raises(ValueError):
        _, loaded_manifest, loaded_checksum = _load_frozen_tuning_split(
            snapshot_path, manifest_path,
        )
        tuning._ensure_snapshot_identity(
            tmp_path, loaded_manifest, snapshot_path, loaded_checksum, allow_creation=True,
        )


@pytest.mark.parametrize("device_name", ["cpu", "mps"])
def test_strict_context_device_step_inference_and_checkpoint(tmp_path, device_name):
    """The experimental path must also support AdamW, early closes, and portable weights."""

    from pricesanity.training.model import TransformerConfig
    from pricesanity.training.tuning_context import ContextExperimentTransformer

    if device_name == "mps" and not torch.backends.mps.is_available():
        pytest.skip("Apple MPS is unavailable in this execution environment")
    device = torch.device(device_name)
    configuration = TransformerConfig(dropout=0.0)
    model = ContextExperimentTransformer(configuration, 16).to(device)
    features = torch.randn(2, 81, 4, device=device)
    mask = torch.zeros(2, 81, dtype=torch.bool, device=device)
    mask[1, 42:] = True
    optimizer = torch.optim.AdamW(model.parameters())
    output = model(features, mask)
    loss = output.current_logits[~mask].square().mean()
    loss = loss + output.anticipated_logits[~mask].square().mean()
    loss.backward()
    assert all(torch.isfinite(parameter.grad).all() for parameter in model.parameters())
    optimizer.step()
    model.eval()
    path = tmp_path / "context.pt"
    torch.save(model.state_dict(), path)
    restored = ContextExperimentTransformer(configuration, 16).to(device)
    restored.load_state_dict(torch.load(path, weights_only=True, map_location=device))
    restored.eval()
    with torch.inference_mode():
        actual = model(features, mask)
        reloaded = restored(features, mask)
    torch.testing.assert_close(actual.current_logits, reloaded.current_logits)
    torch.testing.assert_close(actual.anticipated_logits, reloaded.anticipated_logits)
    assert next(restored.parameters()).device.type == device_name


def test_trailing_window_retains_absolute_session_position():
    """Moving the same window later in the day must use its later position embeddings."""

    from pricesanity.training.model import TransformerConfig
    from pricesanity.training.tuning_context import ContextExperimentTransformer

    model = ContextExperimentTransformer(TransformerConfig(dropout=0.0), 4).eval()
    features = torch.randn(1, 10, 4)
    mask = torch.zeros(1, 10, dtype=torch.bool)
    with torch.inference_mode():
        actual = model(features, mask).current_logits[:, 7]
        hidden = model.feature_projection(features[:, 4:8])
        hidden += model.position_embedding(torch.arange(4, 8))[None]
        encoded = model.encoder(hidden, mask=model._causal_attention_mask[:4, :4])
        expected = model.current_regime_head(encoded[:, -1])
    torch.testing.assert_close(actual, expected)


def test_context_windows_do_not_mix_sessions_or_early_close_padding():
    """Changing another session or padding must not change a real early-close prediction."""

    from pricesanity.training.model import TransformerConfig
    from pricesanity.training.tuning_context import ContextExperimentTransformer

    model = ContextExperimentTransformer(TransformerConfig(dropout=0.0), 4).eval()
    features = torch.randn(2, 10, 4)
    mask = torch.zeros(2, 10, dtype=torch.bool)
    mask[1, 3:] = True
    changed = features.clone()
    changed[0] += 100
    changed[1, 3:] -= 100
    with torch.inference_mode():
        original = model(features, mask)
        modified = model(changed, mask)
        unpadded = model(features[1:2, :3], mask[1:2, :3])
    for head in ("current_logits", "anticipated_logits"):
        expected = getattr(original, head)[1:2, :3]
        torch.testing.assert_close(expected, getattr(modified, head)[1:2, :3])
        torch.testing.assert_close(expected, getattr(unpadded, head))


def test_incompatible_partial_settings_are_preserved(small_tuning_experiment):
    """A restart may not silently replace a different interrupted experiment."""

    from dataclasses import asdict

    arguments, checksum, _ = small_tuning_experiment
    partial_arguments = list(arguments)
    partial_arguments[0] = "partial"
    directory = arguments[5] / "partial"
    directory.mkdir()
    path = directory / "settings.json"
    original_bytes = json.dumps(asdict(replace(arguments[4], context_length=16))).encode()
    path.write_bytes(original_bytes)
    with pytest.raises(ValueError, match="Partial experiment has different settings"):
        run_experiment(*partial_arguments, snapshot_sha256=checksum)
    assert path.read_bytes() == original_bytes


def test_checklist_passes_practical_context_to_width_and_keeps_test_out(tmp_path, monkeypatch):
    """A corrected ledger must actually steer later experiments, not only relabel the winner."""

    from dataclasses import asdict
    import pricesanity.training.tuning as tuning

    training_sessions = tuple(object() for _ in range(100))
    validation_sessions = tuple(object() for _ in range(20))
    settings = TuningSettings()
    baseline = _selection_candidate("baseline", 0.55, context=None)
    baseline.update(settings=asdict(settings), best_epoch=10)
    calls = []

    def record_candidate(name, training, validation, cache, candidate_settings, *args, **kwargs):
        assert training is training_sessions
        assert validation is validation_sessions
        calls.append((name, candidate_settings))
        score = {16: 0.589932, 32: 0.590212, 64: 0.57285}.get(
            candidate_settings.context_length, 0.55,
        )
        report = _selection_candidate(name, score, context=candidate_settings.context_length)
        report.update(settings=asdict(candidate_settings), best_epoch=10)
        return report

    monkeypatch.setattr(tuning, "run_experiment", record_candidate)
    result = tuning.run_checklist(
        baseline, training_sessions, validation_sessions, {}, tmp_path, torch.device("cpu"),
        snapshot_sha256="frozen",
    )
    widths = [settings for name, settings in calls if "width" in name]
    assert len(widths) == 4
    assert all(settings.context_length == 16 for settings in widths)
    assert all(settings.attention_head_count == 4 for settings in widths)
    assert {settings.model_dimension for settings in widths} == {12, 32, 48, 64}
    assert result["test_evaluated"] is False
    context_stage = next(stage for stage in result["stages"] if stage["stage"] == "context_length")
    assert context_stage["raw_best"] == "context_32"
    assert context_stage["selected"] == "context_16"
    assert json.loads((tmp_path / "stage_summary.json").read_text())["status"] == "complete"


def test_legacy_report_cannot_change_checkpoint_selection(small_tuning_experiment):
    """Before creating a checksum, require the report's loss to match its selected epoch."""

    arguments, checksum, _ = small_tuning_experiment
    directory = arguments[5] / arguments[0]
    (directory / EXPERIMENT_IDENTITY_FILENAME).unlink()
    path = directory / "report.json"
    report = json.loads(path.read_text())
    report["validation_loss"] += 0.5
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="checkpoint selection is invalid"):
        run_experiment(*arguments, snapshot_sha256=checksum, allow_legacy_migration=True)
    assert not (directory / EXPERIMENT_IDENTITY_FILENAME).exists()
