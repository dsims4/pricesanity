"""Verify tuning diagnostics keep test data out and count transition neighborhoods honestly."""

from dataclasses import replace

import numpy as np
import pandas as pd
import torch

from pricesanity.training.dataset import convert_session_to_tensors
from pricesanity.training.tuning import (
    TuningSettings,
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
    """Tolerance regions stop at the session boundary and a new day's first label is not a change."""

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
                            tmp_path, torch.device("cpu"))
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


def test_single_head_diagnostic_does_not_update_other_head() -> None:
    """Ablating one loss leaves that output head unchanged while training the selected task."""

    from pricesanity.training.dataset import TensorSession, TensorSessionDataset
    from pricesanity.training.model import RegimeTransformer, TransformerConfig
    from pricesanity.training.trainer import FeatureStandardizer, _run_epoch
    from pricesanity.training.tuning import build_loader

    timestamps = tuple(pd.date_range("2016-01-04T14:30:00Z", periods=3, freq="5min"))
    session = TensorSession(timestamps[0].date(), tuple(map(str, timestamps)), timestamps,
                            torch.randn(3, 4), torch.tensor([0, 1, 2]), torch.tensor([2, 1, 0]))
    model = RegimeTransformer(TransformerConfig(dropout=0.0))
    previous_current = model.current_regime_head.weight.detach().clone()
    previous_anticipated = model.anticipated_regime_head.weight.detach().clone()
    _run_epoch(
        model, build_loader([session], TuningSettings(), shuffle=False),
        FeatureStandardizer(torch.zeros(4), torch.ones(4)), device=torch.device("cpu"),
        optimizer=torch.optim.AdamW(model.parameters(), lr=0.01), gradient_clip=1.0,
        loss_setup="current",
    )
    assert not torch.equal(previous_current, model.current_regime_head.weight)
    torch.testing.assert_close(previous_anticipated, model.anticipated_regime_head.weight)


def test_resume_uses_frozen_snapshot_without_reading_live_annotations(tmp_path, monkeypatch) -> None:
    """Additional annotation work must not change an in-progress tuning comparison."""

    import hashlib
    import json
    import sqlite3
    import pricesanity.training.tuning as tuning

    sessions = []
    for timestamp in pd.date_range("2016-01-04T14:30:00Z", periods=120, freq="D"):
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
        "training_dates": [str(session["session_date"].iloc[0]) for session in sessions[:100]],
        "validation_dates": [str(session["session_date"].iloc[0]) for session in sessions[100:]],
        "reserved_test_dates": ["reserved"],
        "training_validation_snapshot_sha256": snapshot.hexdigest(),
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
