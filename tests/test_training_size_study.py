"""Verify fixed-duration research keeps the shared forward holdout out of fitting."""

from dataclasses import replace
import json

import pandas as pd
import pytest
import torch

from pricesanity.training import training_size_study as study
from pricesanity.training import tuning


def make_sessions():
    """Create enough chronological sessions to cover every fixed training prefix."""

    # One row per session isolates membership from sequence length: 150 historical sessions
    # precede the same 26-session holdout, and the opening feature reveals each prefix's mean.
    sessions = []
    for index, timestamp in enumerate(
        pd.date_range("2016-01-01", periods=176, tz="UTC")
    ):
        sessions.append(
            pd.DataFrame(
                {
                    "session_date": [timestamp.date()],
                    "ts_event": [timestamp],
                    "candlestick_id": [str(timestamp)],
                    "open_gap": [index / 1000],
                    "body": [0.01],
                    "high_from_close": [0.02],
                    "low_from_close": [-0.01],
                    "current_target": [index % 3],
                    "anticipated_target": [(index + 1) % 3],
                }
            )
        )
    return tuple(sessions)


def test_all_prefixes_share_the_exact_forward_holdout():
    """Every training size must face the same untouched chronological population."""

    sessions = make_sessions()
    for count in study.TRAINING_COUNTS:
        training, holdout = study.select_sessions(sessions, count)
        assert all(
            actual is expected
            for actual, expected in zip(training, sessions[:count])
        )
        assert len(training) == count
        assert len(holdout) == 26
        assert all(
            actual is expected
            for actual, expected in zip(holdout, sessions[150:])
        )

    with pytest.raises(ValueError):
        study.select_sessions(sessions, 151)


def test_fixed_training_is_fresh_train_only_and_reusable(tmp_path, monkeypatch):
    """Each prefix gets a fresh model and train-only scaling, then resumes immutably."""

    sessions = make_sessions()
    cache = {
        id(session): tuning.convert_session_to_tensors(
            session,
            timestamp_column="ts_event",
        )
        for session in sessions
    }
    settings = replace(
        tuning.TuningSettings(),
        model_dimension=12,
        epochs=1,
        context_length=4,
    )
    original_factory = study.ContextExperimentTransformer
    original_fit = study.fit_feature_standardizer
    models, initial_hashes, fitted_counts = [], [], []

    def capture_model(*args):
        model = original_factory(*args)
        models.append(model)
        initial_hashes.append(study.calculate_model_state_sha256(model))
        return model

    def capture_fit(training):
        assert len(training) in study.TRAINING_COUNTS
        assert all(session is sessions[index] for index, session in enumerate(training))
        fitted_counts.append(len(training))
        return original_fit(training)

    monkeypatch.setattr(study, 'ContextExperimentTransformer', capture_model)
    monkeypatch.setattr(study, 'fit_feature_standardizer', capture_fit)
    reports = []
    for count in study.TRAINING_COUNTS:
        training, holdout = study.select_sessions(sessions, count)
        reports.append(
            study.run_fixed_candidate(
                training,
                holdout,
                cache,
                settings,
                tmp_path,
                "frozen",
            )
        )
        saved = torch.load(
            tmp_path / f"train_{count:03d}" / "checkpoint.pt",
            weights_only=True,
        )
        torch.testing.assert_close(
            saved["feature_mean"],
            original_fit(training).mean,
        )

    assert fitted_counts == [100, 120, 140, 150]
    assert len({id(model) for model in models}) == 4
    assert len(set(initial_hashes)) == 1
    assert len({report["parameter_count"] for report in reports}) == 1
    assert all(report["epoch_count"] == 1 for report in reports)
    assert all(
        report["holdout_dates"] == reports[0]["holdout_dates"]
        for report in reports
    )

    def forbidden_epoch(*args, **kwargs):
        raise AssertionError("A complete candidate was retrained")

    # Resume must reuse the published checkpoint; retraining would make the learning-curve
    # point a different stochastic experiment despite having the same artifact identity.
    monkeypatch.setattr(study, "_run_epoch", forbidden_epoch)
    training, holdout = study.select_sessions(sessions, 100)
    checkpoint_path = tmp_path / "train_100" / "checkpoint.pt"
    previous_bytes = checkpoint_path.read_bytes()
    reused = study.run_fixed_candidate(
        training,
        holdout,
        cache,
        settings,
        tmp_path,
        "frozen",
    )
    assert reused["model_state_sha256"] == reports[0]["model_state_sha256"]
    assert checkpoint_path.read_bytes() == previous_bytes

    with pytest.raises(ValueError, match="identity changed"):
        study.run_fixed_candidate(
            training,
            holdout,
            cache,
            replace(settings, epochs=2),
            tmp_path,
            "frozen",
        )

    report_path = tmp_path / "train_100" / "report.json"
    report_path.write_text(report_path.read_text() + " ")
    with pytest.raises(ValueError, match="identity changed"):
        study.run_fixed_candidate(
            training,
            holdout,
            cache,
            settings,
            tmp_path,
            "frozen",
        )


def test_completed_tuning_directory_cannot_be_study_output(tmp_path):
    """A completed tuning pass remains read-only evidence for later size studies."""

    original = tmp_path / "tuning_pass_001"
    original.mkdir()
    plan = original / "study_plan.json"
    plan.write_text(json.dumps({"protected_directory": str(original)}))
    previous_bytes = plan.read_bytes()
    with pytest.raises(ValueError, match="read-only"):
        study.load_study(original)
    assert plan.read_bytes() == previous_bytes


def test_final_epoch_is_kept_even_when_reported_training_loss_increases(tmp_path, monkeypatch):
    """A learning curve must not quietly substitute minimum-training-loss checkpoints."""

    from dataclasses import replace as replace_metrics

    sessions = make_sessions()
    training, holdout = study.select_sessions(sessions, 100)
    cache = {
        id(session): tuning.convert_session_to_tensors(
            session,
            timestamp_column="ts_event",
        )
        for session in sessions
    }
    settings = replace(tuning.TuningSettings(), model_dimension=12, epochs=2)
    original_epoch = study._run_epoch
    trained_states = []

    def capture_epoch(model, *args, **kwargs):
        metrics = original_epoch(model, *args, **kwargs)
        if kwargs["optimizer"] is not None:
            trained_states.append(
                {
                    name: value.clone()
                    for name, value in model.state_dict().items()
                }
            )
            # Force worsening reported loss without changing real optimization. Saving the first
            # state would reveal accidental best-loss selection in this fixed-duration study.
            return replace_metrics(metrics, total_loss=float(len(trained_states)))
        assert len(trained_states) == 2
        return metrics

    monkeypatch.setattr(study, "_run_epoch", capture_epoch)
    study.run_fixed_candidate(
        training,
        holdout,
        cache,
        settings,
        tmp_path,
        "frozen",
    )
    weights = torch.load(
        tmp_path / "train_100" / "checkpoint.pt",
        weights_only=True,
    )
    assert all(
        torch.equal(value, trained_states[-1][name])
        for name, value in weights["model_state"].items()
    )
    assert any(
        not torch.equal(value, trained_states[0][name])
        for name, value in weights["model_state"].items()
    )
