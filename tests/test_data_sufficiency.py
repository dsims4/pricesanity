"""Development learning curves must change history size, never their evaluation population."""

from dataclasses import replace
from pathlib import Path
import json
import sqlite3

import numpy as np
import pandas as pd
import pytest

from pricesanity.annotation.schema import CandlestickAnnotation, MarketRegime
from pricesanity.annotation.store import AnnotationStore
from pricesanity.benchmark.analysis import paired_cluster_bootstrap_difference
from pricesanity.benchmark.data_sufficiency import plan_data_sufficiency, run_data_sufficiency
from pricesanity.benchmark.protocol import (
    load_benchmark_config, plan_benchmark, resolve_benchmark_config,
)
from pricesanity.benchmark.snapshot import freeze_benchmark_snapshot_from_sessions
from pricesanity.benchmark.sufficiency_analysis import (
    SCORE_NAMES, assess_curve, paired_curve_difference, summarize_curve_point,
)
from pricesanity.benchmark.sufficiency_data import load_development_sessions
from pricesanity.config import load_config
from pricesanity.data.annotation_evidence import EVIDENCE_ATTRIBUTE, EVIDENCE_POLICY
from pricesanity.data.identifiers import build_candlestick_id
from test_benchmark_snapshot import _sessions


def test_nested_prefixes_fixed_later_population():
    points = plan_data_sufficiency(536, (100, 200, 300, 400, 500), development_limit=2421)
    for smaller, larger in zip(points, points[1:]):
        assert set(range(smaller.training_end_index)) < set(range(larger.training_end_index))
    assert {(point.evaluation_start_index, point.evaluation_end_index) for point in points} == {
        (500, 536)
    }
    assert all(point.training_end_index <= point.evaluation_start_index for point in points)


@pytest.mark.parametrize("count,sizes", [(500, (500,)), (519, (500,)), (536, (200, 100)),
                                        (536, (100, 100)), (2422, (500,))])
def test_invalid_or_tiny_plans_fail(count, sizes):
    with pytest.raises(ValueError):
        plan_data_sufficiency(count, sizes, development_limit=2421)


def test_small_evaluation_requires_override_and_two_sessions():
    assert plan_data_sufficiency(
        510, (500,), development_limit=2421, allow_small_evaluation=True,
    )[0].evaluation_end_index == 510
    with pytest.raises(ValueError):
        plan_data_sufficiency(501, (500,), development_limit=2421, allow_small_evaluation=True)
    assert plan_benchmark(536, load_benchmark_config("configs/benchmark/default.yaml")).final_holdout.session_count == 54


def prediction_frame(seed=3):
    generator = np.random.default_rng(seed)
    frame = pd.DataFrame({
        "candlestick_id": [f"candle-{index}" for index in range(40)],
        "session_index": np.repeat(np.arange(5), 8),
    })
    for head in ("current", "anticipated"):
        for kind in ("human", "predicted"):
            frame[f"{kind}_{head}_regime"] = generator.choice(["bull", "bear", "range"], 40)
    return frame


def test_paired_bootstrap_matches_existing_pooled_metric():
    earlier = prediction_frame()
    later = earlier.copy()
    later.loc[:15, "predicted_current_regime"] = later.loc[:15, "human_current_regime"]
    difference = paired_curve_difference([earlier], [later], repetitions=75)
    existing = paired_cluster_bootstrap_difference(later, earlier, repetitions=75)
    mean = difference["scores"]["mean_head_macro_f1"]
    assert mean["gain"] == pytest.approx(existing.estimate)
    assert mean["lower"] == pytest.approx(existing.lower)
    assert mean["upper"] == pytest.approx(existing.upper)
    assert difference["scores"]["anticipated_macro_f1"]["gain"] == 0
    same = paired_curve_difference([earlier, later], [earlier, later], repetitions=30)
    assert all(score["lower"] == score["upper"] == 0 for score in same["scores"].values())


@pytest.mark.parametrize("mutation", ["order", "target", "id", "seed_count"])
def test_pairing_rejects_misalignment(mutation):
    earlier = prediction_frame()
    later = earlier.copy()
    if mutation == "order":
        later = later.iloc[::-1].reset_index(drop=True)
    elif mutation == "target":
        later["human_current_regime"] = "bull"
    elif mutation == "id":
        later.loc[0, "candlestick_id"] = "wrong"
    with pytest.raises(ValueError):
        paired_curve_difference([earlier], [] if mutation == "seed_count" else [later])


def test_seed_scores_remain_separate_from_pooled_sessions():
    first = prediction_frame()
    second = first.copy()
    for head in ("current", "anticipated"):
        second[f"predicted_{head}_regime"] = second[f"human_{head}_regime"]
    combined = summarize_curve_point([first, second])
    individual = [summarize_curve_point([frame]) for frame in (first, second)]
    for name in SCORE_NAMES:
        expected = [row[name] for row in individual]
        assert combined[name] == pytest.approx(np.mean(expected))
        assert combined[name + "_seed_sd"] == pytest.approx(np.std(expected, ddof=0))
    assert combined["mean_head_macro_f1"] == pytest.approx(
        (combined["current_macro_f1"] + combined["anticipated_macro_f1"]) / 2
    )


def gain(value, lower, upper, sd=0):
    scores = {name: {"gain": value, "lower": lower, "upper": upper,
                     "seed_difference_sd": sd} for name in SCORE_NAMES}
    return {"scores": scores, "per_100_sessions": scores}


@pytest.mark.parametrize("gains,expected", [
    ([gain(.02, .01, .025)] * 2, "STILL_RISING"),
    ([gain(.001, -.001, .004)] * 2, "PLAUSIBLY_PLATEAUING"),
    ([gain(.02, -.05, .10)] * 2, "INCONCLUSIVE"),
    ([gain(-.02, -.03, -.01), gain(.02, .01, .025)], "INCONCLUSIVE"),
])
def test_transparent_assessment(gains, expected):
    result = assess_curve(gains, evaluation_sessions=36, stochastic=False,
                          seed_count=1, latest_seed_sd=None)
    assert result["label"] == expected
    assert result["evaluation_strength"] == "preliminary"
    assert "user-defined" in result["interpretation"]


def test_thresholds_seed_noise_and_small_blocks_limit_claims():
    values = [gain(.02, .01, .025)] * 2
    for overrides in ({"meaningful_gain": .03}, {"evaluation_sessions": 10},
                      {"stochastic": True, "seed_count": 1},
                      {"stochastic": True, "latest_seed_sd": .03}, {"reference_only": True}):
        options = {"evaluation_sessions": 50, "stochastic": False,
                   "seed_count": 3, "latest_seed_sd": 0, **overrides}
        assert assess_curve(values, **options)["label"] == "INCONCLUSIVE"
    with pytest.raises(ValueError):
        assess_curve(values, evaluation_sessions=36, stochastic=False,
                     seed_count=1, latest_seed_sd=None, meaningful_gain=.001, small_gain=.005)


@pytest.fixture
def inputs(tmp_path):
    config = load_config("configs/default.yaml")
    config = replace(config, session=replace(config.session, end_time="11:10"))
    base = load_benchmark_config("configs/benchmark/default.yaml")
    rules = replace(
        base.rules,
        initial_training_fraction=0.20,
        learning_evaluation_fraction=0.25,
        fold_count=2,
        learning_curve_fractions=(0.50, 0.75, 1.0),
        minimum_training_sessions=1,
        minimum_validation_sessions=1,
        minimum_test_sessions=1,
        minimum_curve_sessions=1,
    )
    benchmark = resolve_benchmark_config(
        7,
        replace(
            base,
            rules=rules,
            output_root=tmp_path / "official",
        ),
    )
    timestamps = pd.DatetimeIndex([
        timestamp for day in pd.bdate_range("2016-01-04", periods=7)
        for timestamp in pd.date_range(day.strftime("%Y-%m-%d") + " 09:30", periods=20,
                                       freq="5min", tz="America/New_York").tz_convert("UTC")
    ])
    normalized = pd.DataFrame({
        "ts_event": timestamps, "instrument": "ES.v.0", "open_gap": 0.0,
        "body": 0.0, "high_from_close": .01, "low_from_close": -.01,
    })
    ohlc = pd.DataFrame({"ts_event": timestamps, "instrument": "ES.v.0",
                         "open": 100., "high": 101., "low": 99., "close": 100.})
    records = []
    for opening in timestamps[::20]:
        previous_open = opening - pd.Timedelta(days=1)
        records.append({
            "session_open": opening.isoformat(),
            "session_close": (opening + pd.Timedelta(minutes=100)).isoformat(),
            "condition": "available", "previous_condition": "available",
            "previous_session_open": previous_open.isoformat(),
            "previous_session_close": (previous_open + pd.Timedelta(minutes=100)).isoformat(),
            "reference_timestamp": (previous_open + pd.Timedelta(minutes=95)).isoformat(),
            "reference_close": 100.,
        })
    evidence = {
        "policy": EVIDENCE_POLICY, "instrument": "ES.v.0", "interval": "0 days 00:05:00",
        "timezone": "America/New_York", "session_start": "09:30", "session_end": "11:10",
        "sessions": records,
    }
    for frame in (normalized, ohlc):
        frame.attrs[EVIDENCE_ATTRIBUTE] = evidence
    normalized_path, ohlc_path = tmp_path / "normalized.parquet", tmp_path / "ohlc.parquet"
    normalized.to_parquet(normalized_path, index=False)
    ohlc.to_parquet(ohlc_path, index=False)
    database_path = tmp_path / "annotations.db"
    store = AnnotationStore(database_path)
    try:
        for index, timestamp in enumerate(timestamps):
            store.save(CandlestickAnnotation(
                build_candlestick_id("ES.v.0", timestamp, "5min"),
                list(MarketRegime)[index % 3], list(MarketRegime)[(index + 1) % 3],
            ))
    finally:
        store.close()
    # A forbidden final label is unparseable on purpose: any read through the normal label
    # conversion path would fail, making accidental holdout access visible in this fixture.
    with sqlite3.connect(database_path) as connection:
        connection.execute("UPDATE annotations SET current_regime = 'SEALED' "
                           "WHERE candlestick_id = ?", (build_candlestick_id(
                               "ES.v.0", timestamps[-1], "5min"),))
    return dict(normalized_path=normalized_path, candlestick_path=ohlc_path,
                database_path=database_path, app_config=config, benchmark_config=benchmark)


def test_loader_filters_before_loading_final_rows(inputs, monkeypatch):
    original_read = pd.read_parquet
    def guarded_read(path, **kwargs):
        assert kwargs.get("filters"), "Source rows must be filtered before entering Python"
        result = original_read(path, **kwargs)
        assert len(result) == 120
        return result
    monkeypatch.setattr(pd, "read_parquet", guarded_read)
    sessions, sources = load_development_sessions(**inputs)
    assert len(sessions) == 6
    assert sources["kind"] == "development_data_sufficiency"
    assert sources["development_session_limit"] == 6


def test_cropped_source_needs_partition_evidence(inputs):
    for path in (inputs["normalized_path"], inputs["candlestick_path"]):
        frame = pd.read_parquet(path).iloc[:120].copy()
        frame.attrs[EVIDENCE_ATTRIBUTE]["sessions"] = (
            frame.attrs[EVIDENCE_ATTRIBUTE]["sessions"][:6]
        )
        frame.to_parquet(path, index=False)
    with pytest.raises(ValueError, match="boundary"):
        load_development_sessions(**inputs)


def test_snapshot_guard_never_opens_sealed_file(inputs, monkeypatch, tmp_path):
    sessions, _ = load_development_sessions(**inputs)
    last = sessions[-1].copy()
    last["session_date"] = last.session_date + pd.Timedelta(days=3)
    last["ts_event"] = last.ts_event + pd.Timedelta(days=3)
    last["candlestick_id"] = "last-" + last.candlestick_id
    snapshot = freeze_benchmark_snapshot_from_sessions(
        [*sessions, last], output_directory=tmp_path / "official" / "snapshot",
        development_session_count=6,
    )
    original_open = Path.open
    def guarded_open(path, *args, **kwargs):
        assert path.name != "sealed_holdout.parquet"
        return original_open(path, *args, **kwargs)
    monkeypatch.setattr(Path, "open", guarded_open)
    loaded, sources = load_development_sessions(**inputs, benchmark_snapshot=snapshot.directory)
    assert len(loaded) == 6
    assert sources["partition_snapshot_identities"] == [snapshot.identity_sha256]


def test_real_adapters_artifacts_resumption_and_train_only_scaler(inputs, tmp_path, monkeypatch):
    settings = tmp_path / "settings.json"
    settings.write_text(json.dumps({
        "majority_class": {}, "previous_regime": {}, "logistic_regression": {},
        "transformer": {"model_dimension": 8, "attention_head_count": 2,
                        "layer_count": 1, "feedforward_dimension": 16, "epochs": 1},
    }))
    arguments = {**inputs, "output_directory": tmp_path / "diagnostic",
                 "model_names": ("majority_class", "previous_regime", "logistic_regression",
                                 "transformer"), "train_sizes": (1, 2, 3), "seeds": (42, 137),
                 "fixed_config": settings, "allow_small_evaluation": True,
                 "bootstrap_repetitions": 20}
    result = run_data_sufficiency(**arguments)
    assert len(result["runs"]) == 15
    assert result["curves"]["majority_class"][0]["seed_count"] == 1
    assert result["curves"]["transformer"][0]["seed_count"] == 2
    assert result["evaluation_session_count"] == 3
    assert all(value["label"] == "INCONCLUSIVE" for value in result["assessments"].values())
    population = json.loads((arguments["output_directory"] / "population.json").read_text())
    assert len(population["evaluation_candlestick_ids"]) == 15
    scaler = json.loads((arguments["output_directory"] / "standardizer_1.json").read_text())
    assert scaler["mean"] == pytest.approx([0, 0, .01, -.01])
    # Resume must validate frozen artifacts without rereading a now-changing annotation DB
    # or fitting even the inexpensive baselines a second time.
    import pricesanity.benchmark.data_sufficiency as diagnostic
    monkeypatch.setattr(
        diagnostic, "load_development_sessions", lambda *a, **k: pytest.fail("reread"),
    )
    monkeypatch.setattr(diagnostic, "run_model_once", lambda *a, **k: pytest.fail("refit"))
    resumed = run_data_sufficiency(**arguments, resume=True)
    assert resumed["curves"] == result["curves"]
    with pytest.raises(ValueError, match="identity"):
        run_data_sufficiency(**{**arguments, "train_sizes": (1, 3)}, resume=True)
    first_predictions = arguments["output_directory"] / result["runs"][0]["run_directory"]
    with (first_predictions / "predictions.parquet").open("ab") as artifact:
        artifact.write(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        run_data_sufficiency(**arguments, resume=True)


def test_interrupted_fit_resumes_same_model(inputs, tmp_path, monkeypatch):
    import pricesanity.benchmark.data_sufficiency as diagnostic
    arguments = {**inputs, "output_directory": tmp_path / "resume",
                 "model_names": ("majority_class",), "train_sizes": (1, 2),
                 "allow_small_evaluation": True, "bootstrap_repetitions": 20}
    original_save = diagnostic.save_benchmark_run
    monkeypatch.setattr(diagnostic, "save_benchmark_run", lambda *a, **k: (_ for _ in ()).throw(
        RuntimeError("interrupt after fitting")))
    with pytest.raises(RuntimeError, match="interrupt"):
        run_data_sufficiency(**arguments)
    monkeypatch.setattr(diagnostic, "save_benchmark_run", original_save)
    original_fit = diagnostic.run_model_once
    fitted_sizes = []
    def record_fit(*args, **kwargs):
        fitted_sizes.append(len(set(kwargs["training"].session_indices)))
        return original_fit(*args, **kwargs)
    monkeypatch.setattr(diagnostic, "run_model_once", record_fit)
    result = run_data_sufficiency(**arguments, resume=True)
    assert fitted_sizes == [2]
    assert len(result["runs"]) == 2


def test_cli_defaults_and_multiple_models(monkeypatch):
    from pricesanity.benchmark.cli import main
    import pricesanity.benchmark.data_sufficiency as diagnostic
    captured = {}
    monkeypatch.setattr(
        diagnostic, "run_data_sufficiency", lambda **kwargs: captured.update(kwargs),
    )
    assert main(["data-sufficiency", "--normalized", "n.parquet", "--candlesticks", "o.parquet",
                 "--database", "a.db", "--project-config", "configs/default.yaml",
                 "--output-directory", "data/models/data_sufficiency/check",
                 "--model", "transformer", "gru"]) == 0
    assert captured["model_names"] == ("transformer", "gru")
    assert captured["train_sizes"] == (100, 200, 300, 400, 500)
    assert captured["seeds"] is None


def test_window_reuse_matches_unique_candle_scaling_and_ignores_evaluation(
    inputs, tmp_path, monkeypatch,
):
    """The optimization must match the official preprocessing order numerically."""

    import pricesanity.benchmark.data_sufficiency as diagnostic
    from pricesanity.features.representations import (
        fit_unique_candle_standardizer, transform_sessions,
    )
    from pricesanity.features.specification import (
        EvaluationUniverse, RepresentationSpec, build_representation_corpus,
    )
    from pricesanity.features import FEATURE_COLUMNS

    sessions = _sessions(count=6, length=20)
    for session in sessions[3:]:
        session.loc[:, list(FEATURE_COLUMNS)] *= 1000
    monkeypatch.setattr(diagnostic, "load_development_sessions",
                        lambda *a, **k: (sessions, {"kind": "synthetic", "development_session_limit": 6}))
    original_run = diagnostic.run_model_once
    observed_means = []

    def checked_run(*args, **kwargs):
        size = len(set(kwargs["training"].session_indices))
        standardizer = fit_unique_candle_standardizer(
            sessions[:size], feature_columns=FEATURE_COLUMNS,
        )
        observed_means.append(standardizer.mean.copy())
        spec = RepresentationSpec("controlled_16", 16, "sequential", "none", None,
                                  "none", FEATURE_COLUMNS)
        for label, selected in (("training", sessions[:size]), ("evaluation", sessions[3:])):
            expected = build_representation_corpus(
                transform_sessions(selected, standardizer, feature_columns=FEATURE_COLUMNS),
                spec, EvaluationUniverse(15),
            )
            np.testing.assert_array_equal(kwargs[label].features, expected.features)
            assert kwargs[label].candlestick_ids == expected.candlestick_ids
        return original_run(*args, **kwargs)

    monkeypatch.setattr(diagnostic, "run_model_once", checked_run)
    diagnostic.run_data_sufficiency(
        **inputs, output_directory=tmp_path / "scaling", model_names=("majority_class",),
        train_sizes=(1, 2, 3), allow_small_evaluation=True, bootstrap_repetitions=10,
    )
    assert len(observed_means) == 3
    assert all(np.abs(mean).max() < 30 for mean in observed_means)


def test_failed_initialization_does_not_touch_annotation_database(inputs, tmp_path):
    from pricesanity.benchmark.artifacts import file_sha256

    checksum = file_sha256(inputs["database_path"])
    with pytest.raises(ValueError, match="20"):
        run_data_sufficiency(
            **inputs, output_directory=tmp_path / "too_small", model_names=("majority_class",),
            train_sizes=(1, 2, 3),
        )
    assert file_sha256(inputs["database_path"]) == checksum
    assert not (tmp_path / "too_small").exists()


def test_resume_rejects_changed_execution_environment(inputs, tmp_path, monkeypatch):
    import pricesanity.benchmark.data_sufficiency as diagnostic

    arguments = {**inputs, "output_directory": tmp_path / "environment",
                 "model_names": ("majority_class",), "train_sizes": (1, 2),
                 "allow_small_evaluation": True, "bootstrap_repetitions": 10}
    diagnostic.run_data_sufficiency(**arguments)
    monkeypatch.setattr(diagnostic, "resume_environment_fingerprint", lambda **kwargs: {})
    with pytest.raises(ValueError, match="environment"):
        diagnostic.run_data_sufficiency(**arguments, resume=True)



def test_mixed_official_snapshot_rejected_before_its_rows_are_read(inputs, tmp_path, monkeypatch):
    snapshot = freeze_benchmark_snapshot_from_sessions(
        _sessions(7, 20), output_directory=tmp_path / "mixed",
    )
    import pricesanity.benchmark.sufficiency_data as source_loader
    monkeypatch.setattr(source_loader, "load_benchmark_snapshot",
                        lambda *a, **k: pytest.fail("Mixed snapshot rows were opened"))
    with pytest.raises(ValueError, match="physically separate"):
        load_development_sessions(**inputs, benchmark_snapshot=snapshot.directory)
