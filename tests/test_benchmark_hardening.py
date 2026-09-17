from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest
import torch

from pricesanity.benchmark.aggregation import aggregate_seed_results
from pricesanity.benchmark.analysis import (
    bootstrap_session_mean,
    cluster_bootstrap_pooled_f1,
    paired_cluster_bootstrap_difference,
    paired_session_bootstrap_difference,
)
from pricesanity.benchmark.registry import build_model
from pricesanity.benchmark.search_spaces import (
    load_search_spaces,
    representative_candidate,
)
from pricesanity.features import (
    EvaluationUniverse,
    FEATURE_COLUMNS,
    RepresentationSpec,
    assert_same_evaluation_universe,
    build_representation_corpus,
)
from pricesanity.models.sequence import RegimeGRU, RegimeTCN, TorchSequenceAdapter
from pricesanity.models.sklearn_adapter import SklearnDualHeadAdapter


def _long_session(session_index: int, length: int = 70) -> pd.DataFrame:
    """Expose candle position in values so context and causal-boundary errors are visible."""

    session_date = date(2020, 1, 2) + timedelta(days=session_index)
    values = np.arange(length, dtype=np.float32)
    return pd.DataFrame({
        "session_date": [session_date] * length,
        "candlestick_id": [f"{session_index}-{position}" for position in range(length)],
        "ts_event": pd.date_range(
            pd.Timestamp(session_date, tz="UTC") + pd.Timedelta(hours=14, minutes=30),
            periods=length,
            freq="5min",
        ),
        "open_gap": values,
        "body": values / 10,
        "high_from_close": values / 20,
        "low_from_close": -values / 20,
        "current_target": np.arange(length) % 3,
        "anticipated_target": (np.arange(length) + 1) % 3,
    })


def test_context_lengths_score_the_same_ordered_candles() -> None:
    """A longer context cannot improve its score by skipping difficult early targets."""

    # The second synthetic session mirrors an early close: it is shorter than the 64-candle
    # context but must still contribute exactly the same target IDs to every family.
    sessions = [_long_session(0), _long_session(1, length=42)]
    universe = EvaluationUniverse(first_scored_candle_position=1)
    common = dict(
        layout="sequential",
        standardization="training_only",
        polynomial_degree=None,
        padding_policy="left_zero_masked",
        feature_columns=FEATURE_COLUMNS,
    )
    short = build_representation_corpus(
        sessions,
        RepresentationSpec(name="short", window_length=16, **common),
        universe,
    )
    long = build_representation_corpus(
        sessions,
        RepresentationSpec(name="long", window_length=64, **common),
        universe,
    )
    assert short.features.shape[1] == 16
    assert long.features.shape[1] == 64
    assert set(long.session_dates) == {date(2020, 1, 2), date(2020, 1, 3)}
    assert not long.valid_history_mask[0, :-2].any()
    assert_same_evaluation_universe(short, long)


@pytest.mark.parametrize("model_type", [RegimeTCN, RegimeGRU])
def test_sequence_architectures_do_not_read_future_positions(model_type) -> None:
    """Changing a later candle cannot alter logits emitted for an earlier candle."""

    torch.manual_seed(42)
    model = model_type(feature_count=4)
    original = torch.randn(2, 8, 4)
    changed = original.clone()
    changed[:, 6:, :] += 1000.0
    model.eval()
    with torch.inference_mode():
        first = model(original).current_logits[:, :6]
        second = model(changed).current_logits[:, :6]
    torch.testing.assert_close(first, second)


@pytest.mark.parametrize("model_name", ["tcn", "gru", "transformer"])
def test_sequence_initialization_and_cpu_training_follow_declared_seed(model_name) -> None:
    """Construction and DataLoader order cannot depend on ambient PyTorch RNG state."""

    parameters = {
        "epochs": 2,
        "batch_size": 4,
        "learning_rate": 1e-3,
        "weight_decay": 0.0,
    }
    parameters.update({
        "tcn": {"channel_width": 4, "kernel_size": 2, "layer_count": 1},
        "gru": {"hidden_size": 4, "layer_count": 1},
        "transformer": {
            "model_dimension": 4, "attention_head_count": 1, "layer_count": 1,
            "feedforward_dimension": 8, "dropout": 0.0,
        },
    }[model_name])
    torch.manual_seed(999)
    first = build_model(model_name, random_seed=17, parameters=parameters)
    torch.manual_seed(1234)
    second = build_model(model_name, random_seed=17, parameters=parameters)
    third = build_model(model_name, random_seed=18, parameters=parameters)
    first_state = first._model.state_dict()
    second_state = second._model.state_dict()
    third_state = third._model.state_dict()
    assert all(torch.equal(first_state[name], second_state[name]) for name in first_state)
    assert any(not torch.equal(first_state[name], third_state[name]) for name in first_state)

    generator = np.random.default_rng(42)
    features = generator.normal(size=(12, 4, 4)).astype(np.float32)
    current = np.array([0, 1, 2] * 4, dtype=np.int64)
    anticipated = np.roll(current, 1)
    first.fit(features, current, anticipated)
    second.fit(features, current, anticipated)
    first_output = first.predict_output(features)
    second_output = second.predict_output(features)
    np.testing.assert_array_equal(
        first_output.predictions.current, second_output.predictions.current
    )
    np.testing.assert_array_equal(
        first_output.predictions.anticipated,
        second_output.predictions.anticipated,
    )
    np.testing.assert_array_equal(
        first_output.probabilities.current, second_output.probabilities.current
    )


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS is unavailable"
)
@pytest.mark.parametrize("model_name", ["tcn", "gru", "transformer"])
def test_sequence_benchmark_adapter_mps_smoke(model_name, tmp_path) -> None:
    """Exercise the real adapter on MPS without silently substituting CPU."""

    parameters = {"epochs": 1, "batch_size": 3, "learning_rate": 1e-3}
    parameters.update({
        "tcn": {"channel_width": 4, "kernel_size": 2, "layer_count": 1},
        "gru": {"hidden_size": 4, "layer_count": 1},
        "transformer": {
            "model_dimension": 4, "attention_head_count": 1, "layer_count": 1,
            "feedforward_dimension": 8, "dropout": 0.0,
        },
    }[model_name])
    model = build_model(
        model_name, random_seed=42, parameters=parameters, device="mps"
    )
    features = np.random.default_rng(1).normal(size=(6, 4, 4)).astype(np.float32)
    targets = np.array([0, 1, 2, 0, 1, 2], dtype=np.int64)
    model.fit(features, targets, np.roll(targets, 1))
    before = model.predict_output(features)
    path = tmp_path / f"{model_name}.bin"
    model.save(path)
    restored = TorchSequenceAdapter.load(path, device="mps")
    after = restored.predict_output(features)
    np.testing.assert_array_equal(before.predictions.current, after.predictions.current)


def test_every_search_space_builds_fits_predicts_and_reloads(tmp_path) -> None:
    """Real candidate smoke fits catch drift between readable YAML and adapters."""

    spaces = load_search_spaces("configs/benchmark/search_spaces.yaml")
    generator = np.random.default_rng(42)
    targets = np.array([0, 1, 2] * 4)
    # Short fits validate estimator compatibility and round trips, not convergence. All three
    # labels are present so every family's ordinary multiclass output path is exercised.
    for model_name, space in spaces.items():
        parameters = representative_candidate(space)
        if model_name == "mlp":
            parameters["max_iter"] = 5
        if model_name in {"tcn", "gru", "transformer"}:
            parameters["epochs"] = 1
            features = generator.normal(size=(12, 4, 4)).astype(np.float32)
        else:
            features = generator.normal(size=(12, 8)).astype(np.float32)
        model = build_model(model_name, random_seed=42, parameters=parameters)
        model.fit(features, targets, np.roll(targets, 1))
        output = model.predict_output(features)
        assert output.predictions.current.shape == (12,)
        assert (
            output.probabilities is not None
            or output.scores is not None
        )
        path = tmp_path / f"{model_name}.bin"
        model.save(path)
        loader = (
            TorchSequenceAdapter.load
            if model_name in {"tcn", "gru", "transformer"}
            else SklearnDualHeadAdapter.load
        )
        assert loader(path).predict_output(features).predictions.current.shape == (12,)


def test_svc_native_prediction_can_disagree_with_probability_argmax() -> None:
    """The artifact contract must not assert an equality libsvm itself does not promise."""

    from sklearn.svm import SVC

    training = np.array([
        [0.125730, -0.132105], [0.640423, 0.104900], [-0.535669, 0.361595],
        [1.304000, 0.947081], [-0.703735, -1.265421], [-0.623274, 0.041326],
        [-2.325031, -0.218792], [-1.245911, -0.732267], [-0.544259, -0.316300],
        [0.411631, 1.042514], [-0.128535, 1.366463], [-0.665195, 0.351510],
    ])
    targets = np.array([2, 1, 1, 1, 2, 0, 0, 2, 0, 2, 0, 1])
    test = np.array([[0.540846, 0.214659]])
    model = SVC(probability=True, random_state=0).fit(training, targets)
    native = model.predict(test)
    probability_class = model.classes_[model.predict_proba(test).argmax(axis=1)]
    assert native.tolist() != probability_class.tolist()


def test_seed_aggregation_never_selects_one_lucky_seed() -> None:
    # Deliberately uneven scores make choosing the best seed differ from the required mean.
    rows = []
    for seed, current, anticipated in [(1, 0.2, 0.4), (2, 0.8, 0.6), (3, 0.5, 0.5)]:
        rows.append({
            "track": "controlled", "model_name": "mlp",
            "model_configuration_sha256": "a", "representation_sha256": "b",
            "test_session_ids_sha256": "c", "seed": seed,
            "current_macro_f1": current,
            "anticipated_macro_f1": anticipated,
        })
    aggregated = aggregate_seed_results(
        pd.DataFrame(rows), stochastic_models=["mlp"],
        expected_stochastic_seed_count=3,
    )
    assert aggregated.loc[0, "current_macro_f1_mean"] == pytest.approx(0.5)
    assert aggregated.loc[0, "mean_head_macro_f1"] == pytest.approx(0.5)
    with pytest.raises(ValueError, match="requires 3"):
        aggregate_seed_results(
            pd.DataFrame(rows[:1]), stochastic_models=["mlp"],
            expected_stochastic_seed_count=3,
        )


@pytest.mark.parametrize("fingerprints, comparable", [
    ([{"device": "cpu"}, {"device": "cpu"}], True),
    ([{"device": "cpu"}, {"device": "mps"}], False),
    ([{"device": "cpu"}, None], False),
    ([None, None], False),
])
def test_seed_timing_requires_complete_compatible_hardware(fingerprints, comparable):
    """Missing timing provenance must not remove valid predictive evidence."""

    rows = pd.DataFrame([
        {
            "track": "controlled", "model_name": "mlp",
            "model_configuration_sha256": "a", "representation_sha256": "b",
            "test_session_ids_sha256": "c", "seed": seed,
            "current_macro_f1": score, "anticipated_macro_f1": score,
            "hardware_fingerprint": fingerprint,
            "training_seconds": 2.0 * seed,
            "inference_seconds": float(seed),
            "inference_samples_per_second": 10.0 * seed,
        }
        for seed, score, fingerprint in zip((1, 2), (0.25, 0.75), fingerprints)
    ])
    result = aggregate_seed_results(
        rows, stochastic_models=["mlp"], expected_stochastic_seed_count=2
    ).iloc[0]
    assert bool(result["hardware_compatible"]) is comparable
    assert result["seed_count"] == 2
    assert result["mean_head_macro_f1"] == pytest.approx(0.5)
    assert result["individual_current_macro_f1"] == (0.25, 0.75)
    for field, expected in (
        ("training_seconds_mean", 3.0),
        ("inference_seconds_mean", 1.5),
        ("inference_samples_per_second_mean", 15.0),
    ):
        if comparable:
            assert result[field] == pytest.approx(expected)
        else:
            assert pd.isna(result[field])
    assert result["hardware_fingerprint"] == ({"device": "cpu"} if comparable else None)


def test_session_bootstrap_is_reproducible_and_paired() -> None:
    first = pd.DataFrame({"session_index": [1, 2, 3], "macro_f1": [0.6, 0.7, 0.8]})
    second = pd.DataFrame({"session_index": [1, 2, 3], "macro_f1": [0.5, 0.6, 0.7]})
    one = bootstrap_session_mean(first["macro_f1"].to_numpy(), repetitions=100)
    two = bootstrap_session_mean(first["macro_f1"].to_numpy(), repetitions=100)
    assert one == two
    difference = paired_session_bootstrap_difference(first, second, repetitions=100)
    assert difference.estimate == pytest.approx(0.1)
    with pytest.raises(ValueError, match="identical ordered sessions"):
        paired_session_bootstrap_difference(first, second.iloc[::-1])


def test_cluster_bootstrap_recomputes_pooled_f1_with_absent_session_classes() -> None:
    """Whole-session resampling keeps the fixed three-class absent-class convention."""

    predictions = pd.DataFrame({
        "candlestick_id": ["a", "b", "c", "d"],
        "session_index": [0, 0, 1, 1],
        "human_current_regime": ["bull", "bull", "bear", "bear"],
        "predicted_current_regime": ["bull", "bull", "bear", "bull"],
        "human_anticipated_regime": ["range", "range", "bear", "bear"],
        "predicted_anticipated_regime": ["range", "bull", "bear", "bear"],
    })
    first = cluster_bootstrap_pooled_f1(
        predictions, repetitions=100, random_seed=11
    )
    second = cluster_bootstrap_pooled_f1(
        predictions, repetitions=100, random_seed=11
    )
    assert first == second
    assert first.current.session_count == 2
    # Range has zero support in the current head but still contributes zero to the fixed
    # three-class macro average, matching the primary benchmark metric.
    assert first.current.estimate < 1.0

    improved = predictions.copy()
    improved["predicted_current_regime"] = improved["human_current_regime"]
    improved["predicted_anticipated_regime"] = improved["human_anticipated_regime"]
    difference = paired_cluster_bootstrap_difference(
        improved, predictions, repetitions=100, random_seed=11
    )
    assert difference.estimate > 0
    with pytest.raises(ValueError, match="identical ordered candles"):
        paired_cluster_bootstrap_difference(improved.iloc[::-1], predictions)
