import numpy as np
import pandas as pd
import pytest

from pricesanity.benchmark.artifacts import validate_benchmark_predictions
from pricesanity.benchmark.registry import (
    build_model,
    get_model_family,
    list_model_families,
)
from pricesanity.models.baselines import MajorityClassBaseline, PreviousRegimeBaseline
from pricesanity.models.protocol import PredictionContext
from pricesanity.models.sklearn_adapter import SklearnDualHeadAdapter


class FakeProbabilityEstimator:
    """Tiny pickleable estimator used without adding scikit-learn to core tests."""

    def fit(self, features, targets):
        self.classes_ = np.array([0, 1, 2])
        self.selected_class = int(np.bincount(targets, minlength=3).argmax())
        return self

    def predict(self, features):
        return np.full(len(features), self.selected_class, dtype=np.int64)

    def predict_proba(self, features):
        probabilities = np.zeros((len(features), 3), dtype=float)
        probabilities[:, self.selected_class] = 1.0
        return probabilities


def test_registry_contains_each_requested_family_once() -> None:
    """The benchmark plan names one representative per intended learning family."""

    families = list_model_families()
    names = [family.name for family in families]
    assert len(names) == len(set(names)) == 14
    assert {"tcn", "gru", "transformer", "gradient_boosting"}.issubset(names)
    assert all(family.available for family in families)


def test_histogram_boosting_is_seed_invariant_at_benchmark_scale() -> None:
    """No configured HGB random path is active for the benchmark's maximum fit size."""

    # WHY: 2,190 development sessions have at most 80 scored five-minute candles each,
    # staying below sklearn's 200,000-row randomized histogram-binning threshold.
    sample_count = 2_190 * 80
    generator = np.random.default_rng(20260915)
    features = generator.normal(size=(sample_count, 4)).astype(np.float32)
    current = ((features[:, 0] > 0).astype(int) + (features[:, 1] > 0).astype(int)) % 3
    anticipated = np.roll(current, 1)
    models = [
        build_model(
            "gradient_boosting",
            random_seed=seed,
            parameters={"max_iter": 2, "max_leaf_nodes": 7},
        )
        for seed in (11, 29)
    ]
    for model in models:
        estimator = model.estimator_factory()
        assert estimator.early_stopping is False
        assert estimator.max_features == 1.0
        model.fit(features, current, anticipated)

    assert get_model_family("gradient_boosting").stochastic is False
    sample = features[::997]
    first = models[0].predict_output(sample).probabilities
    second = models[1].predict_output(sample).probabilities
    np.testing.assert_array_equal(first.current, second.current)
    np.testing.assert_array_equal(first.anticipated, second.anticipated)


def test_majority_baseline_is_deterministic_and_training_only(tmp_path) -> None:
    """A class-frequency tie and saved reload resolve the same way every time."""

    features = np.ones((4, 64), dtype=np.float32)
    model = MajorityClassBaseline()
    model.fit(features, np.array([1, 0, 1, 0]), np.array([2, 2, 1, 2]))
    assert model.predict(features).current.tolist() == [0, 0, 0, 0]
    assert model.predict(features).anticipated.tolist() == [2, 2, 2, 2]

    path = tmp_path / "majority.json"
    model.save(path)
    loaded = MajorityClassBaseline.load(path)
    assert loaded.predict(features).current.tolist() == [0, 0, 0, 0]


def test_previous_regime_requires_explicit_prior_label_context() -> None:
    """Persistence cannot accidentally read current labels from the feature matrix."""

    model = PreviousRegimeBaseline()
    features = np.zeros((3, 64), dtype=np.float32)
    model.fit(features, np.array([0, 1, 2]), np.array([2, 1, 0]))
    with pytest.raises(ValueError, match="prior-label"):
        model.predict(features)
    context = PredictionContext(
        session_indices=np.array([0, 0, 0]),
        previous_current_targets=np.array([2, 0, 1]),
        previous_anticipated_targets=np.array([1, 2, 1]),
    )
    assert model.predict(features, context=context).current.tolist() == [2, 0, 1]


@pytest.mark.parametrize("model_name", ["tcn", "gru", "transformer"])
def test_sequence_model_can_fit_predict_save_and_reload(tmp_path, model_name) -> None:
    """Every advertised sequence family has a real tiny CPU execution path."""

    model = build_model(
        model_name,
        random_seed=42,
        parameters={"epochs": 1},
    )
    features = np.random.default_rng(42).normal(size=(9, 4, 4)).astype(np.float32)
    current = np.array([0, 1, 2] * 3)
    anticipated = np.array([2, 0, 1] * 3)
    model.fit(features, current, anticipated)
    output = model.predict_output(features)
    assert output.predictions.current.shape == (9,)
    assert output.probabilities.current.shape == (9, 3)

    path = tmp_path / f"{model_name}.pt"
    model.save(path)
    loaded = type(model).load(path)
    assert loaded.predict_output(features).predictions.current.shape == (9,)


def test_estimator_adapter_saves_fitted_state_without_factory(tmp_path) -> None:
    """A local construction callable is excluded from portable fitted artifacts."""

    local_factory = lambda: FakeProbabilityEstimator()
    model = SklearnDualHeadAdapter(
        "fake", local_factory, {"purpose": "test"}, standardize=True
    )
    features = np.arange(24, dtype=np.float32).reshape(6, 4)
    model.fit(features, np.array([0, 0, 0, 1, 1, 2]), np.array([2, 2, 1, 1, 1, 0]))
    path = tmp_path / "model.bin"
    model.save(path)

    loaded = SklearnDualHeadAdapter.load(path)
    assert loaded.predict(features).current.tolist() == [0] * 6
    assert loaded.predict(features).anticipated.tolist() == [1] * 6


@pytest.mark.parametrize("classes", [(0, 1), (0, 2), (1, 2)])
def test_binary_svc_scores_preserve_classes_and_serialize(tmp_path, classes):
    """Absent classes have finite placeholders without changing native binary margins."""

    features = np.array([[-2.0], [-1.0], [1.0], [2.0]], dtype=np.float32)
    targets = np.array([classes[0], classes[0], classes[1], classes[1]])
    model = build_model("rbf_svm", random_seed=42)
    model.fit(features, targets, targets[::-1])
    output = model.predict_output(features)
    assert output.probabilities is None
    assert output.uncertainty_kind == "uncalibrated_decision_score"

    transformed, current, anticipated = model._prepare(features, None)
    names = np.array(["bull", "bear", "range"])
    absent = next(iter({0, 1, 2}.difference(classes)))
    timestamps = pd.date_range("2026-01-05T14:30Z", periods=4, freq="5min")
    rows = pd.DataFrame({
        "candlestick_id": [f"binary-{i}" for i in range(4)],
        "timestamp": timestamps,
        "session_date": timestamps.date,
        "session_index": 0,
        "candle_position": np.arange(15, 19),
        "uncertainty_kind": output.uncertainty_kind,
    })
    for head, estimator, truth in (
        ("current", current, targets),
        ("anticipated", anticipated, targets[::-1]),
    ):
        scores = getattr(output.scores, head)
        predicted = getattr(output.predictions, head)
        margins = estimator.decision_function(transformed)
        assert scores.shape == (4, 3)
        assert np.isfinite(scores).all()
        np.testing.assert_allclose(scores[:, classes[0]], -margins)
        np.testing.assert_allclose(scores[:, classes[1]], margins)
        assert (scores[:, absent] < scores[:, classes].min(axis=1)).all()
        np.testing.assert_array_equal(predicted, estimator.predict(transformed))
        np.testing.assert_array_equal(predicted, truth)

        rows[f"predicted_{head}_regime"] = names[predicted]
        rows[f"human_{head}_regime"] = names[truth]
        for index, name in enumerate(names):
            rows[f"{head}_score_{name}"] = scores[:, index]
            # Score-only artifacts deliberately leave probability columns unavailable.
            rows[f"{head}_probability_{name}"] = np.nan

    validated = validate_benchmark_predictions(rows)
    path = tmp_path / "predictions.parquet"
    validated.to_parquet(path, index=False)
    pd.testing.assert_frame_equal(
        validate_benchmark_predictions(pd.read_parquet(path)), validated
    )
