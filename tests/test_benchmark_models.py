import numpy as np
import pytest

from pricesanity.benchmark.registry import build_model, list_model_families
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
