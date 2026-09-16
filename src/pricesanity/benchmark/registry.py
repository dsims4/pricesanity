"""Describe benchmark families and build their tested model adapters."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import importlib.util
from typing import Any

from pricesanity.models.baselines import MajorityClassBaseline, PreviousRegimeBaseline
from pricesanity.models.sklearn_adapter import SklearnDualHeadAdapter


@dataclass(frozen=True)
class ModelFamily:
    """Stable identity, representation, implementation, and tuning policy."""

    name: str
    display_name: str
    learning_family: str
    representation: str
    implementation: str
    tuning_required: bool
    stochastic: bool
    available: bool
    notes: str


MODEL_FAMILIES = (
    ModelFamily(
        "majority_class", "Majority class", "baseline", "none", "native",
        False, False, True, "Training-only class-frequency reference.",
    ),
    ModelFamily(
        "previous_regime", "Previous regime", "baseline", "prior_label", "native",
        False, False, True,
        "Uses the preceding human label and is not deployable without annotations.",
    ),
    ModelFamily(
        "gaussian_naive_bayes", "Gaussian Naive Bayes", "probabilistic", "tabular",
        "scikit-learn", True, False, True, "Independent Gaussian likelihoods.",
    ),
    ModelFamily(
        "logistic_regression", "Multinomial logistic regression", "linear", "tabular",
        "scikit-learn", True, False, True, "Linear multinomial boundary.",
    ),
    ModelFamily(
        "polynomial_logistic", "Polynomial logistic regression", "linear_interactions",
        "tabular", "scikit-learn", True, False, True,
        "Explicit polynomial interactions before logistic regression.",
    ),
    ModelFamily(
        "knn", "k-nearest neighbors", "neighbor", "tabular", "scikit-learn",
        True, False, True, "Distance-based local classifier.",
    ),
    ModelFamily(
        "decision_tree", "Decision tree", "tree", "tabular", "scikit-learn",
        True, False, True, "One interpretable axis-aligned tree.",
    ),
    ModelFamily(
        "random_forest", "Random forest", "tree_ensemble", "tabular", "scikit-learn",
        True, True, True, "Bagged trees with an explicit worker budget.",
    ),
    ModelFamily(
        "gradient_boosting", "Gradient-boosted trees", "boosting", "tabular",
        "scikit-learn", True, False, True,
        "Deterministic histogram-gradient boosting at the benchmark's row scale.",
    ),
    ModelFamily(
        "rbf_svm", "RBF SVM", "kernel", "tabular", "scikit-learn",
        True, False, True, "Native classes plus uncalibrated decision scores.",
    ),
    ModelFamily(
        "mlp", "MLP", "feedforward_neural", "tabular", "scikit-learn",
        True, True, True, "Flattened causal-window neural baseline.",
    ),
    ModelFamily(
        "tcn", "1D CNN / TCN", "convolutional_sequence", "sequential", "pytorch",
        True, True, True, "Small causal dilated convolutional encoder.",
    ),
    ModelFamily(
        "gru", "GRU", "recurrent_sequence", "sequential", "pytorch",
        True, True, True, "Small recurrent causal sequence encoder.",
    ),
    ModelFamily(
        "transformer", "Causal Transformer", "attention_sequence", "sequential",
        "existing_native_adapter", True, True, True,
        "Adapts the tested Price Sanity Transformer.",
    ),
)


# YAML uses educational names. Translation here prevents library keyword spelling from
# leaking into protocol configuration and catches drift before a long study starts.
MODEL_CONCEPTUAL_PARAMETERS: dict[str, frozenset[str]] = {
    "gaussian_naive_bayes": frozenset({"var_smoothing"}),
    "logistic_regression": frozenset({"C", "class_weight"}),
    "polynomial_logistic": frozenset({"degree", "C"}),
    "knn": frozenset({"n_neighbors", "weights"}),
    "decision_tree": frozenset({"max_depth", "min_samples_leaf"}),
    "random_forest": frozenset(
        {"n_estimators", "max_depth", "min_samples_leaf"}
    ),
    "gradient_boosting": frozenset(
        {"max_iter", "learning_rate", "max_leaf_nodes"}
    ),
    "rbf_svm": frozenset({"C", "gamma"}),
    "mlp": frozenset(
        {"hidden_width", "hidden_layers", "learning_rate_init", "max_iter"}
    ),
    "tcn": frozenset(
        {
            "channel_width", "kernel_size", "layer_count", "learning_rate", "epochs",
            "batch_size", "weight_decay",
            "context_length",
        }
    ),
    "gru": frozenset(
        {
            "hidden_size", "layer_count", "learning_rate", "epochs", "batch_size",
            "weight_decay",
            "context_length",
        }
    ),
    "transformer": frozenset(
        {
            "model_dimension", "attention_head_count", "layer_count",
            "feedforward_dimension", "dropout", "learning_rate", "epochs",
            "batch_size", "weight_decay",
            "context_length",
        }
    ),
}


def list_model_families() -> tuple[ModelFamily, ...]:
    """Return the stable ordered benchmark registry."""

    return MODEL_FAMILIES


def get_model_family(name: str) -> ModelFamily:
    """Resolve one model name without accepting silent aliases."""

    for model_family in MODEL_FAMILIES:
        if model_family.name == name:
            return model_family
    raise ValueError(f"Unknown benchmark model family: {name}")


def validate_conceptual_parameters(
    name: str,
    parameters: Mapping[str, Any],
) -> None:
    """Reject unsupported conceptual settings before constructing a library estimator."""

    accepted_parameters = MODEL_CONCEPTUAL_PARAMETERS.get(name, frozenset())
    if name in MODEL_CONCEPTUAL_PARAMETERS:
        accepted_parameters = accepted_parameters | {"context_length"}
    unknown_parameters = set(parameters).difference(accepted_parameters)
    if unknown_parameters:
        raise ValueError(
            f"Unsupported {name} parameters: "
            + ", ".join(sorted(unknown_parameters))
        )


def build_model(
    name: str,
    *,
    random_seed: int,
    parameters: dict[str, Any] | None = None,
    cpu_worker_count: int = 1,
    device: str = "cpu",
    prestandardized: bool = False,
) -> Any:
    """Build one runnable adapter from conceptual benchmark settings."""

    model_family = get_model_family(name)
    if not model_family.available:
        raise NotImplementedError(f"{model_family.display_name} is not implemented.")
    if cpu_worker_count <= 0:
        raise ValueError("Benchmark CPU worker count must be positive.")

    parameters = dict(parameters or {})
    validate_conceptual_parameters(name, parameters)
    # Context determines representation construction, not estimator architecture. Accept it in
    # family search spaces but keep it out of library constructors.
    parameters.pop("context_length", None)
    if name == "majority_class":
        return MajorityClassBaseline()
    if name == "previous_regime":
        return PreviousRegimeBaseline()
    if name in {"tcn", "gru", "transformer"}:
        from pricesanity.models.sequence import build_sequence_adapter

        return build_sequence_adapter(
            name,
            parameters={**parameters, "standardize": not prestandardized},
            random_seed=random_seed,
            device=device,
            cpu_worker_count=cpu_worker_count,
        )
    if importlib.util.find_spec("sklearn") is None:
        raise RuntimeError(
            "This model requires the optional benchmark dependency: "
            "python -m pip install -e '.[benchmark]'"
        )

    estimator_factory, configuration, standardize, output_kind = _sklearn_factory(
        name,
        conceptual_parameters=parameters,
        random_seed=random_seed,
        cpu_worker_count=cpu_worker_count,
    )
    return SklearnDualHeadAdapter(
        model_name=name,
        estimator_factory=estimator_factory,
        configuration=configuration,
        standardize=standardize and not prestandardized,
        output_kind=output_kind,
    )


def load_model(name: str, path: str, *, device: str = "cpu") -> Any:
    """Restore one persisted benchmark adapter through its registered family."""

    if name == "majority_class":
        return MajorityClassBaseline.load(path)
    if name == "previous_regime":
        return PreviousRegimeBaseline.load(path)
    if name in {"tcn", "gru", "transformer"}:
        from pricesanity.models.sequence import TorchSequenceAdapter

        return TorchSequenceAdapter.load(path, device=device)
    return SklearnDualHeadAdapter.load(path)


def _sklearn_factory(
    name: str,
    *,
    conceptual_parameters: dict[str, Any],
    random_seed: int,
    cpu_worker_count: int,
) -> tuple[Callable[[], Any], dict[str, Any], bool, str]:
    """Translate stable benchmark names into one estimator's library arguments."""

    if name == "gaussian_naive_bayes":
        from sklearn.naive_bayes import GaussianNB

        configuration = {"var_smoothing": 1e-9, **conceptual_parameters}
        return lambda: GaussianNB(**configuration), configuration, True, "probability_estimate"

    if name == "logistic_regression":
        from sklearn.linear_model import LogisticRegression

        configuration = {
            "C": 1.0,
            "class_weight": None,
            "max_iter": 1000,
            "random_state": random_seed,
            **conceptual_parameters,
        }
        return lambda: LogisticRegression(**configuration), configuration, True, "probability_estimate"

    if name == "polynomial_logistic":
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import PolynomialFeatures

        degree = int(conceptual_parameters.get("degree", 2))
        logistic_configuration = {
            "C": float(conceptual_parameters.get("C", 1.0)),
            "max_iter": 1000,
            "random_state": random_seed,
        }
        configuration = {"degree": degree, **logistic_configuration}
        return (
            lambda: make_pipeline(
                PolynomialFeatures(degree=degree, include_bias=False),
                LogisticRegression(**logistic_configuration),
            ),
            configuration,
            True,
            "probability_estimate",
        )

    if name == "knn":
        from sklearn.neighbors import KNeighborsClassifier

        configuration = {
            "n_neighbors": 15,
            "weights": "uniform",
            **conceptual_parameters,
        }
        return lambda: KNeighborsClassifier(**configuration), configuration, True, "probability_estimate"

    if name == "decision_tree":
        from sklearn.tree import DecisionTreeClassifier

        configuration = {
            "max_depth": 8,
            "min_samples_leaf": 1,
            "random_state": random_seed,
            **conceptual_parameters,
        }
        return lambda: DecisionTreeClassifier(**configuration), configuration, False, "probability_estimate"

    if name == "random_forest":
        from sklearn.ensemble import RandomForestClassifier

        configuration = {
            "n_estimators": 300,
            "max_depth": 12,
            "min_samples_leaf": 1,
            "random_state": random_seed,
            "n_jobs": cpu_worker_count,
            **conceptual_parameters,
        }
        return lambda: RandomForestClassifier(**configuration), configuration, False, "probability_estimate"

    if name == "gradient_boosting":
        from sklearn.ensemble import HistGradientBoostingClassifier

        configuration = {
            "max_iter": 200,
            "learning_rate": 0.05,
            "max_leaf_nodes": 31,
            # Random internal validation leaks overlapping windows across its row split.
            # Chronological outer folds select max_iter instead.
            "early_stopping": False,
            # WHY: This seed is inert for the current <200k-row benchmark with full feature
            # use, but retaining it keeps existing configurations/artifacts compatible.
            "random_state": random_seed,
            **conceptual_parameters,
        }
        return lambda: HistGradientBoostingClassifier(**configuration), configuration, False, "probability_estimate"

    if name == "rbf_svm":
        from sklearn.svm import SVC

        configuration = {
            "C": 1.0,
            "gamma": "scale",
            "kernel": "rbf",
            "decision_function_shape": "ovr",
            "random_state": random_seed,
            **conceptual_parameters,
        }
        return lambda: SVC(**configuration), configuration, True, "uncalibrated_score"

    if name == "mlp":
        from sklearn.neural_network import MLPClassifier

        hidden_width = int(conceptual_parameters.get("hidden_width", 64))
        hidden_layers = int(conceptual_parameters.get("hidden_layers", 2))
        configuration = {
            "hidden_layer_sizes": (hidden_width,) * hidden_layers,
            "learning_rate_init": float(
                conceptual_parameters.get("learning_rate_init", 1e-3)
            ),
            "max_iter": int(conceptual_parameters.get("max_iter", 300)),
            # MLP early stopping randomly withholds overlapping rows. The active search space
            # keeps max_iter fixed at this value while outer folds select the exposed settings.
            "early_stopping": False,
            "random_state": random_seed,
        }
        return lambda: MLPClassifier(**configuration), configuration, True, "probability_estimate"

    raise ValueError(f"No scikit-learn implementation is registered for {name}.")
