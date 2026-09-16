"""Estimate benchmark shape and job count before costly searches begin."""

from dataclasses import asdict, dataclass
from math import comb


@dataclass(frozen=True)
class ScalabilityPreflight:
    """Transparent workload dimensions and educational scaling warnings."""

    training_samples: int
    evaluation_samples: int
    feature_dimension: int
    transformed_feature_dimension: int
    candidate_count: int
    fold_count: int
    seed_count: int
    approximate_job_count: int
    training_dense_float32_bytes: int
    training_dense_float64_bytes: int
    evaluation_dense_float32_bytes: int
    evaluation_dense_float64_bytes: int
    acknowledgement_required: bool
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def build_scalability_preflight(
    *,
    model_name: str,
    training_samples: int,
    evaluation_samples: int,
    feature_dimension: int,
    candidate_count: int,
    fold_count: int,
    seed_count: int,
    polynomial_degree: int | None = None,
) -> ScalabilityPreflight:
    """Calculate dimensions without fitting an estimator or exposing the holdout."""

    values = (
        training_samples, evaluation_samples, feature_dimension,
        candidate_count, fold_count, seed_count,
    )
    if any(value <= 0 for value in values):
        raise ValueError("Preflight dimensions must be positive.")
    transformed = feature_dimension
    if polynomial_degree is not None:
        if polynomial_degree < 1:
            raise ValueError("Polynomial degree must be positive.")
        transformed = comb(feature_dimension + polynomial_degree, polynomial_degree) - 1
    warnings = []
    acknowledgement_required = False
    if model_name == "rbf_svm" and training_samples >= 100_000:
        warnings.append(
            "Exact RBF SVM requires a declared scalability pilot before full-corpus launch."
        )
        acknowledgement_required = True
    if model_name == "knn" and evaluation_samples * training_samples >= 1_000_000_000:
        warnings.append("kNN inference compares evaluation rows with the training set.")
        acknowledgement_required = True
    if model_name == "polynomial_logistic" and transformed >= 10_000:
        warnings.append("Polynomial expansion creates a high-dimensional design matrix.")
        acknowledgement_required = True
    training_values = training_samples * transformed
    evaluation_values = evaluation_samples * transformed
    return ScalabilityPreflight(
        training_samples=training_samples,
        evaluation_samples=evaluation_samples,
        feature_dimension=feature_dimension,
        transformed_feature_dimension=transformed,
        candidate_count=candidate_count,
        fold_count=fold_count,
        seed_count=seed_count,
        # One seeded search chooses hyperparameters. Final seeds characterize only the frozen
        # winner; multiplying the whole search by final seeds would triple cost without purpose.
        approximate_job_count=candidate_count * fold_count + seed_count,
        training_dense_float32_bytes=training_values * 4,
        training_dense_float64_bytes=training_values * 8,
        evaluation_dense_float32_bytes=evaluation_values * 4,
        evaluation_dense_float64_bytes=evaluation_values * 8,
        acknowledgement_required=acknowledgement_required,
        warnings=tuple(warnings),
    )
