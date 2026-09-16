"""Validate portable run bundles without Qt, inference, or annotation writes."""

import hashlib
import json
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import torch

from pricesanity.config import AppConfig
from pricesanity.data.identifiers import build_candlestick_id
from pricesanity.training.dataset import FEATURE_COLUMNS
from pricesanity.training.trainer import load_training_checkpoint


def file_sha256(path: Path) -> str:
    """Hash artifact bytes without loading the entire file into memory."""

    with path.open("rb") as artifact_file:
        return hashlib.file_digest(artifact_file, "sha256").hexdigest()


def resolve_run_artifact(run_directory: Path, recorded_path: object) -> Path:
    """Resolve local artifacts, including basenames from older absolute-path metadata."""

    # A moved bundle must never accidentally load the original directory's model.
    # Version-one runs used absolute paths; their local basename is the migration path.
    artifact_path = Path(str(recorded_path))
    if artifact_path.is_absolute():
        artifact_path = Path(artifact_path.name)

    resolved_path = (run_directory / artifact_path).resolve()
    if not resolved_path.is_relative_to(run_directory.resolve()):
        raise ValueError("Run artifact paths must remain inside the run directory.")

    return resolved_path


PREDICTION_COLUMNS = (
    "run_index",
    "model_state_sha256",
    "candlestick_id",
    "timestamp",
    "session_date",
    "predicted_current_regime",
    "predicted_anticipated_regime",
    "current_probability_bull",
    "current_probability_bear",
    "current_probability_range",
    "anticipated_probability_bull",
    "anticipated_probability_bear",
    "anticipated_probability_range",
    "human_current_regime",
    "human_anticipated_regime",
)


def load_run_artifacts(
    run_directory: str | Path,
    *,
    config: AppConfig,
) -> tuple[dict, pd.DataFrame]:
    """Read a completed bundle and verify identities, ordering, and prediction integrity.

    No inference or database access occurs. Version-one metadata may omit checksums,
    but its checkpoint must still satisfy the current model-artifact contract.
    """

    run_directory = Path(run_directory)
    metadata_path = run_directory / "run_metadata.json"
    prediction_path = run_directory / "test_predictions.parquet"
    try:
        with metadata_path.open(encoding="utf-8") as metadata_file:
            metadata = json.load(metadata_file)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not load run metadata: {metadata_path}") from error
    if not isinstance(metadata, dict):
        raise ValueError("Run metadata must contain a named mapping.")

    required_metadata = {
        "run_index",
        "run_count",
        "test",
        "instrument",
        "target_interval",
        "session_timezone",
        "checkpoint_path",
        "prediction_path",
        "model_state_sha256",
    }
    missing_metadata = required_metadata.difference(metadata)
    if missing_metadata:
        raise ValueError(
            "Run metadata is missing fields: "
            + ", ".join(sorted(missing_metadata))
        )
    if metadata["instrument"] != config.data.instrument:
        raise ValueError("Run instrument does not match the project configuration.")
    if metadata["target_interval"] != config.data.target_interval:
        raise ValueError("Run interval does not match the project configuration.")
    if metadata["session_timezone"] != config.data.session_timezone:
        raise ValueError("Run timezone does not match the project configuration.")

    checkpoint_path = resolve_run_artifact(run_directory, metadata["checkpoint_path"])
    prediction_path = resolve_run_artifact(run_directory, metadata["prediction_path"])

    # Hashes bind the displayed predictions and the complete checkpoint (including scaling)
    # to this completed experiment. Check bytes before deserialization or display.
    version = metadata.get("format_version", 1)
    if version not in (1, 2):
        raise ValueError("Unsupported run metadata format version.")

    for field, artifact_path in (
        ("test_predictions_sha256", prediction_path),
        ("checkpoint_sha256", checkpoint_path),
    ):
        expected_checksum = metadata.get(field)
        if expected_checksum is None:
            if version == 2:
                raise ValueError(f"Run metadata is missing {field}.")
            warnings.warn(f"Older run has no {field}; integrity is limited.", UserWarning)
        elif file_sha256(artifact_path) != expected_checksum:
            raise ValueError(f"Run artifact checksum mismatch: {artifact_path.name}")

    # Reconstructing through the shared loader proves the prediction file names
    # a valid checkpoint with the expected feature order and class vocabulary.
    loaded_checkpoint = load_training_checkpoint(
        checkpoint_path,
        device=torch.device("cpu"),
        expected_feature_columns=FEATURE_COLUMNS,
    )
    if (
        loaded_checkpoint.metadata["model_state_sha256"]
        != metadata["model_state_sha256"]
    ):
        raise ValueError("Run metadata and checkpoint identify different model weights.")

    try:
        predictions = pd.read_parquet(prediction_path)
    except OSError as error:
        raise ValueError(f"Could not load test predictions: {prediction_path}") from error
    missing_prediction_columns = set(PREDICTION_COLUMNS).difference(
        predictions.columns
    )
    if missing_prediction_columns:
        raise ValueError(
            "Test predictions are missing columns: "
            + ", ".join(sorted(missing_prediction_columns))
        )
    if predictions.empty:
        raise ValueError("Test predictions cannot be empty.")

    predictions = predictions.loc[:, PREDICTION_COLUMNS].copy()
    predictions["timestamp"] = pd.to_datetime(
        predictions["timestamp"],
        utc=True,
        errors="raise",
    )
    predictions["session_date"] = pd.to_datetime(
        predictions["session_date"],
        errors="raise",
    ).dt.date
    if (
        predictions["candlestick_id"].isna().any()
        or predictions["candlestick_id"].duplicated().any()
        or predictions["timestamp"].duplicated().any()
        or not predictions["timestamp"].is_monotonic_increasing
    ):
        raise ValueError("Test predictions must be uniquely identified and chronological.")

    expected_run_index = int(metadata["run_index"])
    if not predictions["run_index"].eq(expected_run_index).all():
        raise ValueError("Test prediction run index does not match run metadata.")
    if not predictions["model_state_sha256"].eq(
        metadata["model_state_sha256"]
    ).all():
        raise ValueError("Test predictions identify different model weights.")

    valid_regimes = {"bull", "bear", "range"}
    regime_columns = (
        "predicted_current_regime",
        "predicted_anticipated_regime",
        "human_current_regime",
        "human_anticipated_regime",
    )
    if any(
        not set(predictions[regime_column]).issubset(valid_regimes)
        for regime_column in regime_columns
    ):
        raise ValueError("Test predictions contain an unknown regime label.")

    probability_groups = (
        (
            "current_probability_bull",
            "current_probability_bear",
            "current_probability_range",
        ),
        (
            "anticipated_probability_bull",
            "anticipated_probability_bear",
            "anticipated_probability_range",
        ),
    )
    for probability_columns in probability_groups:
        probability_values = predictions.loc[:, probability_columns].to_numpy(
            dtype=float
        )
        if (
            not np.isfinite(probability_values).all()
            or (probability_values < 0.0).any()
            or (probability_values > 1.0).any()
            or not np.allclose(probability_values.sum(axis=1), 1.0, atol=1e-5)
        ):
            raise ValueError("Test prediction probabilities are invalid.")

    # Identical weights alone are not enough: two runs could share weights but differ in
    # dates, scaling, or labels. Compare the experiment recorded inside the checkpoint too.
    checkpoint_experiment = loaded_checkpoint.metadata["experiment"]
    if not isinstance(checkpoint_experiment, dict):
        raise ValueError("Checkpoint experiment must be a named mapping.")

    for field in (
        "run_index", "training", "validation", "test", "instrument",
        "target_interval", "session_timezone", "best_epoch", "experiment_signature",
        "test_predictions_sha256", "protocol", "initialization_mode",
        "initial_training_session_count", "training_session_count",
        "validation_session_count", "test_session_count", "training_growth_session_count",
        "real_candle_counts", "total_real_candle_count", "label_distributions",
        "training_elapsed_seconds",
    ):
        if field in metadata and checkpoint_experiment.get(field) != metadata[field]:
            raise ValueError(f"Checkpoint experiment does not match run metadata: {field}")

    # A prediction names the maximum-probability class, never a human comparison label.
    # This also catches accidental column swaps in older bundles without checksums.
    regime_names = np.array(["bull", "bear", "range"])
    for head, probability_columns in zip(
        ("current", "anticipated"), probability_groups, strict=True
    ):
        predicted_regimes = regime_names[
            predictions.loc[:, probability_columns].to_numpy(dtype=float).argmax(axis=1)
        ]
        if not predictions[f"predicted_{head}_regime"].eq(predicted_regimes).all():
            raise ValueError("Predicted regimes do not match their probability columns.")

    boundary = metadata["test"]
    if not isinstance(boundary, dict):
        raise ValueError("Run test boundary must be a named mapping.")

    try:
        expected_count = int(boundary["end_index"]) - int(boundary["start_index"])
        start_date = pd.Timestamp(boundary["start_date"]).date()
        end_date = pd.Timestamp(boundary["end_date"]).date()
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Run test boundary is malformed.") from error

    session_dates = predictions["session_date"].drop_duplicates().tolist()
    derived_dates = predictions["timestamp"].dt.tz_convert(config.data.session_timezone).dt.date
    if (
        expected_count <= 0
        or len(session_dates) != expected_count
        or session_dates[0] != start_date
        or session_dates[-1] != end_date
        or not predictions["session_date"].equals(derived_dates)
    ):
        raise ValueError("Prediction sessions do not match the recorded test boundary.")

    # Resume has no OHLC input to join against, so the bundle itself must retain consistent
    # candle identities and all real test rows counted during the official evaluation.
    expected_ids = [
        build_candlestick_id(config.data.instrument, timestamp, config.data.target_interval)
        for timestamp in predictions["timestamp"]
    ]
    if predictions["candlestick_id"].tolist() != expected_ids:
        raise ValueError("Prediction identifiers do not match their timestamps and interval.")

    for head in ("current", "anticipated"):
        if loaded_checkpoint.metadata["test"][head]["support"] != len(predictions):
            raise ValueError("Prediction count does not match checkpoint test support.")

    return metadata, predictions
