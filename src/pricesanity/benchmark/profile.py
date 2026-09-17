"""Reproducibly measure benchmark infrastructure without fitting study models."""

from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from typing import Any
import json

import numpy as np
import pandas as pd

from pricesanity.benchmark.artifacts import load_benchmark_run, load_benchmark_summary
from pricesanity.benchmark.snapshot import (
    freeze_benchmark_snapshot_from_sessions,
    load_benchmark_snapshot,
    load_snapshot_sessions,
)
from pricesanity.features import (
    EvaluationUniverse,
    FEATURE_COLUMNS,
    RepresentationSpec,
    build_representation_corpus,
    fit_unique_candle_standardizer,
    tabular_representation,
)


def profile_synthetic_infrastructure(
    *,
    session_count: int = 2690,
    candles_per_session: int = 81,
    context_length: int = 16,
    artifact_root: str | Path | None = None,
) -> dict[str, Any]:
    """Measure the reusable data path on deterministic synthetic sessions.

    The report deliberately excludes model fitting. Its timings characterize this machine and
    invocation only; they are evidence for optimization choices, not universal benchmarks.
    """

    if min(session_count, candles_per_session, context_length) <= 0:
        raise ValueError("Synthetic profile dimensions must be positive.")
    if candles_per_session < context_length:
        raise ValueError("Synthetic sessions must be at least as long as the context.")

    # A fixed synthetic corpus makes repeated infrastructure measurements comparable without
    # reading the private annotated dataset or accidentally timing model fitting.
    sessions = _synthetic_sessions(session_count, candles_per_session)
    measurements: dict[str, float | int | None] = {}

    with TemporaryDirectory(prefix="pricesanity-profile-") as temporary_directory:
        snapshot_directory = Path(temporary_directory) / "snapshot"
        freeze_benchmark_snapshot_from_sessions(
            sessions,
            output_directory=snapshot_directory,
            expected_session_count=session_count,
            source_identities={"kind": "deterministic_synthetic_profile"},
        )
        started = perf_counter()
        snapshot = load_benchmark_snapshot(snapshot_directory)
        loaded_sessions = load_snapshot_sessions(snapshot)
        measurements["snapshot_load_seconds"] = perf_counter() - started

        started = perf_counter()
        standardizer = fit_unique_candle_standardizer(
            loaded_sessions, feature_columns=FEATURE_COLUMNS
        )
        measurements["unique_candle_standardization_fit_seconds"] = (
            perf_counter() - started
        )
        # Building windows is timed from immutable snapshot rows. This is the operation a
        # future cache would need to beat, so it is measured before adding cache complexity.
        specification = RepresentationSpec(
            name="profile_sequential",
            window_length=context_length,
            layout="sequential",
            standardization="training_only",
            polynomial_degree=None,
            padding_policy="none",
            feature_columns=FEATURE_COLUMNS,
        )
        started = perf_counter()
        corpus = build_representation_corpus(
            loaded_sessions,
            specification,
            EvaluationUniverse(first_scored_candle_position=context_length - 1),
        )
        measurements["causal_window_construction_seconds"] = perf_counter() - started

        started = perf_counter()
        tabular = tabular_representation(corpus)
        measurements["tabular_representation_seconds"] = perf_counter() - started
        started = perf_counter()
        selected = corpus.select_sessions(range(min(2000, session_count)))
        measurements["fold_session_selection_seconds"] = perf_counter() - started

        measurements.update({
            "session_count": session_count,
            "candles_per_session": candles_per_session,
            "context_length": context_length,
            "sample_count": len(corpus.current_targets),
            "sequence_feature_bytes": int(corpus.features.nbytes),
            "validity_mask_bytes": int(corpus.valid_history_mask.nbytes),
            "tabular_feature_bytes": int(tabular.nbytes),
            "selected_sample_count": len(selected.current_targets),
            "standardizer_feature_count": int(standardizer.mean.size),
        })

    measurements.update(_profile_saved_artifacts(artifact_root))
    return {
        "format_version": 1,
        "scope": "benchmark infrastructure only; no model training",
        "timing_warning": "Machine-specific measurements; do not treat as universal.",
        "measurements": measurements,
    }


def write_profile_report(report: dict[str, Any], path: str | Path) -> None:
    """Write one local machine-readable profile without changing scientific artifacts."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _profile_saved_artifacts(artifact_root: str | Path | None) -> dict[str, float | None]:
    """Contrast summary-only scanning with one verified full artifact load."""

    if artifact_root is None:
        return {"summary_artifact_scan_seconds": None, "full_artifact_load_seconds": None}

    # Metadata files identify only published runs, excluding partial directories from both
    # measurements so resume debris cannot distort the summary-versus-full-load comparison.
    metadata_paths = sorted(Path(artifact_root).glob("*/*/*/benchmark_metadata.json"))
    if not metadata_paths:
        return {"summary_artifact_scan_seconds": None, "full_artifact_load_seconds": None}

    started = perf_counter()
    for metadata_path in metadata_paths:
        load_benchmark_summary(metadata_path.parent)
    summary_seconds = perf_counter() - started
    started = perf_counter()
    load_benchmark_run(metadata_paths[0].parent)
    return {
        "summary_artifact_scan_seconds": summary_seconds,
        "full_artifact_load_seconds": perf_counter() - started,
    }


def _synthetic_sessions(
    session_count: int,
    candles_per_session: int,
) -> tuple[pd.DataFrame, ...]:
    """Create deterministic session-shaped rows without reading private market data."""

    # Keep the seed local to the profile so its synthetic evidence is reproducible without
    # mutating the random state used by any benchmark model in the same process.
    generator = np.random.default_rng(8675309)
    sessions = []
    first_date = pd.Timestamp("2010-01-04")
    for session_index in range(session_count):
        session_date = (first_date + pd.offsets.BDay(session_index)).date()
        timestamp = pd.Timestamp(session_date, tz="UTC") + pd.Timedelta(
            hours=14,
            minutes=30,
        )
        features = generator.normal(
            size=(candles_per_session, len(FEATURE_COLUMNS))
        ).astype(np.float32)
        classes = np.arange(candles_per_session, dtype=np.int64) % 3
        sessions.append(
            pd.DataFrame(
                {
                    "session_date": [session_date] * candles_per_session,
                    "candlestick_id": [
                        f"profile-{session_index:04d}-{position:02d}"
                        for position in range(candles_per_session)
                    ],
                    "ts_event": pd.date_range(
                        timestamp,
                        periods=candles_per_session,
                        freq="5min",
                    ),
                    **{
                        column: features[:, column_index]
                        for column_index, column in enumerate(FEATURE_COLUMNS)
                    },
                    "current_target": classes,
                    "anticipated_target": np.roll(classes, -1),
                }
            )
        )
    return tuple(sessions)
