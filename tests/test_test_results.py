"""Regression checks for saved test evidence and the read-only results window."""

import json
import os
from pathlib import Path
import shutil
import sqlite3

import pandas as pd
import pytest
import torch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from pricesanity.annotation.schema import CandlestickAnnotation, MarketRegime
from pricesanity.annotation.store import AnnotationStore
from pricesanity.config import load_config
from pricesanity.data.identifiers import build_candlestick_id
from pricesanity.gui.test_results import (
    TestResultsWindow as ResultsWindow,
    find_regime_change_markers,
    load_test_run,
)
from pricesanity.training.artifacts import file_sha256
from pricesanity.training.training_cli import main


@pytest.fixture(scope="module")
def completed_bundle(tmp_path_factory):
    """Train a tiny real bundle once; each mutation test receives its own copy."""

    directory = tmp_path_factory.mktemp("completed-test-run")
    # The shorter final session exercises real-length artifact alignment and GUI navigation;
    # these minimal synthetic days do not assert production schedule eligibility.
    timestamps = pd.DatetimeIndex([
        timestamp
        for session_date, length in zip(
            ("2016-01-04", "2016-01-05", "2016-01-06", "2016-01-07"),
            (3, 3, 3, 2),
            strict=True,
        )
        for timestamp in pd.date_range(f"{session_date}T14:30:00Z", periods=length, freq="5min")
    ])
    normalized = pd.DataFrame({
        "ts_event": timestamps,
        "instrument": "ES.v.0",
        "open_gap": 0.01,
        "body": 0.02,
        "high_from_close": 0.01,
        "low_from_close": -0.01,
    })
    normalized.to_parquet(directory / "normalized.parquet", index=False)
    candlesticks = normalized[["ts_event", "instrument"]].copy()
    for column, value in (("open", 100), ("high", 102), ("low", 99), ("close", 101)):
        candlesticks[column] = value
    candlesticks.to_parquet(directory / "ohlc.parquet", index=False)

    store = AnnotationStore(directory / "annotations.db")
    try:
        for timestamp in timestamps:
            store.save(CandlestickAnnotation(
                build_candlestick_id("ES.v.0", timestamp, "5min"),
                MarketRegime.BEAR,
                MarketRegime.RANGE,
            ))
    finally:
        store.close()

    arguments = [
        "--normalized", str(directory / "normalized.parquet"),
        "--database", str(directory / "annotations.db"),
        "--candlesticks", str(directory / "ohlc.parquet"),
        "--config", "configs/default.yaml",
        "--checkpoint", str(directory / "run" / "model.pt"),
        "--training-sessions", "1", "--validation-sessions", "1",
        "--test-sessions", "2", "--epochs", "1", "--device", "cpu",
    ]
    assert main(arguments) == 0

    return directory, arguments


@pytest.fixture
def bundle(completed_bundle, tmp_path):
    """Keep deliberate artifact corruption isolated from other checks."""

    shutil.copytree(completed_bundle[0], tmp_path / "bundle")
    return tmp_path / "bundle"


def rewrite_predictions(bundle: Path, predictions: pd.DataFrame) -> None:
    """Maintain integrity hashes while testing deeper semantic validation separately."""

    run_directory = bundle / "run"
    predictions.to_parquet(run_directory / "test_predictions.parquet", index=False)
    metadata = json.loads((run_directory / "run_metadata.json").read_text())
    metadata["test_predictions_sha256"] = file_sha256(run_directory / "test_predictions.parquet")
    checkpoint = torch.load(run_directory / "model.pt", weights_only=True)
    checkpoint["experiment"]["test_predictions_sha256"] = metadata["test_predictions_sha256"]
    torch.save(checkpoint, run_directory / "model.pt")
    metadata["checkpoint_sha256"] = file_sha256(run_directory / "model.pt")
    (run_directory / "run_metadata.json").write_text(json.dumps(metadata))


def test_marker_positions_name_the_new_regime() -> None:
    """Only a change at t creates a marker at t, labeled with that new prediction."""

    assert find_regime_change_markers(["bull", "bull", "bear", "range", "range"]) == (
        (2, "bear"), (3, "range")
    )
    assert find_regime_change_markers(["bull"]) == ()
    assert find_regime_change_markers(["bear", "bear"]) == ()
    with pytest.raises(ValueError):
        find_regime_change_markers([])


def test_portable_bundle_loads_without_original_internal_paths(bundle) -> None:
    """Both relative metadata and older absolute paths resolve inside the selected bundle."""

    metadata_path = bundle / "run/run_metadata.json"
    metadata = json.loads(metadata_path.read_text())
    assert metadata["checkpoint_path"] == "model.pt"
    assert metadata["prediction_path"] == "test_predictions.parquet"

    # Simulate version-one path spelling without allowing fallback to the original directory.
    metadata["checkpoint_path"] = "/missing/old/run/model.pt"
    metadata["prediction_path"] = "/missing/old/run/test_predictions.parquet"
    metadata_path.write_text(json.dumps(metadata))
    loaded = load_test_run(
        bundle / "run", bundle / "ohlc.parquet",
        config=load_config("configs/default.yaml"),
    )

    assert len(loaded.candlesticks) == 5


@pytest.mark.parametrize("artifact", ["test_predictions.parquet", "model.pt"])
def test_artifact_checksum_rejects_changed_bytes(bundle, artifact) -> None:
    """Even parseable file changes must fail before the GUI can display them."""

    with (bundle / "run" / artifact).open("ab") as artifact_file:
        artifact_file.write(b"changed")

    with pytest.raises(ValueError, match="checksum mismatch"):
        load_test_run(
            bundle / "run", bundle / "ohlc.parquet",
            config=load_config("configs/default.yaml"),
        )


@pytest.mark.parametrize("damage", [
    "missing", "extra", "duplicate", "reordered", "one_minute", "nonfinite",
    "geometry", "instrument", "identifier",
])
def test_ohlc_must_match_whole_test_sessions(bundle, damage) -> None:
    """Reject overlapping subsets, bad market identity, and unsafe candle geometry."""

    path = bundle / "ohlc.parquet"
    candlesticks = pd.read_parquet(path)
    if damage == "missing":
        candlesticks = candlesticks.drop(index=7)
    elif damage in ("extra", "duplicate", "one_minute"):
        extra = candlesticks.iloc[[7]].copy()
        if damage != "duplicate":
            extra["ts_event"] += pd.Timedelta(minutes=1 if damage == "one_minute" else 5)
        candlesticks = pd.concat([candlesticks, extra]).sort_values("ts_event")
    elif damage == "reordered":
        candlesticks = candlesticks.iloc[::-1]
    elif damage == "nonfinite":
        candlesticks["open"] = float("nan")
    elif damage == "geometry":
        candlesticks["low"] = 103
    elif damage == "instrument":
        candlesticks["instrument"] = "NQ.v.0"
    else:
        candlesticks["candlestick_id"] = "wrong"
    candlesticks.to_parquet(path, index=False)

    with pytest.raises(ValueError):
        load_test_run(bundle / "run", path, config=load_config("configs/default.yaml"))


@pytest.mark.parametrize("damage", [
    "missing", "reordered", "duplicate", "bad_probability", "swapped_label", "wrong_run",
    "wrong_model", "wrong_date", "wrong_id", "missing_column",
])
def test_prediction_semantics_fail_before_display(bundle, damage) -> None:
    """Checksums complement, rather than replace, structural and alignment validation."""

    predictions = pd.read_parquet(bundle / "run/test_predictions.parquet")
    if damage == "missing":
        predictions = predictions.drop(index=1)
    elif damage == "reordered":
        predictions = predictions.iloc[::-1]
    elif damage == "duplicate":
        predictions = pd.concat([predictions, predictions.iloc[[-1]]])
    elif damage == "bad_probability":
        predictions["current_probability_bull"] = 5
    elif damage == "swapped_label":
        original = predictions.loc[0, "predicted_current_regime"]
        predictions.loc[0, "predicted_current_regime"] = "bear" if original == "bull" else "bull"
    elif damage == "wrong_run":
        predictions["run_index"] = 99
    elif damage == "wrong_model":
        predictions["model_state_sha256"] = "wrong"
    elif damage == "wrong_date":
        predictions["session_date"] = "2020-01-01"
    elif damage == "wrong_id":
        predictions.loc[0, "candlestick_id"] = "wrong"
    else:
        predictions = predictions.drop(columns="human_current_regime")
    rewrite_predictions(bundle, predictions)

    with pytest.raises(ValueError):
        load_test_run(
            bundle / "run", bundle / "ohlc.parquet",
            config=load_config("configs/default.yaml"),
        )


def test_results_navigation_labels_timezone_and_read_only(bundle, monkeypatch) -> None:
    """Navigate saved evidence without SQL access, inference, or cross-session marker carryover."""

    predictions = pd.read_parquet(bundle / "run/test_predictions.parquet")
    predictions["predicted_current_regime"] = ["bull", "bull", "range", "bear", "bear"]
    for regime in ("bull", "bear", "range"):
        predictions[f"current_probability_{regime}"] = (
            predictions["predicted_current_regime"].eq(regime).astype(float)
        )
    rewrite_predictions(bundle, predictions)
    test_run = load_test_run(
        bundle / "run", bundle / "ohlc.parquet",
        config=load_config("configs/default.yaml"),
    )

    # The window receives its validated timezone explicitly through run metadata.
    test_run.metadata["session_timezone"] = "America/Chicago"
    application = QApplication.instance() or QApplication([])
    database_checksum = file_sha256(bundle / "annotations.db")

    def forbidden_database(*args, **kwargs):
        pytest.fail("The test-results GUI opened a database")

    monkeypatch.setattr(sqlite3, "connect", forbidden_database)

    def forbidden_navigation_work(*args, **kwargs):
        pytest.fail("Navigation reread predictions or invoked model inference")

    from pricesanity.training.model import RegimeTransformer

    monkeypatch.setattr(pd, "read_parquet", forbidden_navigation_work)
    monkeypatch.setattr(RegimeTransformer, "forward", forbidden_navigation_work)
    window = ResultsWindow(test_run)
    try:
        assert window.model_current_label.text() == "Model current: Bull"
        assert window.chart.regime_change_markers == ((2, "range"),)
        assert "08:30 CST" in window.candle_information.text()
        assert window.starting_regime_label.text() == "Starting model regime: Bull"
        assert window.chart.regime_tracks == (
            ("MODEL · CURRENT", ("bull", "bull", "range")),
        )
        window.show_human_annotations.setChecked(True)
        assert window.human_current_label.text() == "Human current: Bear"
        assert window.chart.regime_tracks == (
            ("MODEL · CURRENT", ("bull", "bull", "range")),
            ("HUMAN · CURRENT", ("bear", "bear", "bear")),
            ("HUMAN · ANTICIPATED", ("range", "range", "range")),
        )

        window.move_candle(1)
        window.move_candle(1)
        assert window.model_current_label.text() == "Model current: Range"
        window.next_session_button.click()
        assert window.model_current_label.text() == "Model current: Bear"
        assert len(window.candlestick_data) == 2
        assert window.chart.regime_change_markers == ()
        assert window.active_candlestick_position == 0
        window.previous_session_button.click()
        assert window.model_current_label.text() == "Model current: Bull"

        # Backward boundary navigation lands on the previous day's last actual candle.
        window.move_session(1)
        window.move_candle(-1)
        assert window.model_current_label.text() == "Model current: Range"
    finally:
        window.close()
        application.processEvents()

    assert file_sha256(bundle / "annotations.db") == database_checksum


def test_human_timeline_preserves_missing_annotation_positions(bundle) -> None:
    """Missing human labels remain neutral at their original candle indices."""

    test_run = load_test_run(
        bundle / "run",
        bundle / "ohlc.parquet",
        config=load_config("configs/default.yaml"),
    )
    test_run.candlesticks.loc[1, "human_current_regime"] = None
    test_run.candlesticks.loc[2, "human_anticipated_regime"] = pd.NA
    application = QApplication.instance() or QApplication([])
    window = ResultsWindow(test_run)
    try:
        window.show_human_annotations.setChecked(True)

        assert window.chart.regime_tracks[1] == (
            "HUMAN · CURRENT",
            ("bear", None, "bear"),
        )
        assert window.chart.regime_tracks[2] == (
            "HUMAN · ANTICIPATED",
            ("range", "range", None),
        )
        neutral_spans = [
            patch
            for patch in window.chart.timeline_axes.patches
            if patch.get_hatch() == "///"
        ]
        assert sorted(patch.get_x() for patch in neutral_spans) == [0.5, 1.5]

        window.move_candle(1)
        assert window.human_current_label.text() == "Human current: Missing"
    finally:
        window.close()
        application.processEvents()


def test_resume_skips_only_valid_completed_runs(bundle, capsys) -> None:
    """A completion marker alone cannot authorize reuse or silent replacement."""

    from pricesanity.training.training_cli import RunOutputPaths, _run_needs_training

    directory = bundle / "run"
    paths = RunOutputPaths(
        directory, directory / "model.pt", directory / "test_predictions.parquet",
        directory / "run_metadata.json",
    )
    metadata = json.loads(paths.metadata.read_text())
    arguments = {
        "resume": True, "overwrite": False,
        "config": load_config("configs/default.yaml"),
        "experiment_signature": metadata["experiment_signature"],
    }
    assert not _run_needs_training(paths, **arguments)
    assert "Skipping verified completed run" in capsys.readouterr().out

    arguments["experiment_signature"] = "different annotations or settings"
    with pytest.raises(ValueError, match="Cannot resume"):
        _run_needs_training(paths, **arguments)

    # A corrupt completed bundle is protected, not mistaken for an interrupted run.
    paths.predictions.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum mismatch"):
        _run_needs_training(paths, **arguments)

    paths.metadata.unlink()
    assert _run_needs_training(paths, **arguments)
    assert "Rebuilding incomplete run from epoch 1" in capsys.readouterr().out

    arguments["resume"] = False
    with pytest.raises(FileExistsError):
        _run_needs_training(paths, **arguments)
    arguments["overwrite"] = True
    assert _run_needs_training(paths, **arguments)


def test_walk_forward_selection_resume_and_interruption(bundle, completed_bundle, capsys) -> None:
    """Selected runs retain original numbering and interrupted runs can be rebuilt explicitly."""

    normalized_path = bundle / "normalized.parquet"
    normalized = pd.read_parquet(normalized_path)
    extra_sessions = []
    for day_offset in (1, 2):
        extra_session = normalized.iloc[-2:].copy()
        extra_session["ts_event"] += pd.Timedelta(days=day_offset)
        extra_sessions.append(extra_session)
    extra = pd.concat(extra_sessions, ignore_index=True)
    normalized = pd.concat([normalized, extra], ignore_index=True)
    normalized.to_parquet(normalized_path, index=False)
    store = AnnotationStore(bundle / "annotations.db")
    try:
        for timestamp in extra["ts_event"]:
            store.save(CandlestickAnnotation(
                build_candlestick_id("ES.v.0", timestamp, "5min"),
                MarketRegime.BEAR, MarketRegime.RANGE,
            ))
    finally:
        store.close()

    # The tiny fixture starts at 1/1/2 and grows training by two sessions per run.
    arguments = [argument.replace(str(completed_bundle[0]), str(bundle))
                 for argument in completed_bundle[1]]
    arguments += ["--walk-forward",
                  "--output-directory", str(bundle / "walk")]
    assert main(arguments + ["--run-index", "2"]) == 0
    assert not (bundle / "walk/run_001").exists()
    completed_path = bundle / "walk/run_002/model.pt"
    checksum_before = file_sha256(completed_path)

    assert main(arguments + ["--start-run", "2", "--resume"]) == 0
    assert file_sha256(completed_path) == checksum_before
    assert "Skipping verified completed run" in capsys.readouterr().out

    # Copying just one published artifact represents a crash before final metadata exists.
    (bundle / "walk/run_001").mkdir()
    shutil.copy2(completed_path, bundle / "walk/run_001/model.pt")
    assert main(arguments + ["--resume"]) == 0
    assert "Rebuilding incomplete run" in capsys.readouterr().out
    assert (bundle / "walk/run_001/run_metadata.json").is_file()
    assert file_sha256(completed_path) == checksum_before

    # Changed hyperparameters must not silently reuse completed results.
    with pytest.raises(SystemExit):
        main(arguments + ["--resume", "--learning-rate", "0.01"])
    with pytest.raises(SystemExit):
        main(arguments + ["--run-index", "0"])


def test_older_bundle_has_explicit_integrity_warning(bundle) -> None:
    """Legacy artifacts remain inspectable without pretending an absent checksum was checked."""

    path = bundle / "run/run_metadata.json"
    metadata = json.loads(path.read_text())
    metadata["format_version"] = 1
    metadata.pop("test_predictions_sha256")
    metadata.pop("checkpoint_sha256")
    path.write_text(json.dumps(metadata))

    with pytest.warns(UserWarning, match="Older run has no"):
        load_test_run(bundle / "run", bundle / "ohlc.parquet",
                      config=load_config("configs/default.yaml"))


def test_expansion_preserves_completed_history_and_refits_only_past_data(
    bundle, completed_bundle
) -> None:
    """Growing the corpus must not rewrite old evidence or reuse its fitted standardizer."""

    normalized_path = bundle / "normalized.parquet"
    normalized = pd.read_parquet(normalized_path)

    # Distinct feature values make later training statistics observably different. This is
    # a temporal-boundary test, not a claim about model quality on synthetic prices.
    normalized["body"] = list(range(len(normalized)))
    normalized.to_parquet(normalized_path, index=False)
    arguments = [argument.replace(str(completed_bundle[0]), str(bundle))
                 for argument in completed_bundle[1]]
    arguments += ["--walk-forward", "--resume", "--output-directory", str(bundle / "history")]
    assert main(arguments) == 0
    first_directory = bundle / "history/run_001"
    original_artifacts = {
        path.name: (file_sha256(path), path.stat().st_mtime_ns)
        for path in first_directory.iterdir()
    }
    original_metadata = json.loads((first_directory / "run_metadata.json").read_text())
    assert original_metadata["run_count"] == 1

    # Newly elapsed sessions enable run two. Its first three training days include one
    # original test day; neither active validation nor active test contributes to scaling.
    extra_sessions = []
    extra_prices = []
    prices = pd.read_parquet(bundle / "ohlc.parquet")
    for day_offset in (1, 2):
        extra = normalized.iloc[-2:].copy()
        extra["ts_event"] += pd.Timedelta(days=day_offset)
        extra["body"] = 10000 * day_offset
        extra_sessions.append(extra)
        extra_price = prices.iloc[-2:].copy()
        extra_price["ts_event"] += pd.Timedelta(days=day_offset)
        extra_prices.append(extra_price)
    expanded = pd.concat([normalized, *extra_sessions], ignore_index=True)
    expanded.to_parquet(normalized_path, index=False)
    pd.concat([prices, *extra_prices], ignore_index=True).to_parquet(
        bundle / "ohlc.parquet", index=False
    )
    store = AnnotationStore(bundle / "annotations.db")
    try:
        for extra in extra_sessions:
            for timestamp in extra["ts_event"]:
                store.save(CandlestickAnnotation(
                    build_candlestick_id("ES.v.0", timestamp, "5min"),
                    MarketRegime.BULL, MarketRegime.BEAR,
                ))
    finally:
        store.close()

    assert main(arguments) == 0
    assert original_artifacts == {
        path.name: (file_sha256(path), path.stat().st_mtime_ns)
        for path in first_directory.iterdir()
    }

    from pricesanity.training.dataset import FEATURE_COLUMNS

    for run_index, training_count in ((1, 1), (2, 3)):
        directory = bundle / "history" / f"run_{run_index:03d}"
        metadata = json.loads((directory / "run_metadata.json").read_text())
        checkpoint = torch.load(directory / "model.pt", weights_only=True)
        assert metadata["protocol"] == "expanding_history_v1"
        assert metadata["initialization_mode"] == "fresh"
        assert metadata["training_session_count"] == training_count
        assert metadata["validation_session_count"] == 1
        assert metadata["test_session_count"] == 2
        assert metadata["training_growth_session_count"] == 2
        assert metadata["training_elapsed_seconds"] >= 0
        assert sum(metadata["real_candle_counts"].values()) == metadata["total_real_candle_count"]

        # The larger training pool gets new statistics; future outliers cannot influence them.
        training_rows = expanded["ts_event"].dt.date <= pd.Timestamp(
            metadata["training"]["end_date"]
        ).date()
        expected_features = torch.tensor(
            expanded.loc[training_rows, list(FEATURE_COLUMNS)].to_numpy(), dtype=torch.float32
        )
        torch.testing.assert_close(checkpoint["feature_mean"], expected_features.mean(dim=0))
        for role in ("training", "validation", "test"):
            for head in ("current", "anticipated"):
                distribution = metadata["label_distributions"][role][head]
                assert set(distribution) == {"bull", "bear", "range"}
                assert sum(distribution.values()) == metadata["real_candle_counts"][role]

        # Both bundles still pass checkpoint/prediction checksums and exact chart alignment.
        loaded = load_test_run(
            directory, bundle / "ohlc.parquet", config=load_config("configs/default.yaml")
        )
        assert loaded.candlesticks["run_index"].eq(run_index).all()
        assert "majority_current" in checkpoint
        assert "confusion_matrix" in checkpoint["test"]["current"]
