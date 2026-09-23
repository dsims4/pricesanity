"""Generalized corpus freezing, leakage, shared evaluation, and command orchestration."""

from dataclasses import replace
from pathlib import Path
import sqlite3

import pandas as pd
import pytest

from pricesanity.annotation.store import AnnotationStore
from pricesanity.benchmark.cli import main
from pricesanity.benchmark.execution import BenchmarkExecutor
from pricesanity.benchmark.orchestration import execute_suite
from pricesanity.benchmark.population import discover_population, freeze_population
from pricesanity.benchmark.protocol import BenchmarkTrack, load_benchmark_config
from pricesanity.benchmark.registry import list_model_families
from pricesanity.benchmark.search_spaces import load_search_spaces
from pricesanity.benchmark.snapshot import load_benchmark_snapshot, load_snapshot_sessions
from pricesanity.benchmark.sufficiency_data import load_development_sessions
from pricesanity.config import load_config
from pricesanity.data.annotation_evidence import EVIDENCE_ATTRIBUTE, EVIDENCE_POLICY
from pricesanity.data.identifiers import build_candlestick_id


@pytest.fixture
def inputs(tmp_path):
    app = load_config("configs/default.yaml")
    app = replace(app, session=replace(app.session, end_time="12:10"))
    config = replace(load_benchmark_config("configs/benchmark/default.yaml"), output_root=tmp_path / "official")
    timestamps = pd.DatetimeIndex([
        time for day in pd.bdate_range("2016-01-04", periods=81)
        for time in pd.date_range(day.strftime("%Y-%m-%d") + " 09:30", periods=32,
                                  freq="5min", tz="America/New_York").tz_convert("UTC")
    ])
    normalized = pd.DataFrame({"ts_event": timestamps, "instrument": "ES.v.0", "open_gap": 0.,
                               "body": 0., "high_from_close": .01, "low_from_close": -.01})
    ohlc = pd.DataFrame({"ts_event": timestamps, "instrument": "ES.v.0",
                         "open": 100., "high": 101., "low": 99., "close": 100.})
    records = []
    for opening in timestamps[::32]:
        previous = opening - pd.Timedelta(days=1)
        records.append({"session_open": opening.isoformat(),
            "session_close": (opening + pd.Timedelta(minutes=160)).isoformat(),
            "condition": "available", "previous_condition": "available",
            "previous_session_open": previous.isoformat(),
            "previous_session_close": (previous + pd.Timedelta(minutes=160)).isoformat(),
            "reference_timestamp": (previous + pd.Timedelta(minutes=155)).isoformat(),
            "reference_close": 100.})
    evidence = {"policy": EVIDENCE_POLICY, "instrument": "ES.v.0", "interval": "0 days 00:05:00",
        "timezone": "America/New_York", "session_start": "09:30", "session_end": "12:10", "sessions": records}
    paths = (tmp_path / "normalized.parquet", tmp_path / "ohlc.parquet")
    for frame, path in zip((normalized, ohlc), paths):
        frame.attrs[EVIDENCE_ATTRIBUTE] = evidence
        frame.to_parquet(path, index=False)
    database = tmp_path / "annotations.db"
    store = AnnotationStore(database)
    store.close()
    # Eighty complete sessions and one partly annotated session.
    with sqlite3.connect(database) as db:
        db.executemany("INSERT INTO annotations (candlestick_id, current_regime, anticipated_regime) VALUES (?, ?, ?)",
            [(build_candlestick_id("ES.v.0", time, "5min"), ("bull", "bear", "range")[i % 3],
              ("bull", "bear", "range")[(i + 1) % 3]) for i, time in enumerate(timestamps[:-1])])
    return dict(normalized_path=paths[0], candlestick_path=paths[1], database_path=database,
                app_config=app, config=config)


def test_discovery_auto_cap_and_incomplete_sessions(inputs):
    auto = discover_population(**inputs)
    assert len(auto.sessions) == 80
    assert auto.config.development_session_count == 72
    assert auto.sources["corpus_selection"] == "auto/all-available"
    capped = discover_population(**inputs, session_count=70)
    assert len(capped.sessions) == 70
    assert capped.sources["corpus_selection"] == "explicitly capped"
    assert capped.sources["development_session_ids"] + capped.sources["test_session_ids"] == (
        auto.sources["development_session_ids"] + auto.sources["test_session_ids"]
    )[:70]
    with pytest.raises(ValueError, match="at most 80"):
        discover_population(**inputs, session_count=81)


def test_frozen_population_is_unchanged_by_new_annotation(inputs, tmp_path):
    population = discover_population(**inputs)
    directory = tmp_path / "study"
    snapshot = freeze_population(population, directory, timestamp_column="ts_event")
    data = pd.read_parquet(inputs["normalized_path"])
    candle = build_candlestick_id("ES.v.0", data.ts_event.iloc[-1], "5min")
    with sqlite3.connect(inputs["database_path"]) as db:
        db.execute("INSERT INTO annotations (candlestick_id, current_regime, anticipated_regime) VALUES (?, 'bull', 'bear')", (candle,))
    newer = discover_population(**inputs)
    assert len(newer.sessions) == 81
    assert newer.identity_sha256 != population.identity_sha256
    assert load_benchmark_snapshot(snapshot.directory).identity_sha256 == snapshot.identity_sha256
    assert len(load_snapshot_sessions(snapshot)) == 72
    with pytest.raises(ValueError, match="Resume population differs"):
        freeze_population(newer, directory, timestamp_column="ts_event", resume=True)
    assert freeze_population(population, directory, timestamp_column="ts_event", resume=True) == snapshot


@pytest.mark.parametrize("field", ["test_session_ids", "protocol_version", "source", "normalized_sha256",
                                  "annotation_rows_sha256", "total_session_count"])
def test_resume_rejects_identity_changes(inputs, tmp_path, field):
    population = discover_population(**inputs)
    freeze_population(population, tmp_path / "study", timestamp_column="ts_event")
    changed = replace(population, sources={**population.sources, field: "changed"})
    with pytest.raises(ValueError, match="Resume population differs"):
        freeze_population(changed, tmp_path / "study", timestamp_column="ts_event", resume=True)


def test_strict_evidence_remains_mandatory(inputs):
    for path in (inputs["normalized_path"], inputs["candlestick_path"]):
        frame = pd.read_parquet(path)
        frame.attrs.clear()
        frame.to_parquet(path, index=False)
    with pytest.raises(ValueError):
        discover_population(**inputs)


def test_data_sufficiency_uses_dynamic_catalog_and_known_snapshot(inputs, tmp_path):
    # Remove the partial session: the separate diagnostic deliberately keeps its existing
    # strict treatment of partly annotated sessions within its allowed development prefix.
    diagnostic_inputs = {key: value for key, value in inputs.items() if key != "config"}
    diagnostic_inputs["benchmark_config"] = inputs["config"]
    sessions, sources = load_development_sessions(**diagnostic_inputs)
    assert len(sessions) == 72  # 81 prepared sessions -> ceil(8.1) test sessions.
    population = discover_population(**inputs, session_count=70)
    snapshot = freeze_population(population, tmp_path / "official" / "small", timestamp_column="ts_event")
    sessions, sources = load_development_sessions(**diagnostic_inputs)
    assert len(sessions) == 63
    assert sources["partition_snapshot_identities"] == [snapshot.identity_sha256]


def test_all_models_seeds_and_tracks_share_test_ids_without_development_leakage(inputs, tmp_path, monkeypatch):
    population = discover_population(**inputs)
    snapshot = freeze_population(population, tmp_path / "study", timestamp_column="ts_event")
    executor = BenchmarkExecutor(snapshot=snapshot, config=population.config,
        search_spaces=load_search_spaces("configs/benchmark/search_spaces.yaml"), study_root=tmp_path / "study")
    expected_development = population.sources["development_session_ids"]
    expected_test = population.sources["test_session_ids"]
    assert [str(session.session_date.iloc[0]) for session in executor.sessions] == expected_development
    models = [family.name for family in list_model_families()]
    assert len(models) == 14
    real_open = Path.open
    real_read = pd.read_parquet
    def guard_open(path, *args, **kwargs):
        assert path.name != "sealed_holdout.parquet"
        return real_open(path, *args, **kwargs)
    def guard_read(path, *args, **kwargs):
        assert Path(path).name != "sealed_holdout.parquet"
        return real_read(path, *args, **kwargs)
    observed = []
    def execute(**kwargs):
        train_ids = [str(session.session_date.iloc[0]) for session in kwargs["training_sessions"]]
        eval_ids = [str(session.session_date.iloc[0]) for session in kwargs["evaluation_sessions"]]
        if kwargs["run_name"].startswith("final_seed_"):
            assert train_ids == expected_development
            assert eval_ids == expected_test
            observed.append((kwargs["model_name"], kwargs["track"], kwargs["seed"]))
        else:
            assert set(train_ids + eval_ids) <= set(expected_development)
            assert max(train_ids) < min(eval_ids)
        return tmp_path / "stub-run"
    monkeypatch.setattr(executor, "_execute_persisted_run", execute)
    # Real fold representations/scalers and real final lifecycle; replace only expensive fits.
    with monkeypatch.context() as guarded:
        guarded.setattr(Path, "open", guard_open)
        guarded.setattr(pd, "read_parquet", guard_read)
        for track in BenchmarkTrack:
            for model in models:
                for fold in executor.plan.chronological_validation_folds:
                    training, evaluation = executor._fold_sessions(fold)
                    assert {str(s.session_date.iloc[0]) for s in training + evaluation} <= set(expected_development)
                    executor._corpora(model, {}, track, training, evaluation)
                selected = executor._selected_record(model, track, {}, (.5,) * 5,
                    fold_indices=tuple(range(5)), all_development_folds=True)
                executor._write_selected(selected)
                executor.run_learning_curve(model, track=track)
        executor.freeze_development()
    # The normal orchestrator resumes past the seal and exercises real run_final for all models.
    execute_suite(executor)
    assert {(model, track) for model, track, _ in observed} == {(model, track) for model in models for track in BenchmarkTrack}
    for family in list_model_families():
        expected_seeds = set(executor.config.final_seeds if family.stochastic else (executor.config.tuning_seed,))
        for track in BenchmarkTrack:
            assert {seed for model, found_track, seed in observed if model == family.name and found_track == track} == expected_seeds


def test_command_dry_run_discovers_without_writing(inputs, tmp_path, monkeypatch, capsys):
    import pricesanity.benchmark.cli as cli
    monkeypatch.setattr(cli, "load_config", lambda path: inputs["app_config"])
    root = tmp_path / "no-artifacts"
    assert main(["run", "--normalized", str(inputs["normalized_path"]),
        "--candlesticks", str(inputs["candlestick_path"]), "--database", str(inputs["database_path"]),
        "--project-config", "unused.yaml", "--artifact-root", str(root), "--dry-run"]) == 0
    assert not root.exists()
    output = capsys.readouterr().out
    assert '"total_session_count": 80' in output
    assert '"test_session_count": 8' in output
    assert "Tracks: controlled, best_of_family" in output


def test_suite_finishes_all_development_before_any_final(tmp_path):
    events = []
    class StubExecutor:
        scope = {"models": [family.name for family in list_model_families()],
                 "tracks": [track.value for track in BenchmarkTrack]}
        paths = type("Paths", (), {"root": tmp_path})()
        def tune_model(self, name, **kwargs):
            events.append(("tune", name, kwargs["track"]))
        def run_learning_curve(self, name, **kwargs):
            events.append(("curve", name, kwargs["track"]))
        def freeze_development(self):
            assert len(events) == 56
            events.append(("seal",))
        def run_final(self, name, **kwargs):
            assert events[56] == ("seal",)
            assert kwargs["confirm_final_holdout"]
            events.append(("final", name, kwargs["track"]))
            return [name]
    assert len(execute_suite(StubExecutor())) == 28


def test_normal_command_runs_and_resumes_frozen_population(inputs, tmp_path, monkeypatch):
    """Exercise real CLI discovery, publication, final artifacts, reporting, and frozen resume."""
    import pricesanity.benchmark.cli as cli
    from pricesanity.benchmark.report import collect_aggregated_runs
    monkeypatch.setattr(cli, "load_config", lambda path: inputs["app_config"])
    root = tmp_path / "benchmark"
    arguments = ["run", "--normalized", str(inputs["normalized_path"]),
        "--candlesticks", str(inputs["candlestick_path"]), "--database", str(inputs["database_path"]),
        "--project-config", "unused.yaml", "--artifact-root", str(root), "--model", "majority_class",
        "--acknowledge-scaling-risk", "--resume"]
    assert main(arguments) == 0
    study = next(root.glob("generalized_80_*"))
    report = collect_aggregated_runs(study)
    assert len(report) == 2
    assert set(report.protocol_version) == {"generalized_chronological_90_10_v1"}
    assert set(report.total_session_count) == {80}
    assert set(report.development_session_count) == {72}
    assert set(report.test_session_count) == {8}
    assert set(report.test_fraction) == {.1}
    assert report.test_session_ids_sha256.nunique() == 1
    assert main(arguments) == 0
    # Explicit study resume consumes frozen bytes even if the annotation source disappears.
    inputs["database_path"].unlink()
    assert main(["run", "--study-directory", str(study), "--resume", "--model", "majority_class"]) == 0


def test_normal_command_defaults_to_full_registry_and_both_tracks(inputs, tmp_path, monkeypatch):
    import pricesanity.benchmark.cli as cli
    monkeypatch.setattr(cli, "load_config", lambda path: inputs["app_config"])
    called = []
    def suite(executor, **kwargs):
        assert executor.scope["models"] == sorted(family.name for family in list_model_families())
        assert set(executor.scope["tracks"]) == {track.value for track in BenchmarkTrack}
        assert executor.snapshot.session_count == 80
        assert len(executor.sessions) == 72
        called.append(executor.snapshot.identity_sha256)
        return []
    monkeypatch.setattr(cli, "execute_suite", suite)
    assert main(["run", "--normalized", str(inputs["normalized_path"]),
        "--candlesticks", str(inputs["candlestick_path"]), "--database", str(inputs["database_path"]),
        "--project-config", "unused.yaml", "--artifact-root", str(tmp_path / "benchmark")]) == 0
    assert len(called) == 1


def test_short_session_training_support_fails_without_shrinking_search(inputs):
    from pricesanity.benchmark.execution import validate_training_support
    population = discover_population(**inputs)
    short_sessions = tuple(session.iloc[:16].copy() for session in population.sessions)
    spaces = load_search_spaces("configs/benchmark/search_spaces.yaml")
    with pytest.raises(ValueError, match="kNN search requires at least 75"):
        validate_training_support(population.config, short_sessions, spaces, ("controlled",))
    assert spaces["knn"]["n_neighbors"]["high"] == 75
