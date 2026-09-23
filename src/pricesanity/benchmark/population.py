"""Discover and freeze one complete, strictly eligible chronological benchmark population."""

from contextlib import closing
from dataclasses import asdict, dataclass
from pathlib import Path
import os
import sqlite3
import tempfile

import pandas as pd

from pricesanity.annotation.schema import CandlestickAnnotation, MarketRegime
from pricesanity.benchmark.artifacts import canonical_sha256, file_sha256, source_identity, _run_lock
from pricesanity.benchmark.protocol import BenchmarkConfig, describe_protocol, resolve_benchmark_config, plan_benchmark
from pricesanity.benchmark.snapshot import (
    BenchmarkSnapshot, freeze_benchmark_snapshot_from_sessions, load_benchmark_snapshot,
)
from pricesanity.config import AppConfig
from pricesanity.data.annotation_evidence import EVIDENCE_POLICY, validate_annotation_evidence
from pricesanity.data.identifiers import build_candlestick_id
from pricesanity.training.dataset import build_complete_annotated_sessions


@dataclass(frozen=True)
class BenchmarkPopulation:
    sessions: tuple[pd.DataFrame, ...]
    config: BenchmarkConfig
    sources: dict

    @property
    def identity_sha256(self) -> str:
        return canonical_sha256(self.sources)


def discover_population(
    *, normalized_path: Path, candlestick_path: Path, database_path: Path,
    app_config: AppConfig, config: BenchmarkConfig, session_count: int | None = None,
) -> BenchmarkPopulation:
    """Read one consistent annotation view; incomplete sessions never become split units."""
    hashes = [file_sha256(path) for path in (normalized_path, candlestick_path)]
    normalized, ohlc = (pd.read_parquet(path) for path in (normalized_path, candlestick_path))
    timestamp = app_config.data.timestamp_column
    if not normalized[timestamp].equals(ohlc[timestamp]):
        raise ValueError("Normalized and OHLC snapshot inputs must have exactly aligned timestamps.")
    validate_annotation_evidence(ohlc, normalized, config=app_config)
    candle_ids = [build_candlestick_id(instrument, time, app_config.data.target_interval)
                  for instrument, time in normalized[["instrument", timestamp]].itertuples(index=False, name=None)]
    rows = []
    with closing(sqlite3.connect(database_path.resolve().as_uri() + "?mode=ro", uri=True)) as db:
        db.execute("BEGIN")
        for start in range(0, len(candle_ids), 400):
            selected = candle_ids[start:start + 400]
            placeholders = ",".join("?" for _ in selected)
            rows.extend(db.execute(
                "SELECT candlestick_id, current_regime, anticipated_regime FROM annotations "
                f"WHERE candlestick_id IN ({placeholders})", selected,
            ).fetchall())
    labeled = {row[0] for row in rows}
    dates = pd.to_datetime(normalized[timestamp], utc=True).dt.tz_convert(
        app_config.data.session_timezone
    ).dt.date
    membership = pd.DataFrame({"day": dates, "complete": [value in labeled for value in candle_ids]})
    complete_dates = membership.groupby("day", sort=True)["complete"].all()
    available = list(complete_dates.index[complete_dates])
    if session_count is not None and (type(session_count) is not int or not 0 < session_count <= len(available)):
        raise ValueError(f"Session cap must be positive and at most {len(available)} complete eligible sessions.")
    selected_dates = available if session_count is None else available[:session_count]
    resolved = resolve_benchmark_config(len(selected_dates), config)
    plan_benchmark(len(selected_dates), resolved)
    retained = dates.isin(selected_dates)
    selected_ids = {value for value, keep in zip(candle_ids, retained) if keep}
    selected_rows = sorted(row for row in rows if row[0] in selected_ids)
    annotations = [CandlestickAnnotation(candle, MarketRegime(current), MarketRegime(anticipated))
                   for candle, current, anticipated in selected_rows]
    sessions = tuple(build_complete_annotated_sessions(
        normalized.loc[retained].reset_index(drop=True), annotations,
        timestamp_column=timestamp, interval=app_config.data.target_interval,
        session_timezone=app_config.data.session_timezone,
    ))
    if any(len(session) <= max(config.controlled_first_scored_candle_position,
                              config.best_of_family_first_scored_candle_position)
           for session in sessions):
        raise ValueError("Every selected session must contain scored candles for both benchmark tracks.")
    if hashes != [file_sha256(path) for path in (normalized_path, candlestick_path)]:
        raise ValueError("Prepared market artifacts changed during population discovery; retry.")
    development = resolved.development_session_count
    sources = {
        **describe_protocol(resolved),
        "development_session_ids": [str(day) for day in selected_dates[:development]],
        "test_session_ids": [str(day) for day in selected_dates[development:]],
        "corpus_selection": "auto/all-available" if session_count is None else "explicitly capped",
        "requested_session_cap": session_count,
        "normalized_path": str(normalized_path.resolve()), "normalized_sha256": hashes[0],
        "candlestick_path": str(candlestick_path.resolve()), "candlestick_sha256": hashes[1],
        "annotation_database_path": str(database_path.resolve()),
        "annotation_rows_sha256": canonical_sha256(selected_rows),
        "ordered_candle_ids_sha256": canonical_sha256([
            value for value, keep in zip(candle_ids, retained) if keep
        ]),
        "project_configuration_sha256": canonical_sha256(asdict(app_config)),
        "protocol_configuration_sha256": canonical_sha256(asdict(resolved)),
        "source": source_identity(), "eligibility_policy": EVIDENCE_POLICY,
        "instrument": app_config.data.instrument,
        "target_interval": app_config.data.target_interval,
        "session_timezone": app_config.data.session_timezone,
    }
    return BenchmarkPopulation(sessions, resolved, sources)


def freeze_population(
    population: BenchmarkPopulation, study_directory: Path, *, timestamp_column: str,
    resume: bool = False,
) -> BenchmarkSnapshot:
    """Publish a complete snapshot atomically and validate exact identity on reuse."""
    study_directory.mkdir(parents=True, exist_ok=True)
    destination = study_directory / "snapshot"
    with _run_lock(study_directory / ".population.lock"):
        if destination.exists():
            if not resume:
                raise FileExistsError("Population already exists; use --resume to reuse its frozen snapshot.")
            snapshot = load_benchmark_snapshot(destination)
            if canonical_sha256(snapshot.source_identities) != population.identity_sha256:
                raise ValueError("Resume population differs in protocol, exact session IDs, source, or data identity.")
            return snapshot
        with tempfile.TemporaryDirectory(prefix=".population-", dir=study_directory) as temporary:
            staged = Path(temporary) / "snapshot"
            freeze_benchmark_snapshot_from_sessions(
                population.sessions, output_directory=staged, timestamp_column=timestamp_column,
                expected_session_count=population.config.expected_session_count,
                development_session_count=population.config.development_session_count,
                source_identities=population.sources,
            )
            os.rename(staged, destination)
            descriptor = os.open(study_directory, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
    return load_benchmark_snapshot(destination)
