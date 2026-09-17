from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from pricesanity.benchmark.snapshot import (
    freeze_benchmark_snapshot_from_sessions,
    load_benchmark_snapshot,
    load_snapshot_sessions,
)
from pricesanity.features import FEATURE_COLUMNS


def _sessions(count: int = 3, length: int = 5) -> list[pd.DataFrame]:
    """Build distinct synthetic rows for snapshot identity and mutation checks."""

    sessions = []
    for session_index in range(count):
        session_date = date(2026, 1, 5) + timedelta(days=session_index)
        values = np.arange(length, dtype=np.float32) + session_index
        sessions.append(pd.DataFrame({
            "session_date": [session_date] * length,
            "candlestick_id": [f"{session_index}-{position}" for position in range(length)],
            "ts_event": pd.date_range(
                pd.Timestamp(session_date, tz="UTC") + pd.Timedelta(hours=14, minutes=30),
                periods=length,
                freq="5min",
            ),
            "open_gap": values,
            "body": values + 1,
            "high_from_close": values + 2,
            "low_from_close": -values,
            "current_target": np.arange(length) % 3,
            "anticipated_target": (np.arange(length) + 1) % 3,
        }))
    return sessions


def test_frozen_snapshot_is_immutable_after_live_sources_change(tmp_path) -> None:
    sessions = _sessions()
    snapshot = freeze_benchmark_snapshot_from_sessions(
        sessions,
        output_directory=tmp_path / "snapshot",
        expected_session_count=3,
        source_identities={"annotations": "version-one"},
    )
    original_identity = snapshot.identity_sha256
    original_first_value = load_snapshot_sessions(snapshot)[0][FEATURE_COLUMNS[0]].iloc[0]

    # Mutating the object that represented the live source cannot change the atomically
    # published snapshot used by later trials.
    sessions[0].loc[0, FEATURE_COLUMNS[0]] = 9999.0
    reloaded = load_benchmark_snapshot(snapshot.directory)
    assert reloaded.identity_sha256 == original_identity
    assert load_snapshot_sessions(reloaded)[0][FEATURE_COLUMNS[0]].iloc[0] == original_first_value
    with pytest.raises(FileExistsError, match="immutable"):
        freeze_benchmark_snapshot_from_sessions(
            sessions, output_directory=snapshot.directory
        )
