from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from pricesanity.benchmark.snapshot import (
    freeze_benchmark_snapshot_from_sessions,
    load_benchmark_snapshot,
    load_snapshot_sessions,
    validate_snapshot_data,
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


@pytest.mark.parametrize("targets", [[0, 1, 2], [0.0, 1.0, 2.0], ["0", "1", "2"]])
def test_snapshot_preserves_integer_target_representations(tmp_path, targets):
    """Numeric strings and integral floats retain the same three-class contract."""

    session = _sessions(count=1, length=3)[0]
    for head in ("current", "anticipated"):
        session[f"{head}_target"] = targets
    snapshot = freeze_benchmark_snapshot_from_sessions(
        [session], output_directory=tmp_path / "snapshot"
    )
    loaded = load_snapshot_sessions(snapshot)[0]
    for head in ("current", "anticipated"):
        np.testing.assert_array_equal(loaded[f"{head}_target"], [0, 1, 2])
        assert loaded[f"{head}_target"].dtype == np.int64


@pytest.mark.parametrize("head", ["current", "anticipated"])
@pytest.mark.parametrize("invalid", [0.5, 1.5, 2.5, -1, 3, "1.5", np.nan, np.inf])
def test_snapshot_rejects_invalid_targets_before_coercion(tmp_path, head, invalid):
    """Both direct validation and publication reject original invalid target values."""

    session = _sessions(count=1, length=3)[0]
    session[f"{head}_target"] = [invalid, invalid, invalid]
    data = session.rename(columns={"ts_event": "timestamp"}).assign(
        session_index=0, candle_position=np.arange(3)
    )
    with pytest.raises(ValueError, match="unknown regime target"):
        validate_snapshot_data(data)

    # The publisher must not truncate labels while flattening sessions before validation.
    destination = tmp_path / "snapshot"
    with pytest.raises(ValueError, match="unknown regime target"):
        freeze_benchmark_snapshot_from_sessions([session], output_directory=destination)
    assert not destination.exists()
