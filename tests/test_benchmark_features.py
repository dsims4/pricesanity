from datetime import date

import numpy as np
import pandas as pd
import pytest

from pricesanity.features import (
    ArrayStandardizer,
    FEATURE_COLUMNS,
    build_causal_windows,
    fit_unique_candle_standardizer,
    sequential_representation,
    tabular_representation,
)


def _session(session_index: int, length: int = 18) -> pd.DataFrame:
    """Build one traceable session whose values reveal any boundary crossing."""

    session_date = date(2026, 1, session_index + 2)
    # Separate sessions by much more than the feature offsets so a borrowed neighboring candle
    # is distinguishable from both a reordered time step and a swapped feature column.
    base = session_index * 1000
    return pd.DataFrame({
        "session_date": [session_date] * length,
        "candlestick_id": [f"{session_index}-{position}" for position in range(length)],
        "ts_event": pd.date_range(
            f"2026-01-{session_index + 2:02d} 14:30:00",
            periods=length,
            freq="5min",
            tz="UTC",
        ),
        "open_gap": np.arange(length) + base,
        "body": np.arange(length) + base + 100,
        "high_from_close": np.arange(length) + base + 200,
        "low_from_close": np.arange(length) + base + 300,
        "current_target": np.arange(length) % 3,
        "anticipated_target": (np.arange(length) + 1) % 3,
    })


def test_causal_windows_never_use_future_or_neighboring_sessions() -> None:
    """Every target receives exactly its preceding same-session candles."""

    corpus = build_causal_windows(
        [_session(0), _session(1)],
        feature_columns=FEATURE_COLUMNS,
        window_length=16,
    )

    # Three eligible target candles per session: positions 15, 16, and 17.
    assert corpus.features.shape == (6, 16, 4)
    assert corpus.session_indices.tolist() == [0, 0, 0, 1, 1, 1]
    assert corpus.candle_positions.tolist() == [15, 16, 17, 15, 16, 17]
    assert corpus.features[0, :, 0].tolist() == list(range(16))
    assert corpus.features[1, :, 0].tolist() == list(range(1, 17))
    assert corpus.features[3, :, 0].tolist() == list(range(1000, 1016))
    assert corpus.candlestick_ids == (
        "0-15", "0-16", "0-17", "1-15", "1-16", "1-17"
    )
    assert corpus.current_targets[0] == _session(0)["current_target"].iloc[15]
    assert corpus.previous_current_targets[0] == _session(0)["current_target"].iloc[14]


def test_controlled_tabular_and_sequence_inputs_are_identical_values() -> None:
    """Flattening changes shape only, not controlled information or ordering."""

    corpus = build_causal_windows(
        [_session(0)], feature_columns=FEATURE_COLUMNS, window_length=16
    )
    sequential = sequential_representation(corpus)
    tabular = tabular_representation(corpus)

    assert tabular.shape == (3, 64)
    np.testing.assert_array_equal(tabular, sequential.reshape(3, 64))


def test_array_standardizer_uses_only_explicit_training_values() -> None:
    """Future validation values cannot influence frozen preprocessing statistics."""

    training = np.array([[0.0, 10.0], [2.0, 14.0]], dtype=np.float32)
    validation = np.array([[1000.0, 2000.0]], dtype=np.float32)
    standardizer = ArrayStandardizer.fit(training)

    np.testing.assert_allclose(standardizer.mean, [1.0, 12.0])
    np.testing.assert_allclose(standardizer.transform(training).mean(axis=0), [0.0, 0.0])
    assert standardizer.transform(validation)[0, 0] == pytest.approx(999.0)


def test_controlled_standardizer_counts_unique_candles_not_overlapping_windows() -> None:
    """An interior candle gets one vote even though it appears in many causal windows."""

    session = _session(0, length=18)
    # An interior outlier appears in more windows than edge candles, exposing duplicated weight
    # if preprocessing accidentally fits the window tensor instead of unique training candles.
    session.loc[8, "open_gap"] = 10_000.0
    standardizer = fit_unique_candle_standardizer(
        (session,), feature_columns=FEATURE_COLUMNS
    )
    raw = session.loc[:, FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    np.testing.assert_allclose(
        standardizer.mean, raw.mean(axis=0, dtype=np.float64), rtol=0, atol=1e-10
    )

    windows = build_causal_windows(
        [session], feature_columns=FEATURE_COLUMNS, window_length=16
    ).features
    # Fitting on repeated windows would overweight the center of the session and differs from
    # the declared one-candle/one-vote controlled transformation.
    repeated_mean = windows.reshape(-1, len(FEATURE_COLUMNS)).mean(axis=0)
    assert not np.allclose(repeated_mean, standardizer.mean)
