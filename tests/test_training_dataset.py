"""Tests for aligning model features with human annotations."""

import pandas as pd
import pytest
import torch

from pricesanity.annotation.schema import (
    CandlestickAnnotation,
    MarketRegime,
)
from pricesanity.data.identifiers import build_candlestick_id
from pricesanity.training.dataset import (
    AnnotatedSessionSplit,
    IGNORED_TARGET,
    TensorSession,
    TensorSessionDataset,
    build_annotated_candlesticks,
    build_complete_annotated_sessions,
    collate_tensor_sessions,
    convert_session_to_tensors,
    create_training_data_loaders,
    split_annotated_sessions,
)


def _build_annotated_session(
    session_date: str,
    *,
    candle_count: int = 3,
) -> pd.DataFrame:
    """Build one small completed session for loader tests."""

    timestamps = pd.date_range(
        f"{session_date}T14:30:00Z",
        periods=candle_count,
        freq="5min",
    )

    # Distinct feature constants expose axis reordering, and different head labels expose target
    # swaps. Varying candle_count exercises padding without changing the feature contract.
    return pd.DataFrame(
        {
            "session_date": [timestamps[0].date()] * candle_count,
            "candlestick_id": [
                f"{session_date}-candle-{position}"
                for position in range(candle_count)
            ],
            "ts_event": timestamps,
            "open_gap": [0.01] * candle_count,
            "body": [0.02] * candle_count,
            "high_from_close": [0.03] * candle_count,
            "low_from_close": [-0.01] * candle_count,
            "current_target": [0] * candle_count,
            "anticipated_target": [2] * candle_count,
        }
    )


def test_build_annotated_candlesticks_aligns_by_stable_id() -> None:
    """Annotations match their candles by identity rather than database order."""

    timestamps = pd.to_datetime(
        [
            "2016-01-04T14:30:00Z",
            "2016-01-04T14:35:00Z",
        ]
    )
    normalized_data = pd.DataFrame(
        {
            "ts_event": timestamps,
            "instrument": ["ES.v.0", "ES.v.0"],
            "open_gap": [0.01, 0.00],
            "body": [0.02, -0.01],
            "high_from_close": [0.01, 0.02],
            "low_from_close": [-0.02, -0.01],
        }
    )
    first_id = build_candlestick_id(
        "ES.v.0",
        timestamps[0],
        "5min",
    )
    second_id = build_candlestick_id(
        "ES.v.0",
        timestamps[1],
        "5min",
    )

    # Reverse storage order to prove row position cannot determine alignment.
    annotations = [
        CandlestickAnnotation(
            second_id,
            MarketRegime.BEAR,
            MarketRegime.RANGE,
        ),
        CandlestickAnnotation(
            first_id,
            MarketRegime.BULL,
            MarketRegime.BEAR,
        ),
    ]

    annotated_data = build_annotated_candlesticks(
        normalized_data,
        annotations,
        timestamp_column="ts_event",
        interval="5min",
    )

    assert annotated_data["candlestick_id"].tolist() == [
        first_id,
        second_id,
    ]
    assert annotated_data["current_target"].tolist() == [0, 1]
    assert annotated_data["anticipated_target"].tolist() == [1, 2]


def test_build_annotated_candlesticks_rejects_orphan_annotation() -> None:
    """An annotation from another artifact cannot silently disappear."""

    timestamp = pd.Timestamp("2016-01-04T14:30:00Z")
    normalized_data = pd.DataFrame(
        {
            "ts_event": [timestamp],
            "instrument": ["ES.v.0"],
            "open_gap": [0.01],
            "body": [0.02],
            "high_from_close": [0.01],
            "low_from_close": [-0.02],
        }
    )
    orphaned_annotation = CandlestickAnnotation(
        build_candlestick_id(
            "ES.v.0",
            timestamp + pd.Timedelta(minutes=5),
            "5min",
        ),
        MarketRegime.BULL,
        MarketRegime.RANGE,
    )

    with pytest.raises(ValueError, match="do not match normalized candles"):
        build_annotated_candlesticks(
            normalized_data,
            [orphaned_annotation],
            timestamp_column="ts_event",
            interval="5min",
        )


def test_build_annotated_candlesticks_rejects_duplicate_annotations() -> None:
    """One candle cannot receive two competing training rows."""

    timestamp = pd.Timestamp("2016-01-04T14:30:00Z")
    normalized_data = pd.DataFrame(
        {
            "ts_event": [timestamp],
            "instrument": ["ES.v.0"],
            "open_gap": [0.01],
            "body": [0.02],
            "high_from_close": [0.01],
            "low_from_close": [-0.02],
        }
    )
    annotation = CandlestickAnnotation(
        build_candlestick_id("ES.v.0", timestamp, "5min"),
        MarketRegime.BULL,
        MarketRegime.RANGE,
    )

    with pytest.raises(ValueError, match="identifiers must be unique"):
        build_annotated_candlesticks(
            normalized_data,
            [annotation, annotation],
            timestamp_column="ts_event",
            interval="5min",
        )


def test_build_complete_annotated_sessions_preserves_chronology() -> None:
    """Complete labels become date-ordered sessions with ordered candles."""

    timestamps = pd.to_datetime(
        [
            "2016-01-04T14:30:00Z",
            "2016-01-04T14:35:00Z",
            "2016-01-05T14:30:00Z",
        ]
    )
    normalized_data = pd.DataFrame(
        {
            "ts_event": timestamps,
            "instrument": "ES.v.0",
            "open_gap": [0.01, 0.00, -0.01],
            "body": [0.02, -0.01, 0.01],
            "high_from_close": [0.01, 0.02, 0.01],
            "low_from_close": [-0.02, -0.01, -0.01],
        }
    )
    annotations = [
        CandlestickAnnotation(
            build_candlestick_id("ES.v.0", timestamp, "5min"),
            MarketRegime.BULL,
            MarketRegime.RANGE,
        )
        for timestamp in timestamps
    ]

    sessions = build_complete_annotated_sessions(
        normalized_data,
        annotations,
        timestamp_column="ts_event",
        interval="5min",
        session_timezone="America/New_York",
    )

    assert len(sessions) == 2
    assert sessions[0]["ts_event"].tolist() == timestamps[:2].tolist()
    assert sessions[1]["ts_event"].tolist() == timestamps[2:].tolist()


def test_build_complete_annotated_sessions_rejects_partial_session() -> None:
    """A started session cannot silently become a shortened training sequence."""

    timestamps = pd.to_datetime(
        [
            "2016-01-04T14:30:00Z",
            "2016-01-04T14:35:00Z",
        ]
    )
    normalized_data = pd.DataFrame(
        {
            "ts_event": timestamps,
            "instrument": "ES.v.0",
            "open_gap": [0.01, 0.00],
            "body": [0.02, -0.01],
            "high_from_close": [0.01, 0.02],
            "low_from_close": [-0.02, -0.01],
        }
    )
    one_annotation = CandlestickAnnotation(
        build_candlestick_id("ES.v.0", timestamps[0], "5min"),
        MarketRegime.BULL,
        MarketRegime.RANGE,
    )

    with pytest.raises(ValueError, match="partially annotated"):
        build_complete_annotated_sessions(
            normalized_data,
            [one_annotation],
            timestamp_column="ts_event",
            interval="5min",
            session_timezone="America/New_York",
        )


def test_split_annotated_sessions_uses_chronological_boundaries() -> None:
    """Training, validation, and test sessions remain ordered and separate."""

    sessions = [
        pd.DataFrame({"session_date": [session_date]})
        for session_date in pd.date_range("2016-01-04", periods=5).date
    ]

    split = split_annotated_sessions(
        sessions,
        training_session_count=2,
        validation_session_count=2,
        test_session_count=1,
    )

    assert [session.session_date.iloc[0] for session in split.training] == [
        sessions[0].session_date.iloc[0],
        sessions[1].session_date.iloc[0],
    ]
    assert [session.session_date.iloc[0] for session in split.validation] == [
        sessions[2].session_date.iloc[0],
        sessions[3].session_date.iloc[0],
    ]
    assert [session.session_date.iloc[0] for session in split.test] == [
        sessions[4].session_date.iloc[0]
    ]


def test_split_annotated_sessions_requires_exact_session_count() -> None:
    """A split cannot silently omit sessions or manufacture missing ones."""

    sessions = [
        pd.DataFrame({"session_date": [session_date]})
        for session_date in pd.date_range("2016-01-04", periods=4).date
    ]

    with pytest.raises(ValueError, match="requests 5 sessions"):
        split_annotated_sessions(
            sessions,
            training_session_count=2,
            validation_session_count=2,
            test_session_count=1,
        )


def test_convert_session_to_tensors_preserves_rows_and_feature_order() -> None:
    """One session becomes aligned float features and integer targets."""

    timestamps = pd.to_datetime(
        [
            "2016-01-04T14:30:00Z",
            "2016-01-04T14:35:00Z",
        ]
    )
    session = pd.DataFrame(
        {
            "session_date": [timestamps[0].date()] * 2,
            "candlestick_id": ["candle-1", "candle-2"],
            "ts_event": timestamps,
            "open_gap": [0.01, 0.02],
            "body": [-0.01, -0.02],
            "high_from_close": [0.03, 0.04],
            "low_from_close": [-0.03, -0.04],
            "current_target": [0, 1],
            "anticipated_target": [2, 0],
        }
    )

    tensor_session = convert_session_to_tensors(
        session,
        timestamp_column="ts_event",
    )

    assert tensor_session.features.shape == (2, 4)
    assert tensor_session.features.dtype is torch.float32
    torch.testing.assert_close(
        tensor_session.features,
        torch.tensor(
            [
                [0.01, -0.01, 0.03, -0.03],
                [0.02, -0.02, 0.04, -0.04],
            ],
            dtype=torch.float32,
        ),
    )
    assert tensor_session.current_targets.dtype is torch.int64
    assert tensor_session.anticipated_targets.dtype is torch.int64
    assert tensor_session.current_targets.tolist() == [0, 1]
    assert tensor_session.anticipated_targets.tolist() == [2, 0]
    assert tensor_session.candlestick_ids == ("candle-1", "candle-2")
    assert tensor_session.timestamps == tuple(timestamps)


def test_collate_tensor_sessions_masks_only_shortened_session_padding() -> None:
    """Batch padding extends short sessions without creating real candles."""

    first_timestamps = tuple(
        pd.date_range("2016-01-04T14:30:00Z", periods=3, freq="5min")
    )
    second_timestamps = tuple(
        pd.date_range("2016-01-05T14:30:00Z", periods=2, freq="5min")
    )
    sessions = [
        TensorSession(
            session_date=timestamp_group[0].date(),
            candlestick_ids=tuple(
                f"session-{session_number}-candle-{position}"
                for position in range(len(timestamp_group))
            ),
            timestamps=timestamp_group,
            features=torch.full(
                (len(timestamp_group), 4),
                float(session_number),
                dtype=torch.float32,
            ),
            current_targets=torch.zeros(len(timestamp_group), dtype=torch.int64),
            anticipated_targets=torch.ones(len(timestamp_group), dtype=torch.int64),
        )
        for session_number, timestamp_group in enumerate(
            (first_timestamps, second_timestamps),
            start=1,
        )
    ]

    batch = collate_tensor_sessions(sessions)

    assert batch.features.shape == (2, 3, 4)
    assert batch.current_targets.shape == (2, 3)
    assert batch.anticipated_targets.shape == (2, 3)
    assert batch.lengths.tolist() == [3, 2]
    assert batch.padding_mask.tolist() == [
        [False, False, False],
        [False, False, True],
    ]
    assert batch.features[1, 2].tolist() == [0.0, 0.0, 0.0, 0.0]
    assert batch.current_targets[1, 2].item() == IGNORED_TARGET
    assert batch.anticipated_targets[1, 2].item() == IGNORED_TARGET


def test_collate_tensor_sessions_rejects_empty_batch() -> None:
    """An empty batch cannot define a sequence length for padding."""

    with pytest.raises(ValueError, match="at least one session"):
        collate_tensor_sessions([])


def test_tensor_session_dataset_indexes_complete_sessions() -> None:
    """The Dataset exposes sessions without changing their candle order."""

    first_session = convert_session_to_tensors(
        _build_annotated_session("2016-01-04"),
        timestamp_column="ts_event",
    )
    second_session = convert_session_to_tensors(
        _build_annotated_session("2016-01-05", candle_count=2),
        timestamp_column="ts_event",
    )

    dataset = TensorSessionDataset([first_session, second_session])

    assert len(dataset) == 2
    assert dataset[0] is first_session
    assert dataset[1] is second_session
    assert dataset[1].candlestick_ids == (
        "2016-01-05-candle-0",
        "2016-01-05-candle-1",
    )


def test_tensor_session_dataset_rejects_empty_collection() -> None:
    """A Dataset must contain at least one complete training example."""

    with pytest.raises(ValueError, match="cannot be empty"):
        TensorSessionDataset([])


def test_create_training_data_loaders_preserves_split_behavior() -> None:
    """Training shuffles reproducibly while evaluation stays chronological."""

    sessions = tuple(
        _build_annotated_session(session_date.strftime("%Y-%m-%d"))
        for session_date in pd.date_range("2016-01-04", periods=8)
    )
    split = AnnotatedSessionSplit(
        training=sessions[:4],
        validation=sessions[4:6],
        test=sessions[6:],
    )

    first_loaders = create_training_data_loaders(
        split,
        timestamp_column="ts_event",
        batch_size=2,
        random_seed=42,
    )
    second_loaders = create_training_data_loaders(
        split,
        timestamp_column="ts_event",
        batch_size=2,
        random_seed=42,
    )

    # Two loaders created with the same seed must produce the same first epoch.
    first_training_order = tuple(
        session_date
        for batch in first_loaders.training
        for session_date in batch.session_dates
    )
    second_training_order = tuple(
        session_date
        for batch in second_loaders.training
        for session_date in batch.session_dates
    )
    assert first_training_order == second_training_order
    assert set(first_training_order) == {
        session["session_date"].iloc[0] for session in split.training
    }

    # Evaluation loaders must retain their chronological leakage boundary.
    validation_order = tuple(
        session_date
        for batch in first_loaders.validation
        for session_date in batch.session_dates
    )
    test_order = tuple(
        session_date
        for batch in first_loaders.test
        for session_date in batch.session_dates
    )
    assert validation_order == tuple(
        session["session_date"].iloc[0] for session in split.validation
    )
    assert test_order == tuple(
        session["session_date"].iloc[0] for session in split.test
    )


def test_create_training_data_loaders_rejects_invalid_batch_size() -> None:
    """A loader cannot form batches from a nonpositive session count."""

    one_session = (_build_annotated_session("2016-01-04"),)
    split = AnnotatedSessionSplit(
        training=one_session,
        validation=one_session,
        test=one_session,
    )

    with pytest.raises(ValueError, match="batch size must be positive"):
        create_training_data_loaders(
            split,
            timestamp_column="ts_event",
            batch_size=0,
            random_seed=42,
        )
