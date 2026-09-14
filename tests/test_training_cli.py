"""Tests for the command-line training workflow."""

from contextlib import closing

import pandas as pd
import torch

from pricesanity.annotation.schema import CandlestickAnnotation, MarketRegime
from pricesanity.annotation.store import AnnotationStore
from pricesanity.data.identifiers import build_candlestick_id
from pricesanity.training.training_cli import build_argument_parser, main


def test_training_parser_preserves_first_experiment_defaults() -> None:
    """The default command describes the agreed 100/20/10 experiment."""

    parsed_arguments = build_argument_parser().parse_args(
        [
            "--normalized",
            "normalized.parquet",
            "--config",
            "configs/default.yaml",
        ]
    )

    assert parsed_arguments.training_sessions == 100
    assert parsed_arguments.validation_sessions == 20
    assert parsed_arguments.test_sessions == 10
    assert parsed_arguments.batch_size == 8
    assert parsed_arguments.device == "auto"


def test_training_command_builds_and_saves_one_complete_experiment(
    tmp_path,
) -> None:
    """Prepared features and SQLite labels produce a reloadable checkpoint."""

    timestamps = pd.to_datetime(
        [
            "2016-01-04T14:30:00Z",
            "2016-01-04T14:35:00Z",
            "2016-01-05T14:30:00Z",
            "2016-01-05T14:35:00Z",
            "2016-01-06T14:30:00Z",
            "2016-01-06T14:35:00Z",
        ]
    )
    normalized_data = pd.DataFrame(
        {
            "ts_event": timestamps,
            "instrument": ["ES.v.0"] * len(timestamps),
            "open_gap": [0.01, 0.0, -0.01, 0.0, 0.02, 0.0],
            "body": [0.01, 0.02, -0.01, -0.02, 0.01, 0.02],
            "high_from_close": [0.01] * len(timestamps),
            "low_from_close": [-0.01] * len(timestamps),
        }
    )
    normalized_path = tmp_path / "normalized.parquet"
    normalized_data.to_parquet(normalized_path, index=False)

    database_path = tmp_path / "annotations.sqlite3"
    with closing(AnnotationStore(database_path)) as store:
        for index, timestamp in enumerate(timestamps):
            regime = MarketRegime.BULL if index % 2 == 0 else MarketRegime.BEAR
            store.save(
                CandlestickAnnotation(
                    candlestick_id=build_candlestick_id(
                        "ES.v.0",
                        timestamp,
                        "5min",
                    ),
                    current_regime=regime,
                    anticipated_regime=regime,
                )
            )

    checkpoint_path = tmp_path / "trained-model.pt"
    exit_status = main(
        [
            "--normalized",
            str(normalized_path),
            "--config",
            "configs/default.yaml",
            "--database",
            str(database_path),
            "--checkpoint",
            str(checkpoint_path),
            "--training-sessions",
            "1",
            "--validation-sessions",
            "1",
            "--test-sessions",
            "1",
            "--batch-size",
            "1",
            "--epochs",
            "1",
            "--patience",
            "1",
            "--device",
            "cpu",
        ]
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    assert exit_status == 0
    assert checkpoint_path.is_file()
    assert checkpoint["best_epoch"] == 1
    assert checkpoint["test"]["current"]["support"] == 2


def test_removed_step_option_cannot_silently_select_old_protocol() -> None:
    """Old rolling commands must fail instead of silently changing experiment meaning."""

    import pytest

    with pytest.raises(SystemExit):
        build_argument_parser().parse_args([
            "--normalized", "normalized.parquet", "--config", "configs/default.yaml",
            "--walk-forward", "--step-sessions", "10",
        ])


def test_hidden_width_is_configurable_and_part_of_experiment_identity() -> None:
    """Capacity changes must not silently reuse a smaller completed experiment."""

    import pytest
    from pricesanity.training.training_cli import _model_config

    arguments = build_argument_parser().parse_args([
        "--normalized", "normalized.parquet", "--config", "configs/default.yaml",
    ])
    assert _model_config(arguments).model_dimension == 48
    assert _model_config(arguments).feedforward_dimension == 192
    arguments.model_dimension = 12
    assert _model_config(arguments).feedforward_dimension == 48
    arguments.model_dimension = 13
    with pytest.raises(ValueError, match="divisible"):
        _model_config(arguments)
