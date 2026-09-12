from pathlib import Path

import pandas as pd
import pytest

from pricesanity.data import build_dataset
from pricesanity.data.pipeline import PreparedCandlestickSessions


def test_build_dataset_from_exports_saves_pipeline_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Use distinct marker tables so the test can verify that each loaded export
    # reaches the matching pipeline argument.
    candlestick_data = pd.DataFrame({"source": ["candlesticks"]})
    status_data = pd.DataFrame({"source": ["status"]})
    data_conditions = pd.DataFrame({"source": ["conditions"]})
    prepared_candlestick_data = pd.DataFrame(
        {
            "ts_event": [pd.Timestamp("2026-09-09 13:30:00Z")],
            "instrument": ["ES"],
            "open_gap": [0.01],
            "body": [0.0],
            "high_from_close": [0.01],
            "low_from_close": [-0.01],
        }
    )
    prepared_ohlc_data = pd.DataFrame(
        {
            "ts_event": [pd.Timestamp("2026-09-09 13:30:00Z")],
            "open": [6500.0],
            "high": [6501.0],
            "low": [6499.0],
            "close": [6500.5],
        }
    )

    # Supply only the configuration properties used at this file boundary so
    # the test remains focused on orchestration rather than YAML parsing.
    config = type(
        "TestConfig",
        (),
        {
            "data": type(
                "TestDataConfig",
                (),
                {"timestamp_column": "ts_event", "source_timezone": "UTC"},
            )()
        },
    )()

    # Replace already-tested loaders with marker-returning functions so this
    # test can isolate the order and arguments used by the file workflow.
    monkeypatch.setattr(build_dataset, "load_config", lambda path: config)
    monkeypatch.setattr(
        build_dataset,
        "load_ohlc_csv",
        lambda path, timestamp_column, source_timezone: candlestick_data,
    )
    monkeypatch.setattr(
        build_dataset,
        "load_status_csv",
        lambda path, timestamp_column: status_data,
    )
    monkeypatch.setattr(
        build_dataset,
        "load_dataset_conditions_json",
        lambda path: data_conditions,
    )

    # Capture pipeline inputs while returning a realistic model-ready table for
    # the Parquet write that follows.
    received_pipeline_inputs: dict[str, object] = {}

    def fake_prepare_candlestick_session_tables(
        received_candlestick_data: pd.DataFrame,
        received_status_data: pd.DataFrame,
        received_data_conditions: pd.DataFrame,
        *,
        config: object,
    ) -> PreparedCandlestickSessions:
        # Save object identities because the orchestration should pass each
        # loader's exact result directly into the preparation pipeline.
        received_pipeline_inputs.update(
            {
                "candlesticks": received_candlestick_data,
                "status": received_status_data,
                "conditions": received_data_conditions,
                "config": config,
            }
        )
        return PreparedCandlestickSessions(
            ohlc=prepared_ohlc_data,
            normalized=prepared_candlestick_data,
        )

    monkeypatch.setattr(
        build_dataset,
        "prepare_candlestick_session_tables",
        fake_prepare_candlestick_session_tables,
    )

    # Use a missing nested directory to confirm the command creates only its
    # requested processed-data destination.
    output_path = tmp_path / "processed" / "candlesticks.parquet"
    interim_output_path = tmp_path / "interim" / "candlesticks.parquet"
    returned_candlestick_data = build_dataset.build_dataset_from_exports(
        tmp_path / "ohlc.csv",
        tmp_path / "status.csv",
        tmp_path / "condition.json",
        output_path,
        config_path=tmp_path / "config.yaml",
        interim_output_path=interim_output_path,
    )

    # Confirm every source reached its intended argument without copying or
    # replacing the in-memory tables between the loader and pipeline.
    assert received_pipeline_inputs == {
        "candlesticks": candlestick_data,
        "status": status_data,
        "conditions": data_conditions,
        "config": config,
    }
    assert returned_candlestick_data is prepared_candlestick_data

    # Read the artifact back to prove that it contains the pipeline result and
    # retained its timezone-aware timestamp representation.
    saved_candlestick_data = pd.read_parquet(output_path)
    pd.testing.assert_frame_equal(
        saved_candlestick_data,
        prepared_candlestick_data,
    )

    # The second artifact preserves the validated real prices needed to draw
    # the annotation chart without exposing normalized values to the annotator.
    saved_ohlc_data = pd.read_parquet(interim_output_path)
    pd.testing.assert_frame_equal(saved_ohlc_data, prepared_ohlc_data)


def test_build_dataset_from_exports_protects_existing_output(
    tmp_path: Path,
) -> None:
    # Create an existing processed artifact whose replacement was not approved.
    output_path = tmp_path / "candlesticks.parquet"
    output_path.write_bytes(b"existing dataset")

    # Verify the workflow stops before loading inputs or changing the artifact.
    with pytest.raises(FileExistsError, match="--overwrite"):
        build_dataset.build_dataset_from_exports(
            tmp_path / "ohlc.csv",
            tmp_path / "status.csv",
            tmp_path / "condition.json",
            output_path,
            config_path=tmp_path / "config.yaml",
            interim_output_path=tmp_path / "interim.parquet",
        )

    # Confirm the refusal left the previous dataset unchanged.
    assert output_path.read_bytes() == b"existing dataset"


def test_build_dataset_from_exports_requires_parquet_output(
    tmp_path: Path,
) -> None:
    # Use a CSV destination that would discard parts of the processed schema.
    output_path = tmp_path / "candlesticks.csv"

    # Verify the workflow requires the type-preserving storage format.
    with pytest.raises(ValueError, match="must end in .parquet"):
        build_dataset.build_dataset_from_exports(
            tmp_path / "ohlc.csv",
            tmp_path / "status.csv",
            tmp_path / "condition.json",
            output_path,
            config_path=tmp_path / "config.yaml",
            interim_output_path=tmp_path / "interim.parquet",
        )


@pytest.mark.parametrize("failed_write", [1, 2, None])
@pytest.mark.parametrize("existing_outputs", [False, True])
def test_artifact_pair_is_staged_before_publication(
    tmp_path, monkeypatch, failed_write, existing_outputs
):
    prepared = PreparedCandlestickSessions(
        ohlc=pd.DataFrame({"open": [100.]}),
        normalized=pd.DataFrame({"open_gap": [0.01]}),
    )
    monkeypatch.setattr(build_dataset, "load_ohlc_csv", lambda *a, **k: pd.DataFrame())
    monkeypatch.setattr(build_dataset, "load_status_csv", lambda *a, **k: pd.DataFrame())
    monkeypatch.setattr(build_dataset, "load_dataset_conditions_json", lambda *a: pd.DataFrame())
    monkeypatch.setattr(build_dataset, "prepare_candlestick_session_tables", lambda *a, **k: prepared)
    output = tmp_path / "processed" / "data.parquet"
    interim = tmp_path / "interim" / "data.parquet"
    for path in (output, interim):
        path.parent.mkdir()
        if existing_outputs:
            path.write_bytes(b"original")

    def assert_unpublished():
        for path in (output, interim):
            if existing_outputs:
                assert path.read_bytes() == b"original"
            else:
                assert not path.exists()

    original_write = pd.DataFrame.to_parquet
    write_count = 0

    def write(table, path, **kwargs):
        nonlocal write_count
        write_count += 1
        assert_unpublished()
        assert Path(path).parent.parent in (output.parent, interim.parent)
        if write_count == failed_write:
            Path(path).write_bytes(b"partial")
            raise OSError("simulated disk failure")
        return original_write(table, path, **kwargs)

    monkeypatch.setattr(pd.DataFrame, "to_parquet", write)

    def build():
        return build_dataset.build_dataset_from_exports(
            tmp_path / "source.csv", tmp_path / "status.csv", tmp_path / "conditions.json",
            output, config_path="configs/default.yaml", interim_output_path=interim,
            overwrite=existing_outputs,
        )

    if failed_write is not None:
        with pytest.raises(OSError, match="simulated disk failure"):
            build()
        assert_unpublished()
    else:
        assert build() is prepared.normalized
        pd.testing.assert_frame_equal(pd.read_parquet(output), prepared.normalized)
        pd.testing.assert_frame_equal(pd.read_parquet(interim), prepared.ohlc)
    assert not list(tmp_path.rglob(".pricesanity-*"))
