import pytest

from pricesanity.benchmark.cli import main
from pricesanity.benchmark.search_spaces import load_search_spaces


def test_benchmark_cli_dry_run_validates_without_training(capsys) -> None:
    """Dry-run plans remain usable without private data or model execution."""

    result = main([
        "run", "--model", "logistic_regression", "--track", "controlled",
        "--session-count", "2690", "--dry-run",
    ])
    output = capsys.readouterr().out
    assert result == 0
    assert "No model was trained" in output
    assert "logistic_regression" in output


def test_benchmark_cli_refuses_accidental_execution() -> None:
    """Execution requires actual source inputs and cannot rely on a declared count."""

    with pytest.raises(SystemExit):
        main([
            "run", "--model", "logistic_regression", "--track", "controlled",
            "--session-count", "2690",
        ])


def test_benchmark_cli_rejects_removed_legacy_mode() -> None:
    """Stage selection belongs to the dedicated tune/final/learning-curve commands."""

    with pytest.raises(SystemExit):
        main(["run", "--mode", "tuning", "--session-count", "2690", "--dry-run"])


def test_benchmark_profile_command_measures_infrastructure_only(tmp_path, capsys) -> None:
    """Profiling must measure benchmark infrastructure without opening model search."""

    output = tmp_path / "profile.json"
    result = main([
        "profile", "--synthetic", "--session-count", "4",
        "--candles-per-session", "18", "--output", str(output),
    ])
    assert result == 0
    assert output.is_file()
    assert '"scope": "benchmark infrastructure only; no model training"' in (
        capsys.readouterr().out
    )


def test_search_space_configuration_is_readable_and_validated() -> None:
    """Every proposed distribution is checked without importing Optuna."""

    spaces = load_search_spaces("configs/benchmark/search_spaces.yaml")
    assert spaces["polynomial_logistic"]["degree"] == {
        "type": "fixed", "value": 2
    }
    assert "transformer" in spaces
