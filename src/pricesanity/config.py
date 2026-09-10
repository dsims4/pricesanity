"""Typed configuration structures for PriceSanity."""

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class ProjectConfig:
    """Project identity and reproducibility settings."""

    name: str
    random_seed: int


@dataclass(frozen=True)
class DataConfig:
    """Source-data, timezone, and candlestick interval settings."""

    instrument: str
    source_timezone: str
    session_timezone: str
    timestamp_column: str
    source_interval: str
    target_interval: str
    require_complete_candlesticks: bool


@dataclass(frozen=True)
class SessionConfig:
    """Configured boundaries for the broad intraday trading window."""

    start_time: str
    end_time: str
    trading_weekdays: tuple[int, ...]


@dataclass(frozen=True)
class NormalizationConfig:
    """Price normalization settings used before model input."""

    scheme: str


@dataclass(frozen=True)
class AppConfig:
    """Complete application configuration grouped by responsibility."""

    project: ProjectConfig
    data: DataConfig
    session: SessionConfig
    normalization: NormalizationConfig


def load_config(path: str | Path) -> AppConfig:
    """Load typed project configuration from a YAML file.

    Args:
        path: Location of the YAML configuration file.

    Returns:
        Project configuration organized into typed sections.

    Raises:
        ValueError: If the YAML structure is invalid or a required key is
            missing.
    """
    # Treat string and Path inputs alike so the remaining file operations need
    # only one path representation.
    config_path = Path(path)

    # Parse the UTF-8 YAML into Python data before building the typed settings
    # used throughout the pipeline.
    with config_path.open(encoding="utf-8") as config_file:
        config_data = yaml.safe_load(config_file)

    # A named mapping is required because each typed configuration section is
    # located by its project, data, session, or normalization key.
    if not isinstance(config_data, dict):
        raise ValueError("The configuration must be a YAML mapping.")

    try:
        # Copy the session settings because its weekday list must be changed
        # without altering the data produced directly by the YAML parser.
        session_config_data = dict(config_data["session"])

        # Store weekdays as a tuple so a loaded configuration cannot gain or
        # lose eligible trading days during a run.
        session_config_data["trading_weekdays"] = tuple(
            session_config_data["trading_weekdays"]
        )

        # Build every typed section at load time so missing or misspelled
        # settings fail before the data pipeline begins.
        return AppConfig(
            project=ProjectConfig(**config_data["project"]),
            data=DataConfig(**config_data["data"]),
            session=SessionConfig(**session_config_data),
            normalization=NormalizationConfig(**config_data["normalization"]),
        )
    except KeyError as error:
        # Name the missing setting directly because a raw dictionary error
        # would not explain which part of the project configuration is invalid.
        raise ValueError(
            f"Missing required configuration key: {error.args[0]}"
        ) from error
