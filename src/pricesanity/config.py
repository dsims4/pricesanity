"""Typed configuration structures for PriceSanity."""

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class ProjectConfig:
    name: str
    random_seed: int


@dataclass(frozen=True)
class DataConfig:
    instrument: str
    source_timezone: str
    session_timezone: str
    timestamp_column: str
    source_interval: str
    target_interval: str
    require_complete_candlesticks: bool


@dataclass(frozen=True)
class SessionConfig:
    start_time: str
    end_time: str
    trading_weekdays: tuple[int, ...]


@dataclass(frozen=True)
class NormalizationConfig:
    scheme: str


@dataclass(frozen=True)
class AppConfig:
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
    # Convert the location to a Path so string and Path arguments use the same
    # file operations.
    config_path = Path(path)

    # Load the UTF-8 YAML contents into a dictionary for typed conversion.
    with config_path.open(encoding="utf-8") as config_file:
        config_data = yaml.safe_load(config_file)

    # Require a key-value mapping because each configuration section is named.
    if not isinstance(config_data, dict):
        raise ValueError("The configuration must be a YAML mapping.")

    try:
        # Copy the session settings so the weekday list can be converted without
        # changing the original YAML data.
        session_config_data = dict(config_data["session"])

        # Convert the weekday list into the immutable tuple used by SessionConfig.
        session_config_data["trading_weekdays"] = tuple(
            session_config_data["trading_weekdays"]
        )

        # Build typed sections so later code receives predictable field names
        # and attribute access.
        return AppConfig(
            project=ProjectConfig(**config_data["project"]),
            data=DataConfig(**config_data["data"]),
            session=SessionConfig(**session_config_data),
            normalization=NormalizationConfig(**config_data["normalization"]),
        )
    except KeyError as error:
        # Replace a raw dictionary error with the missing configuration key.
        raise ValueError(
            f"Missing required configuration key: {error.args[0]}"
        ) from error
