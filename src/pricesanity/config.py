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
    market_timezone: str
    timestamp_column: str
    window_size: int
    input_frequency: str
    output_frequency: str
    require_complete_output_candlesticks: bool


@dataclass(frozen=True)
class SessionConfig:
    rth_enabled: bool
    start: str
    end: str
    weekdays: tuple[int, ...]


@dataclass(frozen=True)
class NormalizationConfig:
    name: str


@dataclass(frozen=True)
class AppConfig:
    project: ProjectConfig
    data: DataConfig
    session: SessionConfig
    normalization: NormalizationConfig


def load_config(path: str | Path) -> AppConfig:
    """Load project configuration from a YAML file."""
    config_path = Path(path)

    with config_path.open(encoding="utf-8") as file:
        raw = yaml.safe_load(file)

    if not isinstance(raw, dict):
        raise ValueError("The configuration must be a YAML mapping.")

    try:
        session_values = dict(raw["session"])
        session_values["weekdays"] = tuple(session_values["weekdays"])

        return AppConfig(
            project=ProjectConfig(**raw["project"]),
            data=DataConfig(**raw["data"]),
            session=SessionConfig(**session_values),
            normalization=NormalizationConfig(**raw["normalization"]),
        )
    except KeyError as error:
        raise ValueError(
            f"Missing required configuration key: {error.args[0]}"
        ) from error
