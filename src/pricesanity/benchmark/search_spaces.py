"""Validate readable future tuning spaces without depending on a tuning library."""

from pathlib import Path
from typing import Any

import yaml

from pricesanity.benchmark.registry import (
    get_model_family,
    validate_conceptual_parameters,
)


def load_search_spaces(path: str | Path) -> dict[str, dict[str, dict[str, Any]]]:
    """Reject malformed model or distribution settings before a long study begins."""

    with Path(path).open(encoding="utf-8") as search_file:
        values = yaml.safe_load(search_file)
    if not isinstance(values, dict) or values.get("format_version") != 1:
        raise ValueError("Search-space configuration must use format version 1.")

    spaces = {}
    for model_name, parameters in values.items():
        if model_name == "format_version":
            continue
        get_model_family(str(model_name))
        if not isinstance(parameters, dict) or not parameters:
            raise ValueError(f"Search space for {model_name} must be a nonempty mapping.")
        validated_parameters = {}
        for parameter_name, specification in parameters.items():
            if not isinstance(specification, dict):
                raise ValueError(
                    f"Search parameter {model_name}.{parameter_name} must be a mapping."
                )
            distribution_type = specification.get("type")
            if distribution_type not in {
                "fixed", "categorical", "integer", "float", "log_float"
            }:
                raise ValueError(
                    f"Search parameter {model_name}.{parameter_name} has an unknown type."
                )
            if distribution_type == "categorical" and not specification.get("values"):
                raise ValueError("Categorical search parameters require values.")
            if distribution_type == "fixed" and "value" not in specification:
                raise ValueError("Fixed search parameters require one value.")
            if distribution_type in {"integer", "float", "log_float"}:
                low = specification.get("low")
                high = specification.get("high")
                if not isinstance(low, (int, float)) or not isinstance(
                    high, (int, float)
                ) or low >= high:
                    raise ValueError("Numeric search parameters require low < high.")
                if distribution_type == "log_float" and low <= 0:
                    raise ValueError("Log-float search parameters require a positive low.")
            validated_parameters[str(parameter_name)] = dict(specification)
        spaces[str(model_name)] = validated_parameters
        validate_conceptual_parameters(str(model_name), validated_parameters)
    return spaces


def representative_candidate(
    search_space: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Choose one legal deterministic value from every conceptual distribution.

    This is a construction smoke candidate, not a tuning recommendation. Numeric ranges use
    their low value so models remain small and quick in infrastructure tests.
    """

    candidate = {}
    for parameter_name, specification in search_space.items():
        distribution_type = specification["type"]
        if distribution_type == "fixed":
            value = specification["value"]
        elif distribution_type == "categorical":
            value = specification["values"][0]
        else:
            value = specification["low"]
        candidate[parameter_name] = value
    return candidate
