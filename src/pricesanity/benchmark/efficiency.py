"""Small cross-library helpers for model cost reporting."""

from pathlib import Path
from time import perf_counter
from typing import Callable, TypeVar


Result = TypeVar("Result")


def timed_call(function: Callable[[], Result]) -> tuple[Result, float]:
    """Measure elapsed wall-clock seconds around one explicit operation."""

    # Include waiting and caller-owned synchronization in the operation's observed cost.
    started_at = perf_counter()
    result = function()
    return result, perf_counter() - started_at


def serialized_model_size(path: str | Path) -> int:
    """Measure the actual persisted artifact rather than an in-memory estimate."""

    # Include persisted preprocessing and fitted state instead of estimating live objects.
    model_path = Path(path)
    if not model_path.is_file():
        raise ValueError("Serialized model artifact does not exist.")
    return model_path.stat().st_size


def parameter_count(model: object) -> int | None:
    """Count trainable PyTorch parameters when that concept applies."""

    # No parameter interface means inapplicable, not zero complexity for a classical model.
    parameters = getattr(model, "parameters", None)
    if not callable(parameters):
        return None
    # Exclude frozen tensors because this column describes trainable neural capacity.
    return sum(
        parameter.numel()
        for parameter in parameters()
        if getattr(parameter, "requires_grad", False)
    )
