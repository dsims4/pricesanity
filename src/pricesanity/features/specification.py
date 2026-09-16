"""Describe causal representations without changing their evaluation population."""

from dataclasses import asdict, dataclass
from typing import Literal

import pandas as pd

from pricesanity.features.causal_window import CausalWindowCorpus, build_causal_windows


@dataclass(frozen=True)
class EvaluationUniverse:
    """Stable target-candle rule shared by every model on one leaderboard."""

    first_scored_candle_position: int

    def __post_init__(self) -> None:
        if self.first_scored_candle_position < 1:
            raise ValueError("Evaluation must begin after a prior candle exists.")


@dataclass(frozen=True)
class RepresentationSpec:
    """All causal and preprocessing choices that define one model input."""

    name: str
    window_length: int
    layout: Literal["tabular", "sequential"]
    standardization: Literal["none", "training_only"]
    polynomial_degree: int | None
    padding_policy: Literal["none", "left_zero_masked"]
    feature_columns: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.window_length <= 0:
            raise ValueError("Representation window length must be positive.")
        if not self.feature_columns or len(set(self.feature_columns)) != len(
            self.feature_columns
        ):
            raise ValueError("Representation feature columns must be unique and nonempty.")
        if self.polynomial_degree is not None and self.polynomial_degree < 2:
            raise ValueError("Polynomial representation degree must be at least two.")

    def to_dict(self) -> dict[str, object]:
        """Return a complete experiment-identity mapping."""

        return asdict(self)


def build_representation_corpus(
    sessions: list[pd.DataFrame] | tuple[pd.DataFrame, ...],
    specification: RepresentationSpec,
    universe: EvaluationUniverse,
    *,
    timestamp_column: str = "ts_event",
) -> CausalWindowCorpus:
    """Build context-specific inputs for one unchanged ordered target population."""

    if (
        specification.padding_policy == "none"
        and specification.window_length - 1 > universe.first_scored_candle_position
    ):
        # Silently dropping early targets would let longer-context models face a different
        # evaluation population, so an incompatible unpadded representation fails instead.
        raise ValueError(
            "Representation context does not fit before the common first target candle."
        )
    return build_causal_windows(
        sessions,
        feature_columns=specification.feature_columns,
        window_length=specification.window_length,
        timestamp_column=timestamp_column,
        target_start_position=universe.first_scored_candle_position,
        allow_partial_history=(specification.padding_policy == "left_zero_masked"),
    )


def assert_same_evaluation_universe(
    first: CausalWindowCorpus,
    second: CausalWindowCorpus,
) -> None:
    """Reject comparisons that differ in target identities, order, or human labels."""

    # Values may differ by representation; target identities and labels may not. Checking
    # both prevents a reordered or relabeled corpus from passing as a fair paired comparison.
    if (
        first.candlestick_ids != second.candlestick_ids
        or first.timestamps != second.timestamps
        or first.session_dates != second.session_dates
        or not (first.session_indices == second.session_indices).all()
        or not (first.current_targets == second.current_targets).all()
        or not (first.anticipated_targets == second.anticipated_targets).all()
    ):
        raise ValueError("Representations do not share the same evaluation universe.")
