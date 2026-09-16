"""Shared causal market representations used by every model family."""

from pricesanity.features.causal_window import (
    CausalWindowCorpus,
    build_causal_windows,
)
from pricesanity.features.representations import (
    ArrayStandardizer,
    sequential_representation,
    tabular_representation,
    fit_unique_candle_standardizer,
    transform_sessions,
)
from pricesanity.features.specification import (
    EvaluationUniverse,
    RepresentationSpec,
    assert_same_evaluation_universe,
    build_representation_corpus,
)


# Preserve one named order before arrays remove the meaning of their columns.
FEATURE_COLUMNS = (
    "open_gap",
    "body",
    "high_from_close",
    "low_from_close",
)


__all__ = [
    "ArrayStandardizer",
    "CausalWindowCorpus",
    "FEATURE_COLUMNS",
    "EvaluationUniverse",
    "RepresentationSpec",
    "assert_same_evaluation_universe",
    "build_representation_corpus",
    "build_causal_windows",
    "sequential_representation",
    "tabular_representation",
    "fit_unique_candle_standardizer",
    "transform_sessions",
]
