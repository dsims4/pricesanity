"""Construct fixed causal windows without joining neighboring sessions."""

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CausalWindowCorpus:
    """Aligned fixed-length inputs, labels, and audit identities."""

    features: np.ndarray
    current_targets: np.ndarray
    anticipated_targets: np.ndarray
    previous_current_targets: np.ndarray
    previous_anticipated_targets: np.ndarray
    session_indices: np.ndarray
    candle_positions: np.ndarray
    session_dates: tuple[date, ...]
    candlestick_ids: tuple[str, ...]
    timestamps: tuple[pd.Timestamp, ...]
    feature_columns: tuple[str, ...]
    window_length: int
    valid_history_mask: np.ndarray

    def __post_init__(self) -> None:
        """Refuse arrays whose rows could no longer be audited together."""

        sample_count = self.features.shape[0]
        if self.features.ndim != 3:
            raise ValueError("Causal features must have sample, time, and feature axes.")
        if self.features.shape[1:] != (
            self.window_length,
            len(self.feature_columns),
        ):
            raise ValueError("Causal feature dimensions do not match their description.")
        aligned_lengths = (
            len(self.current_targets),
            len(self.anticipated_targets),
            len(self.previous_current_targets),
            len(self.previous_anticipated_targets),
            len(self.session_indices),
            len(self.candle_positions),
            len(self.session_dates),
            len(self.candlestick_ids),
            len(self.timestamps),
            len(self.valid_history_mask),
        )
        if any(length != sample_count for length in aligned_lengths):
            raise ValueError("Causal inputs, targets, and identities must stay aligned.")
        if self.valid_history_mask.shape != self.features.shape[:2]:
            raise ValueError("Causal validity mask must match sample and time axes.")
        if self.valid_history_mask.dtype != np.bool_:
            raise ValueError("Causal validity mask must contain booleans.")

    def select_sessions(
        self,
        allowed_session_indices: Sequence[int],
    ) -> "CausalWindowCorpus":
        """Select whole sessions while retaining their original corpus identities."""

        allowed_indices = np.asarray(tuple(allowed_session_indices), dtype=np.int64)
        if allowed_indices.ndim != 1 or len(set(allowed_indices.tolist())) != len(
            allowed_indices
        ):
            raise ValueError("Selected session indices must be unique.")

        selected_rows = np.isin(self.session_indices, allowed_indices)
        row_positions = np.flatnonzero(selected_rows)
        return CausalWindowCorpus(
            features=self.features[selected_rows].copy(),
            current_targets=self.current_targets[selected_rows].copy(),
            anticipated_targets=self.anticipated_targets[selected_rows].copy(),
            previous_current_targets=self.previous_current_targets[selected_rows].copy(),
            previous_anticipated_targets=(
                self.previous_anticipated_targets[selected_rows].copy()
            ),
            session_indices=self.session_indices[selected_rows].copy(),
            candle_positions=self.candle_positions[selected_rows].copy(),
            session_dates=tuple(self.session_dates[position] for position in row_positions),
            candlestick_ids=tuple(
                self.candlestick_ids[position] for position in row_positions
            ),
            timestamps=tuple(self.timestamps[position] for position in row_positions),
            feature_columns=self.feature_columns,
            window_length=self.window_length,
            valid_history_mask=self.valid_history_mask[selected_rows].copy(),
        )


def build_causal_windows(
    sessions: Sequence[pd.DataFrame],
    *,
    feature_columns: Sequence[str],
    window_length: int = 16,
    timestamp_column: str = "ts_event",
    target_start_position: int | None = None,
    allow_partial_history: bool = False,
) -> CausalWindowCorpus:
    """Build one full historical window ending at each eligible labeled candle.

    The label belongs to the final candle in a window. Its completed OHLC geometry is
    therefore available, while every later candle remains outside the sample.
    """

    if window_length <= 0:
        raise ValueError("Causal window length must be positive.")
    feature_columns = tuple(feature_columns)
    if not feature_columns or len(set(feature_columns)) != len(feature_columns):
        raise ValueError("Causal feature columns must be unique and nonempty.")
    if not sessions:
        raise ValueError("Causal windows require at least one annotated session.")

    if target_start_position is not None and target_start_position < 0:
        raise ValueError("Causal target start position cannot be negative.")

    feature_window_batches: list[np.ndarray] = []
    valid_history_batches: list[np.ndarray] = []
    current_targets: list[int] = []
    anticipated_targets: list[int] = []
    previous_current_targets: list[int] = []
    previous_anticipated_targets: list[int] = []
    session_indices: list[int] = []
    candle_positions: list[int] = []
    sample_session_dates: list[date] = []
    candlestick_ids: list[str] = []
    sample_timestamps: list[pd.Timestamp] = []
    previous_session_date: date | None = None

    required_columns = {
        *feature_columns,
        "current_target",
        "anticipated_target",
        "session_date",
        "candlestick_id",
        timestamp_column,
    }
    for session_index, session in enumerate(sessions):
        missing_columns = required_columns.difference(session.columns)
        if missing_columns:
            raise ValueError(
                "Annotated session is missing causal-window columns: "
                + ", ".join(sorted(missing_columns))
            )
        if session.empty:
            raise ValueError("A causal-window session cannot be empty.")

        unique_dates = session["session_date"].drop_duplicates()
        if len(unique_dates) != 1:
            raise ValueError("Each causal-window input must contain one session date.")
        session_date = pd.Timestamp(unique_dates.iloc[0]).date()
        if previous_session_date is not None and session_date <= previous_session_date:
            raise ValueError("Causal-window sessions must be uniquely chronological.")
        previous_session_date = session_date

        timestamps = pd.to_datetime(session[timestamp_column], utc=True, errors="coerce")
        if (
            timestamps.isna().any()
            or timestamps.duplicated().any()
            or not timestamps.is_monotonic_increasing
        ):
            raise ValueError("Candles inside each causal window must be chronological.")

        session_features = session.loc[:, feature_columns].to_numpy(
            dtype=np.float32,
            copy=True,
        )
        if not np.isfinite(session_features).all():
            raise ValueError("Causal-window features must be finite numbers.")

        current_values = pd.to_numeric(session["current_target"], errors="coerce")
        anticipated_values = pd.to_numeric(
            session["anticipated_target"], errors="coerce"
        )
        if (
            current_values.isna().any()
            or anticipated_values.isna().any()
            or not current_values.between(0, 2).all()
            or not anticipated_values.between(0, 2).all()
        ):
            raise ValueError("Causal-window targets must be regime class indices.")

        # The target universe may begin later than this representation's context. That lets a
        # 16-candle and 64-candle model score the exact same candle IDs instead of letting the
        # shorter model quietly receive easier opening-period examples.
        first_target_position = max(
            1 if allow_partial_history else window_length - 1,
            target_start_position if target_start_position is not None else 0,
            1,
        )
        if first_target_position >= len(session):
            continue

        target_positions = np.arange(first_target_position, len(session))
        if allow_partial_history:
            # Each row is right-aligned so the target candle remains the final position. The
            # separate mask—not the zero value—defines which history is real market data.
            windows = np.zeros(
                (len(target_positions), window_length, len(feature_columns)),
                dtype=np.float32,
            )
            validity = np.zeros((len(target_positions), window_length), dtype=bool)
            for window_position in range(window_length):
                source_positions = (
                    target_positions - (window_length - 1 - window_position)
                )
                valid_rows = source_positions >= 0
                windows[valid_rows, window_position] = session_features[
                    source_positions[valid_rows]
                ]
                validity[valid_rows, window_position] = True
            feature_window_batches.append(windows)
            valid_history_batches.append(validity)
        else:
            # NumPy creates all overlapping windows as one stride-based view. Selecting the
            # legal targets and concatenating once avoids per-sample Python copies.
            all_windows = np.lib.stride_tricks.sliding_window_view(
                session_features,
                window_shape=window_length,
                axis=0,
            ).transpose(0, 2, 1)
            window_positions = target_positions - window_length + 1
            feature_window_batches.append(all_windows[window_positions])
            valid_history_batches.append(
                np.ones((len(target_positions), window_length), dtype=bool)
            )
        current_targets.extend(current_values.iloc[target_positions].astype(int))
        anticipated_targets.extend(anticipated_values.iloc[target_positions].astype(int))
        previous_current_targets.extend(
            current_values.iloc[target_positions - 1].astype(int)
        )
        previous_anticipated_targets.extend(
            anticipated_values.iloc[target_positions - 1].astype(int)
        )
        source_session_index = session_index
        if "session_index" in session.columns:
            unique_session_indices = session["session_index"].drop_duplicates()
            if len(unique_session_indices) != 1:
                raise ValueError("Each causal-window input must contain one session index.")
            source_session_index = int(unique_session_indices.iloc[0])
        session_indices.extend([source_session_index] * len(target_positions))
        candle_positions.extend(target_positions.tolist())
        sample_session_dates.extend([session_date] * len(target_positions))
        candlestick_ids.extend(
            session["candlestick_id"].iloc[target_positions].astype(str)
        )
        sample_timestamps.extend(timestamps.iloc[target_positions])

    if not feature_window_batches:
        raise ValueError("No session contains enough candles for the causal window.")

    features = np.concatenate(feature_window_batches).astype(np.float32, copy=False)
    features.setflags(write=False)
    valid_history_mask = np.concatenate(valid_history_batches)
    valid_history_mask.setflags(write=False)
    return CausalWindowCorpus(
        features=features,
        current_targets=np.asarray(current_targets, dtype=np.int64),
        anticipated_targets=np.asarray(anticipated_targets, dtype=np.int64),
        previous_current_targets=np.asarray(previous_current_targets, dtype=np.int64),
        previous_anticipated_targets=np.asarray(
            previous_anticipated_targets, dtype=np.int64
        ),
        session_indices=np.asarray(session_indices, dtype=np.int64),
        candle_positions=np.asarray(candle_positions, dtype=np.int64),
        session_dates=tuple(sample_session_dates),
        candlestick_ids=tuple(candlestick_ids),
        timestamps=tuple(sample_timestamps),
        feature_columns=feature_columns,
        window_length=window_length,
        valid_history_mask=valid_history_mask,
    )
