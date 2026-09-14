"""Join normalized candlesticks with human labels for model training."""

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd
import torch

from pricesanity.annotation.schema import CandlestickAnnotation, MarketRegime
from pricesanity.data.identifiers import build_candlestick_id


# Preserve one fixed feature order because tensor columns have no names after conversion.
FEATURE_COLUMNS = (
    "open_gap",
    "body",
    "high_from_close",
    "low_from_close",
)

# Convert categorical judgements into the class indices expected by cross-entropy loss.
REGIME_TO_CLASS = {
    MarketRegime.BULL: 0,
    MarketRegime.BEAR: 1,
    MarketRegime.RANGE: 2,
}

# Match PyTorch cross-entropy's default ignored class for padded target positions.
IGNORED_TARGET = -100


@dataclass(frozen=True)
class AnnotatedSessionSplit:
    """Chronological sessions assigned to one training experiment."""

    training: tuple[pd.DataFrame, ...]
    validation: tuple[pd.DataFrame, ...]
    test: tuple[pd.DataFrame, ...]


@dataclass(frozen=True)
class TensorSession:
    """One annotated session converted into model-ready tensors."""

    session_date: date
    candlestick_ids: tuple[str, ...]
    timestamps: tuple[pd.Timestamp, ...]
    features: torch.Tensor
    current_targets: torch.Tensor
    anticipated_targets: torch.Tensor


@dataclass(frozen=True)
class TensorBatch:
    """Variable-length sessions padded into one rectangular training batch."""

    session_dates: tuple[date, ...]
    candlestick_ids: tuple[tuple[str, ...], ...]
    timestamps: tuple[tuple[pd.Timestamp, ...], ...]
    lengths: torch.Tensor
    features: torch.Tensor
    current_targets: torch.Tensor
    anticipated_targets: torch.Tensor
    padding_mask: torch.Tensor


class TensorSessionDataset(torch.utils.data.Dataset):
    """Index complete tensor sessions without changing candle order."""

    def __init__(self, sessions: Sequence[TensorSession]) -> None:
        """Retain one immutable collection for repeated training epochs.

        Args:
            sessions: Complete sessions already converted into tensors.
        """

        # Copy membership into a tuple so a caller cannot reorder the dataset
        # after a DataLoader has begun sampling it.
        self._sessions = tuple(sessions)
        if not self._sessions:
            raise ValueError("A tensor session dataset cannot be empty.")

    def __len__(self) -> int:
        """Return the number of complete session examples."""

        return len(self._sessions)

    def __getitem__(self, index: int) -> TensorSession:
        """Return one natural-length session by position.

        Args:
            index: Zero-based session position selected by the DataLoader.

        Returns:
            The requested tensor session without shuffling its candles.
        """

        return self._sessions[index]


@dataclass(frozen=True)
class TrainingDataLoaders:
    """Session loaders for one chronological training experiment."""

    training: torch.utils.data.DataLoader
    validation: torch.utils.data.DataLoader
    test: torch.utils.data.DataLoader


def build_annotated_candlesticks(
    normalized_data: pd.DataFrame,
    annotations: Sequence[CandlestickAnnotation],
    *,
    timestamp_column: str,
    interval: str,
) -> pd.DataFrame:
    """Align normalized candle features with their two human targets.

    Args:
        normalized_data: Chronological normalized candlesticks.
        annotations: Complete human judgments loaded from the annotation store.
        timestamp_column: Column containing candle opening timestamps.
        interval: Fixed candle duration included in each stable identifier.

    Returns:
        Annotated normalized rows in their original chronological order.

    Raises:
        ValueError: If features or identities are invalid or annotations cannot be aligned.
    """

    # Every model row needs all four ordered features and enough identity data
    # to reconstruct the same key used by the annotation application.
    required_columns = {
        timestamp_column,
        "instrument",
        *FEATURE_COLUMNS,
    }
    missing_columns = required_columns.difference(normalized_data.columns)
    if missing_columns:
        raise ValueError(
            "Normalized data is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    # Work on a copy so training preparation cannot alter the reproducible
    # normalized artifact already held by its caller.
    aligned_data = normalized_data.copy()

    # Preparation evidence describes the full corpus and is not a model input. Carrying that
    # large metadata dictionary into every session slice would repeatedly copy it in pandas.
    aligned_data.attrs.clear()
    aligned_data[timestamp_column] = pd.to_datetime(
        aligned_data[timestamp_column],
        errors="coerce",
        utc=True,
    )
    if aligned_data[timestamp_column].isna().any():
        raise ValueError("Normalized data contains invalid timestamps.")
    if not aligned_data[timestamp_column].is_monotonic_increasing:
        raise ValueError("Normalized data must be chronological.")

    # Tensor conversion requires real finite values; coercion also prevents a
    # numeric-looking string from silently surviving as an object column.
    for feature_column in FEATURE_COLUMNS:
        aligned_data[feature_column] = pd.to_numeric(
            aligned_data[feature_column],
            errors="coerce",
        )
    if not np.isfinite(
        aligned_data.loc[:, FEATURE_COLUMNS].to_numpy(dtype=float)
    ).all():
        raise ValueError("Normalized features must be finite numbers.")

    # Recreate GUI-compatible identifiers from the immutable market identity
    # instead of assuming SQLite retrieval order matches normalized row order.
    aligned_data["candlestick_id"] = [
        build_candlestick_id(instrument, timestamp, interval)
        for instrument, timestamp in aligned_data[
            ["instrument", timestamp_column]
        ].itertuples(index=False, name=None)
    ]
    if aligned_data["candlestick_id"].duplicated().any():
        raise ValueError("Normalized candlestick identifiers must be unique.")

    # Encode both targets separately so neither human answer becomes a model
    # input and each future classification head receives its own target.
    annotation_rows = [
        {
            "candlestick_id": annotation.candlestick_id,
            "current_target": REGIME_TO_CLASS[annotation.current_regime],
            "anticipated_target": REGIME_TO_CLASS[annotation.anticipated_regime],
        }
        for annotation in annotations
    ]
    annotation_data = pd.DataFrame(
        annotation_rows,
        columns=("candlestick_id", "current_target", "anticipated_target"),
    )
    if annotation_data["candlestick_id"].duplicated().any():
        raise ValueError("Annotation candlestick identifiers must be unique.")

    # An annotation missing from the normalized corpus signals the wrong data,
    # instrument, or interval and must not silently disappear during the join.
    orphaned_annotation_ids = set(annotation_data["candlestick_id"]).difference(
        aligned_data["candlestick_id"]
    )
    if orphaned_annotation_ids:
        raise ValueError(
            f"{len(orphaned_annotation_ids)} annotations do not match normalized candles."
        )

    # Retain only labeled candles while preserving their normalized chronology;
    # one-to-one validation prevents an ambiguous target assignment.
    return aligned_data.merge(
        annotation_data,
        on="candlestick_id",
        how="inner",
        validate="one_to_one",
        sort=False,
    ).reset_index(drop=True)


def build_complete_annotated_sessions(
    normalized_data: pd.DataFrame,
    annotations: Sequence[CandlestickAnnotation],
    *,
    timestamp_column: str,
    interval: str,
    session_timezone: str,
) -> list[pd.DataFrame]:
    """Group aligned training rows into complete chronological sessions.

    Args:
        normalized_data: Complete chronological normalized candlestick corpus.
        annotations: Complete human judgments loaded from the annotation store.
        timestamp_column: Column containing candle opening timestamps.
        interval: Fixed candle duration included in each stable identifier.
        session_timezone: Timezone used to identify trading-session dates.

    Returns:
        Complete annotated sessions ordered by trading date.

    Raises:
        ValueError: If any started session is only partially annotated.
    """

    # Reuse the identity boundary so grouping cannot conceal a mismatched or
    # duplicated annotation behind an otherwise plausible session size.
    annotated_data = build_annotated_candlesticks(
        normalized_data,
        annotations,
        timestamp_column=timestamp_column,
        interval=interval,
    )

    # Derive dates from absolute timestamps in the exchange-local timezone so
    # UTC date changes cannot divide one RTH session or combine neighboring days.
    normalized_session_dates = pd.to_datetime(
        normalized_data[timestamp_column],
        errors="raise",
        utc=True,
    ).dt.tz_convert(session_timezone).dt.date
    annotated_data["session_date"] = annotated_data[
        timestamp_column
    ].dt.tz_convert(session_timezone).dt.date

    # Compare each started session with the complete prepared corpus. Sessions
    # with no annotations are future work; a partly labeled session is unsafe
    # to represent as one complete Transformer sequence.
    expected_session_sizes = normalized_session_dates.value_counts()
    annotated_session_sizes = annotated_data["session_date"].value_counts()
    incomplete_session_dates = [
        session_date
        for session_date, annotated_size in annotated_session_sizes.items()
        if annotated_size != expected_session_sizes.get(session_date, 0)
    ]
    if incomplete_session_dates:
        raise ValueError(
            f"{len(incomplete_session_dates)} sessions are only partially annotated."
        )

    # Each returned table preserves candle order while the outer list preserves
    # session order for the later chronological train/validation/test split.
    return [
        session_data.reset_index(drop=True)
        for _, session_data in annotated_data.groupby(
            "session_date",
            sort=True,
        )
    ]


def split_annotated_sessions(
    sessions: Sequence[pd.DataFrame],
    *,
    training_session_count: int,
    validation_session_count: int,
    test_session_count: int,
) -> AnnotatedSessionSplit:
    """Assign complete sessions to chronological experiment partitions.

    Args:
        sessions: Complete annotated sessions in chronological order.
        training_session_count: Earliest sessions used to fit model weights.
        validation_session_count: Following sessions used to select training settings.
        test_session_count: Final unseen sessions used only for evaluation.

    Returns:
        Three nonoverlapping chronological session partitions.

    Raises:
        ValueError: If counts or supplied session ordering cannot define the requested split.
    """

    split_counts = (
        training_session_count,
        validation_session_count,
        test_session_count,
    )

    # Every partition needs at least one full session; accepting zero or a
    # negative count could create an experiment without meaningful evaluation.
    if any(session_count <= 0 for session_count in split_counts):
        raise ValueError("Session split counts must all be positive.")

    requested_session_count = sum(split_counts)

    # Require an exact accounting so no annotated session is silently ignored
    # or assumed to belong to more than one experiment partition.
    if len(sessions) != requested_session_count:
        raise ValueError(
            f"The split requests {requested_session_count} sessions, "
            f"but {len(sessions)} were supplied."
        )

    session_dates = []
    for session in sessions:
        if session.empty or "session_date" not in session.columns:
            raise ValueError("Every split member must be a nonempty annotated session.")

        unique_session_dates = session["session_date"].drop_duplicates()
        if len(unique_session_dates) != 1:
            raise ValueError("Each split member must contain exactly one session date.")
        session_dates.append(unique_session_dates.iloc[0])

    # Chronology is the leakage boundary: model fitting must never receive a
    # session occurring after validation or test data.
    if session_dates != sorted(session_dates) or len(session_dates) != len(
        set(session_dates)
    ):
        raise ValueError("Annotated sessions must have unique chronological dates.")

    validation_start = training_session_count
    test_start = validation_start + validation_session_count

    # Tuples make the membership of a completed split explicit while retaining
    # each DataFrame for later feature and target tensor conversion.
    return AnnotatedSessionSplit(
        training=tuple(sessions[:validation_start]),
        validation=tuple(sessions[validation_start:test_start]),
        test=tuple(sessions[test_start:]),
    )


def convert_session_to_tensors(
    session: pd.DataFrame,
    *,
    timestamp_column: str,
) -> TensorSession:
    """Convert one completed annotated session into model-ready tensors.

    Args:
        session: One chronological session from the annotated-session builder.
        timestamp_column: Column containing candle opening timestamps.

    Returns:
        Features, targets, and audit identifiers for one session.

    Raises:
        ValueError: If the session cannot form one aligned tensor sequence.
    """

    # An empty table cannot produce a sequence or identify a trading date.
    if session.empty:
        raise ValueError("An annotated session cannot be empty.")

    # Tensor rows require features and targets, while identifiers and timestamps
    # preserve the ability to audit every resulting prediction.
    required_columns = {
        "session_date",
        "candlestick_id",
        timestamp_column,
        *FEATURE_COLUMNS,
        "current_target",
        "anticipated_target",
    }
    missing_columns = required_columns.difference(session.columns)
    if missing_columns:
        raise ValueError(
            "Annotated session is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    # One tensor sequence must represent exactly one trading session.
    unique_session_dates = session["session_date"].drop_duplicates()
    if len(unique_session_dates) != 1:
        raise ValueError(
            "A tensor session must contain exactly one session date."
        )

    # Preserve absolute UTC timestamps and require the same chronological order
    # used when the annotations were aligned.
    timestamps = pd.to_datetime(
        session[timestamp_column],
        errors="coerce",
        utc=True,
    )
    if timestamps.isna().any():
        raise ValueError("Tensor session contains invalid timestamps.")
    if timestamps.duplicated().any() or not timestamps.is_monotonic_increasing:
        raise ValueError(
            "Tensor session timestamps must be unique and chronological."
        )

    # Every tensor row must retain one unique key for later prediction audits.
    candlestick_ids = session["candlestick_id"]
    has_blank_id = (
        candlestick_ids.isna()
        | candlestick_ids.astype(str).str.strip().eq("")
    ).any()
    if has_blank_id or candlestick_ids.duplicated().any():
        raise ValueError(
            "Tensor session candlestick identifiers must be unique and nonempty."
        )

    # Copy the four features in their declared order before PyTorch shares the
    # NumPy storage with the resulting floating-point tensor.
    feature_array = session.loc[
        :,
        list(FEATURE_COLUMNS),
    ].to_numpy(
        dtype=np.float32,
        copy=True,
    )
    if not np.isfinite(feature_array).all():
        raise ValueError("Tensor session features must be finite numbers.")

    # Cross-entropy accepts integer class indices rather than one-hot vectors.
    valid_targets = set(REGIME_TO_CLASS.values())
    current_targets = pd.to_numeric(
        session["current_target"],
        errors="coerce",
    )
    anticipated_targets = pd.to_numeric(
        session["anticipated_target"],
        errors="coerce",
    )
    if (
        not current_targets.isin(valid_targets).all()
        or not anticipated_targets.isin(valid_targets).all()
    ):
        raise ValueError("Tensor session targets must be valid regime classes.")

    current_target_array = current_targets.to_numpy(
        dtype=np.int64,
        copy=True,
    )
    anticipated_target_array = anticipated_targets.to_numpy(
        dtype=np.int64,
        copy=True,
    )

    # Keep audit metadata outside the model inputs while preserving exact row alignment.
    return TensorSession(
        session_date=pd.Timestamp(unique_session_dates.iloc[0]).date(),
        candlestick_ids=tuple(candlestick_ids.astype(str)),
        timestamps=tuple(timestamps),
        features=torch.from_numpy(feature_array),
        current_targets=torch.from_numpy(current_target_array),
        anticipated_targets=torch.from_numpy(anticipated_target_array),
    )


def collate_tensor_sessions(
    sessions: Sequence[TensorSession],
) -> TensorBatch:
    """Pad natural-length sessions into one model-ready batch.

    Args:
        sessions: Nonempty collection of converted annotated sessions.

    Returns:
        Padded features and targets with real lengths and audit metadata.

    Raises:
        ValueError: If a session has inconsistent tensor or metadata dimensions.
    """

    # A batch without a session cannot establish its maximum sequence length.
    if not sessions:
        raise ValueError("A tensor batch requires at least one session.")

    session_lengths = []
    for session in sessions:
        session_length = session.features.shape[0]
        has_valid_feature_shape = (
            session.features.ndim == 2
            and session.features.shape[1] == len(FEATURE_COLUMNS)
        )
        has_aligned_targets = (
            session.current_targets.shape == (session_length,)
            and session.anticipated_targets.shape == (session_length,)
        )
        has_aligned_metadata = (
            len(session.candlestick_ids) == session_length
            and len(session.timestamps) == session_length
        )
        if (
            session_length == 0
            or not has_valid_feature_shape
            or not has_aligned_targets
            or not has_aligned_metadata
        ):
            raise ValueError(
                "Every tensor session must have aligned features, targets, and metadata."
            )
        if session.features.dtype is not torch.float32:
            raise ValueError("Tensor session features must use torch.float32.")
        if (
            session.current_targets.dtype is not torch.int64
            or session.anticipated_targets.dtype is not torch.int64
        ):
            raise ValueError("Tensor session targets must use torch.int64.")

        session_lengths.append(session_length)

    # Pad only inside this batch so each stored TensorSession retains its real
    # exchange-session length outside model execution.
    padded_features = torch.nn.utils.rnn.pad_sequence(
        [session.features for session in sessions],
        batch_first=True,
        padding_value=0.0,
    )
    padded_current_targets = torch.nn.utils.rnn.pad_sequence(
        [session.current_targets for session in sessions],
        batch_first=True,
        padding_value=IGNORED_TARGET,
    )
    padded_anticipated_targets = torch.nn.utils.rnn.pad_sequence(
        [session.anticipated_targets for session in sessions],
        batch_first=True,
        padding_value=IGNORED_TARGET,
    )

    lengths = torch.tensor(session_lengths, dtype=torch.int64)
    candle_positions = torch.arange(padded_features.shape[1]).unsqueeze(0)

    # PyTorch masks True positions, so only artificial values beyond each
    # session's real length are hidden from attention.
    padding_mask = candle_positions >= lengths.unsqueeze(1)

    return TensorBatch(
        session_dates=tuple(session.session_date for session in sessions),
        candlestick_ids=tuple(session.candlestick_ids for session in sessions),
        timestamps=tuple(session.timestamps for session in sessions),
        lengths=lengths,
        features=padded_features,
        current_targets=padded_current_targets,
        anticipated_targets=padded_anticipated_targets,
        padding_mask=padding_mask,
    )


def create_training_data_loaders(
    split: AnnotatedSessionSplit,
    *,
    timestamp_column: str,
    batch_size: int,
    random_seed: int,
    tensor_session_cache: dict[int, TensorSession] | None = None,
) -> TrainingDataLoaders:
    """Convert a chronological split into reproducible session DataLoaders.

    Args:
        split: Nonoverlapping annotated training, validation, and test sessions.
        timestamp_column: Column containing candle opening timestamps.
        batch_size: Maximum number of complete sessions in one batch.
        random_seed: Seed controlling training-session shuffle order.
        tensor_session_cache: Optional invocation-local CPU tensors keyed by session identity.
            The caller must keep the original DataFrames alive and unchanged throughout reuse.

    Returns:
        Shuffled training and chronological evaluation loaders.

    Raises:
        ValueError: If the batch size is not positive or a partition is empty.
    """

    if batch_size <= 0:
        raise ValueError("Training batch size must be positive.")

    def tensor_dataset(sessions: Sequence[pd.DataFrame]) -> TensorSessionDataset:
        """Reuse only raw tensors; each run still fits its own training-only standardizer."""

        converted_sessions = []
        for session in sessions:
            session_identity = id(session)
            converted_session = (
                tensor_session_cache.get(session_identity)
                if tensor_session_cache is not None
                else None
            )

            # Overlapping windows repeatedly use these same immutable frames. Conversion does
            # not fit anything, so reuse cannot leak future statistics into earlier runs.
            if converted_session is None:
                converted_session = convert_session_to_tensors(
                    session, timestamp_column=timestamp_column
                )
                if tensor_session_cache is not None:
                    tensor_session_cache[session_identity] = converted_session

            converted_sessions.append(converted_session)

        return TensorSessionDataset(converted_sessions)

    # Membership stays separate even when two experiments reuse the same historical tensors.
    training_dataset = tensor_dataset(split.training)
    validation_dataset = tensor_dataset(split.validation)
    test_dataset = tensor_dataset(split.test)

    # A dedicated generator makes training order repeatable without altering
    # NumPy, application, or model random state elsewhere in the project.
    training_generator = torch.Generator()
    training_generator.manual_seed(random_seed)

    # These small in-memory sessions do not justify worker startup, interprocess copies, or
    # macOS spawn overhead. Keep collation in the training process on CPU before device upload.
    training_loader = torch.utils.data.DataLoader(
        training_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_tensor_sessions,
        num_workers=0,
        generator=training_generator,
    )
    validation_loader = torch.utils.data.DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_tensor_sessions,
        num_workers=0,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_tensor_sessions,
        num_workers=0,
    )

    return TrainingDataLoaders(
        training=training_loader,
        validation=validation_loader,
        test=test_loader,
    )
