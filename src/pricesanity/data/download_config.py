"""Canonical chunk policy shared by the CLI and Python download entry points."""

DEFAULT_CANDLE_CHUNK_YEARS = 1
DEFAULT_STATUS_CHUNK_MONTHS = 3
CONDITION_CHUNK_YEARS = 1


def status_chunk_config(years: int | None, months: int | None) -> tuple[int, int | None]:
    """Resolve omitted status settings while retaining explicit yearly requests.

    Args:
        years: Number of calendar years in each chunk.
        months: Number of calendar months in each chunk.

    Returns:
        Resolved yearly setting and optional monthly override.

    Raises:
        ValueError: If both partition units are supplied or yearly chunks are not positive.
    """

    # Conflicting units would make the saved partition and resume settings ambiguous.
    if years is not None and months is not None:
        raise ValueError("Choose status chunk months or years, not both")

    # An explicit yearly override takes precedence over the omitted quarterly default.
    if years is not None:
        # A nonpositive partition size cannot advance through the requested range.
        if years < 1:
            raise ValueError("Status chunk years must be positive")

        return years, None

    return 1, DEFAULT_STATUS_CHUNK_MONTHS if months is None else months
