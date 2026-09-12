import pandas as pd

from pricesanity.config import (
    AppConfig,
    DataConfig,
    NormalizationConfig,
    ProjectConfig,
    SessionConfig,
)
from pricesanity.data.pipeline import prepare_candlestick_sessions


def test_prepare_candlestick_sessions_preserves_opening_gap_reference() -> None:
    # Use one complete reference date followed by one complete eligible date so
    # the second opening candle can be checked against the first session's close.
    reference_timestamps = pd.date_range(
        "2026-01-05 09:30:00",
        "2026-01-05 16:15:00",
        freq="1min",
        inclusive="left",
        tz="America/New_York",
    )
    eligible_timestamps = pd.date_range(
        "2026-01-06 09:30:00",
        "2026-01-06 16:15:00",
        freq="1min",
        inclusive="left",
        tz="America/New_York",
    )

    # Keep each minute flat within its date so resampling remains simple and the
    # two-percent overnight opening gap is the only price change.
    candlestick_data = pd.DataFrame(
        {
            "ts_event": reference_timestamps.append(
                eligible_timestamps
            ).tz_convert("UTC"),
            "open": (
                [100.0] * len(reference_timestamps)
                + [102.0] * len(eligible_timestamps)
            ),
            "high": (
                [100.0] * len(reference_timestamps)
                + [102.0] * len(eligible_timestamps)
            ),
            "low": (
                [100.0] * len(reference_timestamps)
                + [102.0] * len(eligible_timestamps)
            ),
            "close": (
                [100.0] * len(reference_timestamps)
                + [102.0] * len(eligible_timestamps)
            ),
        }
    )

    # Include the scheduled open and close surrounding each RTH date so status
    # extraction can build two complete session schedules.
    status_data = pd.DataFrame(
        {
            "ts_event": [
                "2026-01-04 23:00:00Z",
                "2026-01-05 22:00:00Z",
                "2026-01-05 23:00:00Z",
                "2026-01-06 22:00:00Z",
            ],
            "reason": ["scheduled"] * 4,
            "trading_event": ["none"] * 4,
            "is_trading": ["Y", "N", "Y", "N"],
        }
    )

    # Mark both dates available so the first can establish a trustworthy close
    # and the second can become eligible model data.
    data_conditions = pd.DataFrame(
        {
            "date": ["2026-01-05", "2026-01-06"],
            "condition": ["available", "available"],
        }
    )

    # Keep every pipeline setting explicit so the integration test documents
    # the exact one-minute to five-minute RTH transformation.
    config = AppConfig(
        project=ProjectConfig(name="Price Sanity", random_seed=42),
        data=DataConfig(
            instrument="ES",
            source_timezone="UTC",
            session_timezone="America/New_York",
            timestamp_column="ts_event",
            source_interval="1min",
            target_interval="5min",
            require_complete_candlesticks=True,
        ),
        session=SessionConfig(
            start_time="09:30",
            end_time="16:15",
            trading_weekdays=(0, 1, 2, 3, 4),
        ),
        normalization=NormalizationConfig(scheme="relative_ohlc_v1"),
    )

    # Prepare the complete second date while retaining the first date only long
    # enough to normalize the eligible opening candle.
    prepared_candlestick_data = prepare_candlestick_sessions(
        candlestick_data,
        status_data,
        data_conditions,
        config=config,
    )

    # One RTH session contains eighty-one five-minute candles, and its first
    # normalized row must preserve the two-percent move from the prior close.
    assert len(prepared_candlestick_data) == 81
    assert prepared_candlestick_data.loc[0, "ts_event"] == pd.Timestamp(
        "2026-01-06 14:30:00Z"
    )
    assert prepared_candlestick_data.loc[0, "open_gap"] == 0.02
    assert prepared_candlestick_data["body"].eq(0.0).all()


def _prepare_reference_case(*, missing_middle_status=False, middle_rows=None,
                            middle_condition="available", early_close=False):
    from pricesanity.config import load_config
    from pricesanity.data.pipeline import prepare_candlestick_session_tables

    dates = ["2026-09-08", "2026-09-09", "2026-09-10"]
    prices = [100., 102., 104.]
    frames = []
    status_rows = []
    for position, (day, price) in enumerate(zip(dates, prices, strict=True)):
        times = pd.date_range(day + " 09:30", day + " 16:15", freq="1min",
                              inclusive="left", tz="America/New_York").tz_convert("UTC")
        frame = pd.DataFrame({"ts_event": times, "open": price, "high": price,
                              "low": price, "close": price})
        if early_close and position == 0:
            # Valid prices after the official close must never replace its 100 reference.
            frame.loc[frame.ts_event >= pd.Timestamp(day + "T17:15:00Z"),
                      ["open", "high", "low", "close"]] = 200.
        if position == 1 and middle_rows is not None:
            frame = frame.iloc[:middle_rows]
        frames.append(frame)
        if position == 1 and missing_middle_status:
            continue
        close = "17:15" if early_close and position == 0 else "21:00"
        status_rows.extend([
            {"ts_event": day + "T13:00:00Z", "reason": "scheduled", "trading_event": "none", "is_trading": "Y"},
            {"ts_event": day + "T" + close + ":00Z", "reason": "scheduled", "trading_event": "none", "is_trading": "N"},
        ])
    return prepare_candlestick_session_tables(
        pd.concat(frames, ignore_index=True), pd.DataFrame(status_rows),
        pd.DataFrame({"date": dates, "condition": ["available", middle_condition, "available"]}),
        config=load_config("configs/default.yaml"),
    )


def test_missing_status_breaks_reference_chain() -> None:
    result = _prepare_reference_case(missing_middle_status=True, middle_condition="degraded")
    assert result.ohlc.empty
    assert result.normalized.empty


def test_incomplete_bars_cannot_erase_missing_session_evidence() -> None:
    result = _prepare_reference_case(missing_middle_status=True, middle_rows=1)
    assert result.normalized.empty


def test_adverse_condition_breaks_chain_without_prices_or_status() -> None:
    result = _prepare_reference_case(missing_middle_status=True, middle_rows=0,
                                     middle_condition="degraded")
    assert result.normalized.empty


def test_opening_gap_uses_official_early_close() -> None:
    result = _prepare_reference_case(early_close=True)
    assert len(result.ohlc) == len(result.normalized) == 162
    assert result.normalized.iloc[0].open_gap == 0.02
    assert result.ohlc.ts_event.tolist() == result.normalized.ts_event.tolist()
