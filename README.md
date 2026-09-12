# Price Sanity

An educational walkthrough toward a Transformer that identifies the current
market regime and regime changes in E-mini S&P 500 futures price action.
Currently implemented: Databento downloads, session validation, normalized candle
features, and a desktop annotation GUI. Model construction, training, and
inference are future work.

## Setup

Use Python 3.11 or newer:

```zsh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,gui]'
```

Downloads use `DATABENTO_API_KEY` from your environment or local `.env` file.
Preparation and annotation use local files and do not require network access.

## Download data

Price Sanity requests the volume-ranked continuous contract `ES.v.0`. Every
time-series request includes its start date and excludes its end date.
Estimate a request before downloading:

```zsh
pricesanity-estimate --start 2026-09-08 --end 2026-09-10
```

Download only after supplying an approved maximum cost. The example ceiling
below is illustrative; use a ceiling you approve after checking the estimate:

```zsh
pricesanity-download \
  --start 2026-09-08 --end 2026-09-10 \
  --max-cost-usd 0.01
```

The downloader rechecks the estimate before downloading and refuses existing
request directories. It saves candles, status events, and daily data conditions
under `data/raw/ES-v-0_2026-09-08_2026-09-10/`.

## Prepare sessions

Use the same configuration for preparation and annotation:

```zsh
pricesanity-prepare \
  --config configs/default.yaml \
  --candlesticks data/raw/ES-v-0_2026-09-08_2026-09-10/candlesticks.csv \
  --status data/raw/ES-v-0_2026-09-08_2026-09-10/status.csv \
  --conditions data/raw/ES-v-0_2026-09-08_2026-09-10/condition.json \
  --interim-output data/interim/ES-v-0_2026-09-08_2026-09-10_ohlc.parquet \
  --output data/processed/ES-v-0_2026-09-08_2026-09-10.parquet
```

The default pipeline filters to the configured New York intraday window,
combines one-minute candles into complete five-minute candles, and validates
exact timestamps against status-derived closing boundaries and daily quality.
Scheduled early closes retain their shorter sessions.

The first trustworthy session supplies a closing reference; it is not itself
training input. Every retained session needs a trustworthy predecessor. An
incomplete or degraded session breaks that chain, as does an observed date with
missing schedule evidence. The next trustworthy session restores the reference;
the session after it can become training input. Include enough earlier data to
provide this reference when choosing a download range.

Normalization uses only candles inside trustworthy scheduled sessions. Candles
after an early close cannot replace that session's closing reference. A calendar
gap alone is not classified as missing data: weekends and holidays may be valid
gaps. A day absent from price, status, and quality evidence cannot be identified
as a missing trading session by this pipeline alone.

For a candle with previous close `p`, the four dimensionless features are:

| Feature | Formula |
| --- | --- |
| `open_gap` | `(open - p) / p` |
| `body` | `(close - open) / p` |
| `high_from_close` | `(high - close) / p` |
| `low_from_close` | `(low - close) / p` |

At the first candle of an eligible session, `p` is the preceding trustworthy
session's final close. Within a session, it is the previous candle's close.
The features use the current candle's completed OHLC values, so they describe
information available after that candle closes. Timestamps and instrument names
are alignment metadata, not numeric model features.

The interim artifact holds real OHLC prices; the processed artifact holds
normalized features. Existing outputs require an explicit `--overwrite`.
Both artifacts are written to temporary locations before either destination is
replaced. If either Parquet write fails, existing outputs remain unchanged and
temporary files are cleaned up. Each final replacement is atomic individually;
the pair is not crash-atomic, so a failure during replacement can still leave a
partially published pair.

## Annotate

```zsh
pricesanity-annotate \
  --config configs/default.yaml \
  --candlesticks data/interim/ES-v-0_2026-09-08_2026-09-10_ohlc.parquet \
  --normalized data/processed/ES-v-0_2026-09-08_2026-09-10.parquet
```

Both filenames must contain the matching start and exclusive end dates, as in
the example. The GUI's date controls use inclusive dates. Choose a range and
click **Apply date range**; only prepared sessions in that range are navigated.

Click the chart to activate its shortcuts:

- **Left / Right:** move one candle, crossing session boundaries when needed.
- **1 / 2 / 3:** Bull / Bear / Range.
- Enter the **current regime** first, then the **anticipated regime**. The second
  key saves both choices and advances to the next candle.
- Revisit a candle to see its saved choices or replace them with another pair.
  Moving away after only the first key discards that incomplete edit.

Date fields retain their own arrow-key behavior when focused. The opening-gap
box displays the active candle's normalized `open_gap` as a signed percentage.
Annotations default to `data/annotations/pricesanity.sqlite3`; use `--database`
to select another file. A failed save stays on the candle and displays a retry
message. Press the anticipated-regime key again to retry.

The full session remains visible: this is **retrospective annotation**. Feature
normalization is causal, but the human annotator can see subsequent candles.
Both regime targets record your own interpretation; anticipated_regime is not
an automatically generated future-outcome label. Separate classification heads
are a proposed model design.

## Candle identifiers

New IDs include the candle interval, for example
`ES.v.0:5min:2026-09-09T13:30:00+00:00`. The GUI uses the configured
`target_interval`. Typed `RawCandlestick` and `NormalizedCandlestick` objects use
the same builder, and normalization preserves the interval. Equivalent durations
such as `300s` and `5min` have identical keys.

The `interval` field defaults to `5min`; pass `interval="1min"` for one-minute
candles. The `candlestick_id` property is a string. Parquet feature columns and
normalization formulas are unchanged.

## Implementation and checks

The GUI uses Qt's main event loop with no application-created worker threads.
Changing sessions builds the chart; moving within a session moves its existing
marker and schedules a repaint. Session row indices are built once. Each window
owns one SQLite connection, commits each completed label pair, and closes the
connection when the window closes. Standalone storage helpers close theirs after
each operation.

The launcher validates alignment at the file boundary; the window also validates
its required fields because it can be constructed directly from Python. These
boundary checks are intentional. Normalized Parquet loading selects only the
timestamp and opening-gap columns needed by the GUI.

Run the tests without opening desktop windows:

```zsh
QT_QPA_PLATFORM=offscreen python -m pytest -q
```

Raw CME data, prepared training data, and human annotations are excluded from Git.
Never force-add these files or model artifacts to the repository.
