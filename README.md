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

## Managed downloads and resume

Price Sanity requests `ES.v.0` from `GLBX.MDP3` using `continuous` input
symbology. All time-series chunks include their start and exclude their end.
OHLC requests default to calendar-year chunks, including partial first and final
years. Status requests default to calendar-quarter chunks. Conditions use an
independent calendar-year partition; their API's inclusive ending date is translated to one day
before the chunk's exclusive end.

Inspect a request's remaining estimate without downloading or writing checkpoints:

```zsh
pricesanity-download \
  --start 2026-09-08 --end 2026-09-10 \
  --max-cost-usd 0 --estimate-only
```

After approving a ceiling, omit `--estimate-only` and supply that ceiling.
`pricesanity-estimate` remains available for a standalone full-range estimate.
The download command always recalculates remaining cost before paid requests.

Managed requests have this layout beneath `data/raw/ES-v-0_START_END/`:

```text
manifest.json
download.log                 # full warning and exception diagnostics
.download.lock               # OS lock prevents simultaneous writers
chunks/
    candlesticks/START_END.csv
    status/START_END.csv
    conditions/START_END.json
candlesticks.csv
status.csv
condition.json
```

In-progress chunk files have `.partial.csv` or `.partial.json` suffixes.
This is chunk-level checkpointing: the SDK finishes `get_range()` in memory
before the chunk is written to CSV. It does not persist network records while
the response is arriving. An interrupted response must retry that chunk.
A chunk is flushed, validated, checksummed, and renamed before its completion
checkpoint is published. Manifest writes use temporary files and atomic
replacement. Failed partial files never count as complete.

The manifest records request identity, chunk settings and ranges, pending/partial/
complete states, row counts, file sizes, SHA-256 checksums, timestamp bounds,
creation/update times, the original estimate, and the last failure. Each component
tracks its contiguous last successful exclusive boundary. A component is complete
when its chunks and final artifact have been published.

Run the same command to resume. Valid completed chunks are checked and skipped;
missing or corrupted chunks are requested again. A fully completed managed
request makes no further vendor calls through the download runner. Conflicting
identities/settings and unmanaged existing directories are refused. Keep the
same chunk-size options on resume. OHLC defaults to
`--candlestick-chunk-years 1`; status defaults to
`--status-chunk-months 3`.

Use `--status-chunk-months 1` for monthly status chunks or
`--status-chunk-years 1` for annual status chunks. Conditions and OHLC keep
their independent yearly boundaries. Quarterly status requests succeeded
when annual streams repeatedly ended prematurely during the full-corpus repair.
Choose either status months or status years and keep that setting on resume.
Changing it on an existing manifest is refused to protect completed chunks.
Previously checkpointed conditions retain their validated partition on resume;
the independent yearly policy applies to new requests.

`--max-cost-usd` is the maximum **additional estimated cost for this invocation**.
The remaining estimate includes only missing candle and status requests.
Conditions use metadata calls. Completed chunks are excluded from remaining cost.
Every attempted paid retry reserves another chunk estimate, so an exact ceiling
may prevent a retry even when `--retries` allows it. Actual charges are controlled
by Databento; these are estimates, not a billing guarantee.
Failed metadata probes consume a retry attempt but no additional paid estimate.
Extra cost is reserved immediately before another `get_range()` call begins.

The default `--retries 2` permits two retries after the first attempt, with 1- and
2-second backoffs. Retry classification is conservative: interrupted streams,
connection failures/timeouts, and selected temporary HTTP statuses. Authentication,
entitlement, parameter, symbol, schema and certificate errors are not blindly
retried. Downloads are synchronous, with no worker pool.

Final assembly is local and repeatable. CSV headers must match, appear once, and
records must be chronological and free of exact duplicates. Distinct simultaneous
status records remain valid. For status schemas exposing `ts_recv`, download
boundary/order validation follows that SDK request clock; event timestamps remain
available to session preparation. Conditions retain their record structure and
must have unique chronological dates. Final files are validated before atomic
replacement. An assembly failure preserves all completed chunks for local retry.

Interactive terminals redraw one cumulative dashboard showing actual completed
chunks, percentages, committed rows, the contiguous exclusive boundary, retries,
and warning/error totals. Redirected output and dumb terminals use plain text
at component changes, retries, and completion. Final summaries list at most three
missing ranges per component instead of every successful range.
Counters are checkpointed and cumulative across resumes. Full warning text,
warning classes, exception tracebacks and request context go to `download.log`.
Retried exceptions get one entry per failed attempt; the final exception is
logged once. Failures return nonzero and print one short explanation.
Read-only estimate warnings and preflight failures use the separate
`data/raw/download-errors.log`; they do not mutate the managed request directory.
API keys, authorization headers and environment contents
must never be logged. Ctrl-C preserves checkpoints and returns status 130.

All mutating download and repair paths acquire the same nonblocking OS lock,
including repair estimate-only adoption. A losing writer exits without touching
the managed manifest, log, chunks or final files. `.download.lock` can remain
after completion: lock ownership comes from `flock`, not file existence. Normal
estimate-only operations do not acquire the writer lock or checkpoint state.

The advisory lock currently supports macOS/Linux. Atomic file replacement does
not provide a single transaction across all final artifacts or guarantee
survival of every storage-device/power failure. A completely missing trading day
still requires evidence beyond absence alone to distinguish it from a holiday.

## Repair the existing interrupted download

`pricesanity-download` starts a new managed request or resumes its manifest.
`pricesanity-download --repair` is reserved for an interrupted, manifest-less
request directory that already contains `candlesticks.csv`. Repair validates
the existing rows, separates them into the configured OHLC ranges, and compares
each range with Databento's metadata count. Complete ranges are adopted; only
incomplete ranges are requested again. A valid existing `status.csv` is adopted
the same way. After missing ranges finish, all ranges are merged chronologically
and atomically replace the original final CSVs.

Each completeness count is probed once and immediately stored in the manifest.
A resumed repair reuses those counts and adopted ranges. It never guesses where
a missing record belongs, so a range whose count disagrees is downloaded again
as one unit. Structural corruption, invalid OHLC geometry, duplicate timestamps,
or records outside the original request stop repair and are logged.

For the current 2010–2026 corpus, create the repair checkpoint and preview the
remaining paid estimate with:

```zsh
pricesanity-download --repair \
  --start 2010-06-06 --end 2026-09-12 \
  --max-cost-usd 0 --estimate-only
```

After explicit approval, run the same command without `--estimate-only` and
replace `0` with the displayed additional ceiling. Repair estimate-only mode
makes metadata calls and writes adoption checkpoints under the writer lock, but makes no paid
time-series request. New repairs use yearly OHLC chunks and quarterly status
chunks unless explicitly changed. Harmless `.DS_Store` files are ignored;
unrelated files block adoption. Conditions are retrieved from metadata and
replace an earlier condition file only after successful validation.

Once a managed repair is checkpointed, resume it with `pricesanity-download --repair`
and the same chunk settings. Without `--repair`, the command starts or resumes
ordinary managed downloads and refuses unmanaged directories or repair manifests.
The explicit flag prevents accidental adoption of unrelated files.
Repair keeps chunks while work is incomplete, then removes them only
after every merged final file has passed validation. Its manifest retains the
range counts and final checksums needed to verify that completed repair later.

## Prepare sessions

Use the same configuration for preparation and annotation:

```zsh
pricesanity-prepare \
  --config configs/default.yaml \
  --candlesticks data/raw/ES-v-0_2010-06-06_2026-09-12/candlesticks.csv \
  --status data/raw/ES-v-0_2010-06-06_2026-09-12/status.csv \
  --conditions data/raw/ES-v-0_2010-06-06_2026-09-12/condition.json \
  --interim-output data/interim/ES-v-0_2010-06-06_2026-09-12_ohlc.parquet \
  --output data/processed/ES-v-0_2010-06-06_2026-09-12.parquet \
  --csv-chunk-rows 100000
```

Preparation reads the source CSV in pandas chunks (`--csv-chunk-rows`, default
100,000), selecting only timestamps and OHLC. Volume is never used. It carries
the unfinished five-minute aggregation group across reads, so read boundaries
inside candles or sessions do not change the output. Status and conditions are
loaded in memory, then the reduced five-minute corpus is validated and normalized
globally to preserve the cross-session trust chain. The float features remain
compact Parquet columns. Memory therefore scales with the smaller five-minute
corpus and the status evidence, not the full one-minute CSV.

The default pipeline filters to the configured New York intraday window,
combines one-minute candles into complete five-minute candles, and validates
exact timestamps against status-derived closing boundaries and daily quality.
Scheduled early closes retain their shorter sessions.

The older CME status feed does not contain the scheduled transitions needed to
construct normal sessions. Price Sanity derives the coverage boundary from the
first session successfully built from scheduled status data; it does not use a
hard-coded year. For observed OHLC dates strictly before that boundary, an
`available` Databento condition permits a conservative fallback using the
configured normal RTH open and close. The exact full five-minute grid is still
required. Degraded, unavailable, unknown, or missing condition evidence remains
untrustworthy and breaks the normal reference chain.

An `available` dataset condition does not prove that this instrument's candles
exist. Condition records on configured trading weekdays remain date evidence
even if that date has no candles or scheduled status. Such a missing date breaks
the reference chain: Monday cannot borrow an older close when Friday is missing.
Available weekend metadata alone does not interrupt a valid Friday-to-Monday
reference. This is conservative around weekday closures: without sufficient
session evidence, the next complete day restores the reference rather than
becoming annotation-eligible immediately.

Fallback stops on the first status-derived session date. Missing status evidence
on that date or any later date is therefore rejected as before. Historical early
closes are not guessed: without an authoritative scheduled close, their shorter
candle grid fails the configured normal-session completeness check. After the
coverage boundary, scheduled early closes continue to use their authoritative
status-derived boundaries.

The boundary is inferred from the first usable status-derived session present in
the supplied status corpus. The complete 2010–2026 corpus contains the historical
transition into this coverage. An arbitrary standalone subset beginning later
could infer a later boundary if the true first authoritative records were absent.

The local `ES.v.0` corpus starts in June 2010, but its first scheduled trading
transition is November 19, 2015, and its first scheduled RTH session is
November 20, 2015. This is a historical data limitation: Databento documents
that normal scheduled changes did not produce status messages before November
2015 in the older CME feed. See the
[MDP 2 status notes](https://databento.com/docs/knowledge-base/datasets).
Before the first usable normal scheduled transition, 39 local records have
nonscheduled reason codes 2, 3, or 5. Two additional reason-1 records carry
nonzero trading-event codes and therefore also do not qualify as normal scheduled
session boundaries. Accepting these records as daily boundaries would
misinterpret surveillance interventions, market events, or instrument expirations.
Changing the timestamp clock cannot supply the missing schedule evidence.
The existing scheduled-transition filter and session trust rules are retained.

A fresh local rebuild found 4,932,502 one-minute candles, 23,580 status records,
and 5,134 condition records. With `configs/default.yaml`, preparation produced
219,246 five-minute rows across 2,745 eligible sessions, with exact row-for-row
timestamp alignment between the OHLC and normalized tables. The first
authoritative status-derived session is November 20, 2015. Historical fallback
makes June 7, 2010 a trustworthy reference-only session, but missing intervening
weekdays prevent June 14 from using its close. After correcting that gap check,
March 15, 2011 is the first annotation-eligible session, using March 14's close.
Reference-only sessions need not appear in the annotation files; their closing
prices are retained during preparation to normalize the next eligible opening.

The first trustworthy session supplies a closing reference; it is not itself
training input. Every retained session needs a trustworthy predecessor. An
incomplete or degraded session breaks that chain. In the authoritative status
era, an observed date with missing schedule evidence also breaks it. Before that
coverage begins, missing scheduled-status messages are expected and the
conservative historical fallback applies. The next trustworthy session restores
the reference; the session after it can become training input. Include enough
earlier data to provide this reference when choosing a download range.

Normalization uses only candles inside trustworthy validated sessions. Candles
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
  --candlesticks data/interim/ES-v-0_2010-06-06_2026-09-12_ohlc.parquet \
  --normalized data/processed/ES-v-0_2010-06-06_2026-09-12.parquet
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

The launcher validates exact row order, instrument identity, configured candle
spacing, OHLC geometry, and opening-gap alignment at the file boundary. The
window also validates its required fields because it can be constructed directly
from Python. These boundary checks are intentional. Normalized Parquet loading
selects only the timestamp, instrument, and opening-gap columns needed by the GUI.

Run the tests without opening desktop windows:

```zsh
QT_QPA_PLATFORM=offscreen python -m pytest -q
```

Raw CME data, prepared training data, and human annotations are excluded from Git.
Never commit or force-add market data, prepared data, annotations, model artifacts,
API keys, manifests, logs or temporary market-data files. Keep the existing
`.gitignore` protections. Tests use fakes; a suite-wide guard rejects Python
network connection attempts.

There is no custom archive/export script. For a source-only archive of the
committed revision, use `git archive --format=zip --output=/tmp/pricesanity-source.zip HEAD`.
This excludes untracked and ignored workspace files, but also excludes pending
edits. Zipping the whole working directory does not honor `.gitignore`.
Generated Parquet, SQLite databases and sidecars, common model files, macOS
archive metadata, caches, environments, and build output are ignored; source
files already tracked by Git are retained.
