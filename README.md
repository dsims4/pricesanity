# Price Sanity

An educational, leakage-safe time-series machine-learning benchmark built around
manually annotated E-mini S&P 500 market regimes. The original causal Transformer
remains important, but Price Sanity now compares classical, recurrent, convolutional,
and attention-based learners under one chronological protocol.

| Area | Status |
| --- | --- |
| Managed data download and preparation | Implemented and tested |
| Annotation application | Implemented; corpus still being completed |
| Original causal Transformer | Implemented with compatible commands/checkpoints |
| Classical benchmark adapters | Implemented and tiny-fit tested |
| TCN, GRU, Transformer benchmark adapters | Implemented and tiny-fit tested |
| Full 2,690-session benchmark | Not run |
| Final 500-session holdout | Untouched |

## Multi-model benchmark status

**Execution engine ready / awaiting completed annotation corpus.** Price Sanity now has an
immutable benchmark snapshot with physically separate development and holdout files, resumable
Optuna tuning, shared causal representations, chronological folds, a global development seal,
explicitly locked final evaluation, stage-coherent checksummed artifacts, and artifact-only
result views. A tiny synthetic six-family acceptance
test exercises this machinery. The real 2,690-session search and final holdout have not been run.
The [final preparation report](docs/final_preparation_report.md) records validation and the
remaining local accelerator caveat: CPU execution works; RX 7800 XT acceleration is unverified.

```text
Databento market data
        ↓
session validation and normalized OHLC geometry
        ↓
human current + anticipated regime annotations
        ↓
causal 16-candle sequential or identical flattened representations
        ↓
multiple model families
        ↓
chronological development tuning
        ↓
common metrics and untouched final holdout
        ↓
saved-artifact GUI, comparison view, and report notebooks
```

The controlled benchmark gives every algorithm the same 16 × 4 values. The separate
best-of-family benchmark permits appropriate causal representations. Both differ from the
existing production-style Transformer walk-forward evaluation. See
[`docs/benchmark_protocol.md`](docs/benchmark_protocol.md),
[`docs/leakage_rules.md`](docs/leakage_rules.md), and
[`docs/experiment_design.md`](docs/experiment_design.md).

Inspect the frozen 2,690-session plan, registered families, or local infrastructure profile:

```zsh
pricesanity-benchmark plan --session-count 2690
pricesanity-benchmark models
pricesanity-benchmark run --all-models --track controlled \
  --mode tuning --session-count 2690 --dry-run
pricesanity-benchmark-report
pricesanity-benchmark profile --synthetic
```

The execution lifecycle and launch commands are documented in
[`docs/benchmark_execution.md`](docs/benchmark_execution.md). Exact final access requires the
deliberate `--confirm-final-holdout` switch; a focused one-fold diagnostic can never freeze a
winner or unlock it.

Classical estimators and resumable Optuna searches are optional:

```zsh
python -m pip install -e '.[benchmark]'
```

The two report notebooks are templates until the full corpus and completed benchmark artifacts
exist. `pricesanity-compare` opens the saved-artifact leaderboard and shows an honest empty state
before that point.

Create a clean source-only review archive after committing the intended revision:

```zsh
git archive --format=zip --output=pricesanity-source.zip HEAD
```

This excludes ignored environments, market data, annotations, caches, and model output.

## Setup

Use Python 3.11 or newer. Install only the area being used:

```zsh
python3 -m venv .venv
source .venv/bin/activate

# Core download and preparation
python -m pip install -e .

# Annotation and result GUIs
python -m pip install -e '.[gui]'

# Existing Transformer training
python -m pip install -e '.[training]'

# Multi-model benchmark and Optuna
python -m pip install -e '.[benchmark]'

# Full development/research environment and complete test suite
python -m pip install -e '.[dev,gui,training,benchmark,notebooks]'
python -m pytest
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

The intended corpus policy is strict status-validated eligibility. This is a deliberate
research boundary, including for dates before authoritative scheduled status coverage.
The older conservative historical fallback is retired and must not be restored to expand
the corpus. Apparently complete OHLC prices alone cannot establish trading hours, and
early closes must never be guessed. Preserve the previous-session reference-chain rules
below when preparing or validating data.

Every eligible session requires a scheduled status-derived closing boundary,
an `available` dataset condition, and its complete configured candle grid.
Availability alone does not establish trading hours. Dates without scheduled
status evidence cannot become annotation input or supply a previous-session
close, regardless of how complete their prices appear. No historical calendar
fallback or hard-coded year cutoff is used.

An `available` dataset condition does not prove that this instrument's candles
exist. Condition records on configured trading weekdays remain date evidence
even if that date has no candles or scheduled status. Such a missing date breaks
the reference chain: Monday cannot borrow an older close when Friday is missing.
Available weekend metadata alone does not interrupt a valid Friday-to-Monday
reference. This is conservative around weekday closures: without sufficient
session evidence, the next complete day restores the reference rather than
becoming annotation-eligible immediately.

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

A full local audit validated 4,932,502 one-minute candles, 23,580 status records,
and 5,134 condition records. Independent source-minute aggregation reproduced all
286,967 complete five-minute candles. Strict session validation retained 214,791
candles across 2,690 annotation-eligible sessions; all four normalized features
matched independent calculations for every retained candle.
The first eligible session is November 23, 2015, using the complete, available
November 20 session's closing candle at 21:10 UTC (the interval ending at 21:15).
Earlier complete prices cannot supply references without scheduled status evidence.
The actual GUI percentage label was also checked for every retained candle in
all 2,690 sessions. The local audit record, including artifact checksums and each
session's reference date, is `data/processed/annotation_validation_audit.json`.

The first trustworthy scheduled session supplies a closing reference; it is not
itself training input. Every retained session needs a trustworthy predecessor.
An incomplete or degraded session, or an evidenced weekday without scheduled
boundaries, breaks that chain. The next trustworthy session restores the
reference; the session after it can become training input. Include enough earlier
data to provide this reference when choosing a download range.

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

The launcher requires matching session-validation metadata written by the current
preparation pipeline. It verifies each full scheduled candle grid, the retained
preceding-session close, and every candle's `open_gap` before opening the window.
Older exports without this evidence must be prepared again. This metadata lives
in the Parquet metadata rather than the model's numeric feature columns.
The GUI displays each checked candle gap as a signed percentage rounded to three
decimal places; selecting another date range does not recalculate its reference.

Both filenames must contain the matching start and exclusive end dates, as in
the example. The GUI's date controls use inclusive dates. Choose a range and
click **Apply date range**; only prepared sessions in that range are navigated.

The four buttons beside the date controls stay within that applied range:

- **Back one day / Forward one day:** open the adjacent valid session at its first candle.
- **Seek back / Seek forward:** jump to the nearest unannotated candle before or after
  the current candle, skipping saved judgments and crossing valid sessions as needed.

Seeking does not wrap around. If no matching candle remains, the selection stays
in place and the status bar explains why. Navigation returns keyboard focus to the chart.

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
learn the current and anticipated judgments from the same causal representation.

## Train

Install the optional PyTorch dependency and refresh the terminal commands after
changing the project metadata:

```zsh
python -m pip install -e '.[training]'
```

The first experiment uses the earliest 100 complete sessions for training, the
following 20 for validation, and the final 10 for one untouched test:

```zsh
pricesanity-train \
  --config configs/default.yaml \
  --normalized data/processed/ES-v-0_2010-06-06_2026-09-12.parquet \
  --database data/annotations/pricesanity.sqlite3
```

Single-run mode requires exactly 130 complete annotated sessions by default.
For longer history, use expanding-history walk-forward mode:

```zsh
pricesanity-train \
  --config configs/default.yaml \
  --normalized data/processed/ES-v-0_2010-06-06_2026-09-12.parquet \
  --database data/annotations/pricesanity.sqlite3 \
  --walk-forward --resume
```

The first run uses **100 train / 20 validation / 10 test** sessions. Training
keeps session one and expands by 120 sessions per later run. Validation and test
remain fixed at 20 and 10 sessions:

| Run | Train | Validation | Test |
| --- | --- | --- | --- |
| 1 | 1–100 | 101–120 | 121–130 |
| 2 | 1–220 | 221–240 | 241–250 |
| 3 | 1–340 | 341–360 | 361–370 |
| 4 | 1–460 | 461–480 | 481–490 |

The 120 added training sessions comprise the previous 20 validation sessions,
previous 10 test sessions, and 90 newly elapsed sessions. Old test sessions may
enter a later training set because they are then historical; their original
predictions and scores remain the record of what the earlier model knew.
The intervening 90 sessions are not a separate official test block in this schedule.

`--training-sessions` specifies the initial training count. For small smoke tests,
role counts remain configurable; growth is initial training plus validation count
(default 100 + 20). The misleading `--step-sessions` option has been removed and
old commands using it fail explicitly. Insufficient trailing history is reported
and never creates a partial validation/test experiment.

Only training sessions update weights and fit feature means and standard deviations.
Validation uses those frozen statistics and chooses the best epoch by validation loss.
After restoring that epoch's weights, test runs once and cannot influence scaling,
weight updates, or checkpoint selection. Every experiment starts a fresh model;
metadata records `initialization_mode: fresh`. There is no implicit warm-start.
Each expanding training pool fits its own statistics, frozen throughout that run's
validation and test. Later statistics never replace an earlier run's statistics.
Sessions shuffle during training; candles inside each session stay chronological.
Causal attention hides later candles. Early-close sessions retain their natural
length, with padding excluded from attention, losses, metrics, and saved predictions.

The normalized source and annotation snapshot load once per command. Overlapping
runs reuse raw CPU session tensors, but never fitted standardizers or model weights.
DataLoaders use zero worker processes. `--device auto` chooses CUDA if available,
then Apple MPS, then CPU; explicit `--device mps` fails if MPS is unavailable.
No mixed precision or automatic unsupported-operation fallback is enabled by this code.

New CLI training runs default to `--model-dimension 48` (previously 12), with
three attention heads, two layers, and a 192-dimensional feed-forward block.
The four normalized input features are unchanged. Width must be divisible by
three; `--model-dimension 12` reproduces the original capacity. Saved models retain
their own architecture. Compare capacity experiments in separate output directories;
changing width does not alter old predictions or guarantee better validation results.

Single-run output defaults to `data/models/run_001/`; `--checkpoint` can change its
checkpoint filename/location. Walk-forward output defaults to
`data/models/walk_forward/`, configurable with `--output-directory`:

```text
run_001/
    model.pt
    test_predictions.parquet
    run_metadata.json
run_002/
    ...
```

Checkpoints retain selected weights, architecture, train-only feature scaling,
training settings, epoch history, baseline scores, and final test measurements.
Predictions contain each real test candle's stable ID, timestamp, two predicted
regimes, class probabilities, and separately named human labels. Metadata records
boundaries, seed, source paths, model identity, and artifact checksums. Internal
artifact paths are relative, so the bundle can be moved while supplying its OHLC source.
Metadata also records actual session counts, real candle counts by partition and
in total, elapsed fitting/evaluation time, and both label distributions for train,
validation, and test. Distributions are recorded after evaluation and never change
loss weighting. Accuracy, macro-F1, confusion matrices, and majority-class baselines
remain in the checkpoint; a separate `results.json` is not required.
Metadata is published last and marks a completed run.

Use `--walk-forward --run-index 4` to target run four, or `--start-run 4` to run
from four onward. Add `--resume` to skip completed bundles only after checking
integrity and matching inputs/settings. An interrupted run without final metadata
is explicitly rebuilt from epoch one; optimizer-state continuation is not implemented.
A damaged completed bundle or changed labels/settings is refused. `--overwrite`
explicitly authorizes replacement. Version-one metadata paths and missing checksums
remain readable with warnings only when the underlying checkpoint already has the
fields required by the current loader. Older incompatible checkpoints are not migrated
implicitly and cannot be reused without the new input signature.
Starting a later run never rewrites an earlier checkpoint, predictions, metrics,
metadata, or checksums, even when the corpus grows and the old run count becomes
historical. Only explicit `--overwrite` permits replacing a selected run.
Use a new `--output-directory` for experiments created with the former rolling
protocol: those bundles remain viewable, but resume rejects their old signatures
rather than silently relabeling them as expanding-history runs.
Do not run simultaneous writers against the same output directory.

Later histories (100, 220, 340, 460, ...) require more training compute and retained
input memory. This is expected. Terminal output includes session/candle counts,
elapsed training time, run boundaries, train/validation epoch losses, best epoch,
test accuracy and macro-F1 for both heads, and artifact locations. These are measured
per run; this README makes no claim about trained predictive performance.

### Research limitations of this baseline

All historical training sessions are sampled equally, with no recency or class
reweighting. Existing candle-level loss and natural early-close lengths are retained.
Older market regimes may increasingly influence an expanding model; this remains
a research question. Future experiments can compare expanding history, recent
rolling history, and expanding history with recency weighting without changing
this first baseline automatically.

Runs share historical training data and are dependent experiments, not independent
replications. Twenty-session validation scores can be noisy, and class distributions
and fitted standardization can drift between runs. Recorded distributions, frozen
per-run statistics, and immutable results make those changes auditable; they do not
eliminate the modeling risks.

Repeated architecture or hyperparameter changes informed by historical test results
make those tests part of the researcher's effective model selection. For later
serious performance claims, keep a final untouched holdout period. This command
does not reserve, remove, or change any dataset period automatically.

### Controlled tuning while annotation continues

Use the validation-only diagnostic command for tuning. The ordinary training CLI
runs its official test pass automatically and should not be used for repeated
hyperparameter selection.

```zsh
python -m pricesanity.training.tuning \
  --normalized data/processed/ES-v-0_2010-06-06_2026-09-12.parquet \
  --database data/annotations/pricesanity.sqlite3 \
  --config configs/default.yaml \
  --output-directory data/models/tuning_pass_001 \
  --device cpu
```

The first stage reruns the unchanged baseline and checks memorization of five
sessions drawn only from training. If that diagnostic succeeds, add
`--resume --full-checklist` to continue controlled validation comparisons. The snapshot
contains the first run's 100 training and 20 validation sessions; the ten test
sessions are reserved and never evaluated by this command. A resumed pass reads
its frozen snapshot, so additional annotation work cannot change a comparison.
Start a new pass directory to incorporate corrected labels.

Every experiment records settings, seed, device, parameter count, elapsed time,
epoch losses, per-class precision/recall/F1, confusion matrices, class counts,
training-only majority baselines, and validation scores near human transitions
(exact candle and neighborhoods of one/two candles). Neighborhoods never cross
sessions. Diagnostic weights use a separate filename and are not official test
bundles for the GUI. Completed experiments are reused only when their checksums,
frozen-snapshot identity, settings, and execution environment match. A compatible
interrupted experiment restarts from epoch one; completed reports are not overwritten.
When adopting older diagnostic bundles without checksums, the loader first reproduces
their recorded scores and training statistics from the frozen snapshot. It then adds
an identity sidecar without rewriting the historical report or weights.
Resume with the same explicit device as the saved experiments. The existing
`tuning_pass_001` uses `--device cpu`; selecting `auto` on an MPS-capable host can
choose a different device and will correctly fail the environment identity check.

The ledger records both the raw validation winner and the practical selection. Mean
current/anticipated validation macro-F1 is the primary score; validation loss breaks
exact score ties. Within 0.005 of the raw winner, context sweeps prefer shorter history
and width/depth/feed-forward sweeps prefer fewer parameters. Other settings retain
the reference unless the gain reaches 0.005. For this pass, context 32 scored 0.590212
and context 16 scored 0.589932, so context 16 is the practical choice. Historical
context-32 width results remain valid diagnostics but do not drive the context-16 sweep.

Context experiments keep each candle's original session position and restrict its
complete receptive field to the trailing 16, 32, or 64 candles. This requires
separate overlapping windows and is more expensive than full-session attention.
A per-layer band mask would not enforce the same limit because deeper layers could
pass older information through intermediate candles. No production model or GUI
context behavior is changed by these diagnostic experiments.

Width sweeps use a matched four-head control so 12, 32, and 64 dimensions can be
compared without simultaneously changing head count. The feed-forward width stays
fixed during that sweep and is tested separately. Single-head losses are diagnostic
ablations; ordinary training retains its equal two-head loss. No class or recency
weighting is added automatically, and no chosen settings replace production defaults.
Repeated comparisons on 20 validation sessions can overfit validation itself; selected
settings remain provisional until confirmed on additional chronological data.

### Fixed training-size study

The separate `data/models/training_size_study_001/` experiment freezes 176 annotated
sessions. Its epoch-cap check uses only the original 100/20 tuning snapshot. Phase 2
then trains independent 100-, 120-, 140-, and 150-session prefixes for the frozen
number of epochs and compares them on the same sessions 151–176. Each model fits
its own training-only standardizer; the forward holdout never selects checkpoints.

Resume the already-initialized study with:

```zsh
python -m pricesanity.training.training_size_study data/models/training_size_study_001
```

This command reads the frozen study snapshot and recipe, reuses checksum-validated
completed candidates, and rebuilds matching incomplete candidates from epoch one.
It does not read newer annotations or change the completed tuning pass. Reports,
checkpoints, per-candle predictions, and checksums remain in the study directory.

### Inspect saved test results

Install the GUI and training extras, then open a completed run:

```zsh
pricesanity-test \
  --run data/models/walk_forward/run_001 \
  --candlesticks data/interim/ES-v-0_2010-06-06_2026-09-12_ohlc.parquet \
  --config configs/default.yaml
```

The OHLC path may be omitted when the recorded external source still exists.
The loader verifies checkpoint and prediction checksums, model/run identity,
configured instrument and timezone, OHLC geometry, and exact timestamp/ID alignment
for whole test sessions. It rejects overlapping subsets from wrong candle intervals.
Older absolute internal paths resolve to local bundle filenames for portability.

The GUI reads saved predictions; navigation performs no model inference and never
opens the annotation database. Left/right move between candles; previous/next session
buttons change days. Human labels are hidden until explicitly enabled and remain
separate from model predictions. Display times use the validated session timezone.

The chart underlines each predicted regime with a colored span (Bull, Bear, Range),
including the first regime of the session. Bars include text labels (Bu/Be/R on short spans), and each change candle has a
nearby label naming its new regime. A fixed legend explains the colors and abbreviations;
no dotted vertical lines are drawn. A span boundary belongs to candle `t`
exactly when its predicted current
regime differs from candle `t - 1`; its label names the new regime. Candle zero
establishes the displayed starting regime and has no change marker. Comparisons
restart each session. The full chart is retrospective, while stored model predictions
were produced with causal attention.

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
marker and schedules a repaint. Session row indices are built once. Each annotation window
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

### Preparation guides

See [WSL / AMD training](docs/wsl_amd_training.md), [GUI design](docs/gui_design.md), and
[benchmark lifecycle](docs/benchmark_execution.md). Finish development in both tracks, then
run `pricesanity-benchmark freeze-development` before any explicitly confirmed final evaluation.
The accepted corpus policy is strict status-validated eligibility; historical fallback is retired.
