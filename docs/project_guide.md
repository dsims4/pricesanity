# Price Sanity project guide

This guide is the authoritative technical reference for the current Price Sanity repository.
The [README](../README.md) is the shorter entry point; specialized documents linked at the end
cover the research protocol and artifact contracts in greater depth.

## Purpose and boundaries

Price Sanity asks a supervised-learning question: how well can different causal classifiers
reproduce one person's candle-by-candle interpretation of ES price action? It does not optimize
returns, issue trade instructions, or infer economic causality.

One completed five-minute candle has two targets:

- `current_regime`: Bull, Bear, or Range after the candle;
- `anticipated_regime`: Bull, Bear, or Range the candle is most likely to produce next.

Both targets are judgments stored in SQLite and aligned to stable candlestick IDs. They are never
included among model market-data inputs. Retrospective annotation may use the completed chart, but
model examples remain causal and end at their target candle.

## Command matrix

| Command | Access | Primary product | Appropriate stage |
| --- | --- | --- | --- |
| `pricesanity-estimate` | reads Databento metadata | terminal cost estimate | any time before download |
| `pricesanity-download` | reads Databento; writes raw data | managed CSV/JSON corpus and manifest | data acquisition |
| `pricesanity-prepare` | reads raw data; writes derived data | aligned OHLC and normalized Parquet | after a complete or repaired download |
| `pricesanity-annotate` | reads Parquet; mutates SQLite | atomic candle label pairs | while building the corpus |
| `pricesanity-train` | reads labels/features; writes model runs | standalone Transformer artifacts | after enough complete annotations exist |
| `pricesanity-test` | read-only | saved-run Qt view | after a Transformer run |
| `pricesanity-benchmark` | varies by subcommand | diagnostics, snapshot, selections, and runs | diagnostics at any time; studies only at their guarded stage |
| `pricesanity-benchmark-report` | read-only | terminal leaderboard | any time; empty before final runs |
| `pricesanity-compare` | read-only | Benchmark Explorer | any time; empty before final runs |

## Installation and environments

The package metadata requires Python 3.11 or newer; the `training` extra requires PyTorch >=2.4.
A project-local virtual environment keeps Qt, PyTorch, and benchmark dependencies isolated.
For accelerator execution, first choose the platform-specific PyTorch binary as described in
[accelerator setup](accelerator_setup.md), then install the required extras:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev,gui,training,benchmark]'
```

Core installation provides NumPy, pandas, PyArrow, PyYAML, python-dotenv, and the pinned-compatible
Databento SDK series. Optional extras are:

| Extra | Packages and use |
| --- | --- |
| `dev` | pytest, JupyterLab, Matplotlib |
| `gui` | PySide6 and Matplotlib |
| `training` | PyTorch |
| `benchmark` | scikit-learn, Optuna, threadpoolctl, psutil |
| `notebooks` | JupyterLab and Matplotlib only |

### Device selection

Sequence models accept `cpu`, Apple Silicon `mps`, or `cuda` for compatible NVIDIA CUDA and
AMD ROCm builds. Classical estimators stay on CPU. The standalone Transformer and synthetic
smoke helper accept `auto`; benchmark study commands default to CPU. Verify the selected
environment before launching a study:

```bash
pricesanity-benchmark hardware
pricesanity-benchmark device-smoke --device cpu
pricesanity-benchmark device-smoke --device cuda --compare
```

Use `--device mps --compare` for Apple Silicon. Installation instructions, build interpretation,
WSL filesystem guidance, and tested-environment limitations live in
[accelerator setup](accelerator_setup.md); saved hardware evidence describes each actual run.

## Configuration

`configs/default.yaml` describes ES (`ES.v.0`), UTC source timestamps, New York session dates,
one-minute source candles, five-minute target candles, a maximum 09:30–16:15 America/New_York RTH
window, and the `relative_ohlc_v1` feature scheme.

`configs/benchmark/default.yaml` describes the full-corpus split, validation folds, learning-curve
prefixes, final seeds, worker budget, model budgets, and incumbent Transformer configuration.
`configs/benchmark/search_spaces.yaml` is the bounded, readable best-of-family search policy.
Changing either benchmark file changes scientific identity and must produce a distinct study.

## Data pipeline

### Source request

The fixed Databento request uses:

| Field | Value |
| --- | --- |
| Dataset | `GLBX.MDP3` |
| Symbol | `ES.v.0` |
| Input symbology | `continuous` |
| Candlestick schema | `ohlcv-1m` |
| Session schema | `status` |
| Date interval | start inclusive, end exclusive |

`pricesanity-estimate` prices both time-series requests without downloading. `pricesanity-download`
repeats that estimate, enforces the configured maximum additional cost, and then manages chunks. OHLC
defaults to one calendar year per chunk; status defaults to three months. The manifest records
request identity, completed ranges, rows, states, and checksums. Temporary files become final only
after validation and atomic publication. Transient failures retry only the active chunk. A later
status failure does not invalidate an already committed OHLC chunk.

Reissuing a matching managed request resumes missing work. `--repair` is for adopting an older
unmanaged request directory or continuing its repair manifest; it merges validated rows into the
canonical request CSV rather than leaving probe data as a separate corpus. Cost protection still
applies to missing paid ranges.

Raw CME/Databento records, logs, manifests, annotations, generated Parquet, and model outputs live
under ignored paths. They must not be committed to the source repository.

### Preparation and session trust

`pricesanity-prepare` performs these operations:

1. Read the one-minute OHLC CSV in bounded row chunks.
2. Parse timezone-aware UTC timestamps and finite OHLC values.
3. Restrict candidates to configured weekdays and the broad RTH window.
4. Aggregate complete one-minute groups into five-minute OHLC candles.
5. Extract only scheduled trading open/close transitions from Databento status records.
6. Build date-specific schedules using the configured open and the earlier of the scheduled close
   or configured maximum close. The date-specific result preserves official early closes.
7. Attach daily dataset conditions and retain only `available` sessions.
8. Require the exact five-minute timestamp grid for the scheduled interval.
9. Maintain reference-only trustworthy sessions so each eligible session opening has a validated
   preceding close.
10. Normalize globally, then select eligible timestamps and publish aligned Parquet files.

Missing schedule evidence does not fall back to a visually plausible OHLC day. Degraded,
unavailable, unknown, incomplete, duplicated, out-of-order, or impossible candles are rejected.
The status-derived close is floored to its minute because the transport record may arrive
milliseconds after the scheduled boundary.

Preparation writes two products:

- `data/interim/*_ohlc.parquet`: clean five-minute OHLC used by annotation and charts;
- `data/processed/*.parquet`: aligned timestamp, instrument, and normalized model features.

Both contain the same session-validation metadata. It records exact current and previous session
boundaries, conditions, reference timestamp, and reference close. Loaders recompute every opening
gap and exact session grid before allowing annotation or benchmark initialization.

### Normalized features

For a current candle with prices `O`, `H`, `L`, `C` and the chronologically preceding close
`P`, the feature vector is:

```text
open_gap         = (O - P) / P
body             = (C - O) / P
high_from_close  = (H - C) / P
low_from_close   = (L - C) / P
```

The single denominator preserves the entire candle on one causal relative scale. The preceding
close may be the prior candle in the same session or the validated close of the preceding
available session. A zero reference is invalid. The first source candle has no output row; it
exists only to provide history for the next candle.

### Annotation store

SQLite stores one row per candlestick ID with `current_regime`, `anticipated_regime`, and update
time. One transaction inserts or updates both labels together; the application never persists a
half annotation. The GUI displays validated OHLC, existing label spans, the current target candle,
opening-gap information, session and corpus progress, and keyboard or button choices. It can
navigate a date range without reopening the process.

Treat the SQLite file as one database, not as a mergeable text document. Two independently edited
copies can contain conflicting rows, timestamps, and schema state; Git or file-copy merging cannot
reconcile SQLite pages safely. Move or back up the complete closed database, or implement an
explicit row-level export/import policy before distributing annotation work across machines.

## Dataset construction and representations

Only sessions with a complete label pair for every required candle enter training. The join keeps
candlestick ID, timestamp, session date/index, candle position, four features, and two integer
targets. Sessions stay chronological and windows never cross a boundary.

The controlled benchmark creates one example from 16 completed candles ending at the target:

- sequential families receive `[sample, 16, 4]` float32 values;
- tabular families receive the same values flattened to `[sample, 64]`;
- scoring begins at zero-based candle position 15, so every value is real history.

The best-of-family track selects a context of 16, 32, or 64 for learned families and scores the
same target candle IDs from zero-based position 1. Early examples are right-aligned with historical
zero padding plus a Boolean validity mask. A real standardized zero is therefore distinguishable
from missing history. Sequence models mask or pack invalid positions. Tabular models append the
validity bits to flattened market values.

Preprocessing is fitted on training history only. The controlled track fits four feature means
and scales once across unique training candles before overlapping windows are built. The
best-of-family tabular and sequence adapters fit lag-appropriate or feature-appropriate statistics
using only validity-marked training values, restore padded positions to zero, and leave Boolean
indicators unstandardized. Tree families do not require adapter scaling; families based on
distance, likelihood, margin, or dense gradients do.

## Benchmark protocol

The configured corpus contains 2,690 eligible sessions: 2,190 development sessions followed by a
500-session final holdout. Configuration validation refuses a different count or a fold that
reaches the holdout. Five expanding chronological development folds select configurations. Fixed
development sessions 2,101–2,190 evaluate the configured learning-curve prefixes.

The primary objective is:

```text
(current macro-F1 + anticipated macro-F1) / 2
```

Macro-F1 always uses the fixed Bull/Bear/Range class universe. Accuracy, per-class scores,
confusion matrices, transition timing, neighborhoods around transitions, uncertainty evidence,
runtime, throughput, parameter count, serialized size, and learning curves remain secondary
outputs. Confidence intervals and paired differences resample entire sessions, not overlapping
candle rows.

`initialize` makes a format-2 snapshot with physically separate `development.parquet` and
`sealed_holdout.parquet`. Development loading does not open or hash holdout bytes. Tuning and
learning curves consume only development. `freeze-development` verifies and binds every declared
family and track selection. Only `final --confirm-final-holdout` can open the final file after that
global seal exists.

## Model reference

Every learned adapter implements fit, prediction, uncertainty semantics, description, save, load,
and fitted-state identity. The two output heads use separate classifiers or linear heads while
sharing one input representation.

| Registry identity | Concrete class/library | Input and preprocessing | Search controls | Determinism, output, persistence, scaling |
| --- | --- | --- | --- | --- |
| `majority_class` | `MajorityClassBaseline`, native | labels; ignores market values | none | deterministic one-hot distribution; readable JSON state; constant model size |
| `previous_regime` | `PreviousRegimeBaseline`, native | previous human labels; diagnostic and not deployable | none | deterministic distribution; native state; constant model size |
| `gaussian_naive_bayes` | `GaussianNB` via `SklearnDualHeadAdapter` | flattened values, training-only standardization | `var_smoothing` | deterministic probabilities; pickle; linear fit/storage in features and classes |
| `logistic_regression` | multinomial `LogisticRegression` | standardized flattened values | `C`, `class_weight` | seeded solver, probabilities, pickle; compact linear model |
| `polynomial_logistic` | `PolynomialFeatures` + `LogisticRegression` pipeline | standardized values followed by degree-two expansion | fixed degree 2, `C` | seeded probabilities, pickle; dense feature count grows quadratically |
| `knn` | `KNeighborsClassifier` | standardized flattened values | neighbors, uniform/distance weighting | deterministic probabilities, pickle; stores training rows and has expensive exact inference |
| `decision_tree` | `DecisionTreeClassifier` | flattened values; no adapter scaling | depth, minimum leaf size | seeded probabilities, pickle; nodes grow with fitted complexity |
| `random_forest` | `RandomForestClassifier` | flattened values; no adapter scaling | trees, depth, minimum leaf size | seeded stochastic fit, probabilities, pickle; CPU workers explicitly bounded |
| `gradient_boosting` | `HistGradientBoostingClassifier` | flattened values; no adapter scaling | iterations, learning rate, leaf nodes | deterministic at current policy, probabilities, pickle; random-row internal early stopping disabled |
| `rbf_svm` | exact `SVC(kernel="rbf")` | standardized flattened values | `C`, `gamma` | native predicted class plus uncalibrated decision scores; pickle; potentially quadratic/cubic fit cost |
| `mlp` | `MLPClassifier` | standardized flattened values | width, layers, learning rate; fixed `max_iter=300` | seeded stochastic probabilities, pickle; random-row early stopping disabled |
| `tcn` | `RegimeTCN` through `TorchSequenceAdapter` | sequential values and validity mask; training-only feature standardization | context, channels, kernel, layers, optimization | seeded PyTorch softmax estimates; `torch.save`; parallel causal convolutions |
| `gru` | `RegimeGRU` through `TorchSequenceAdapter` | compacted/packed real sequence history | context, hidden size, layers, optimization | seeded PyTorch softmax estimates; `torch.save`; recurrence limits time-axis parallelism |
| `transformer` | `RegimeTransformer` through `ExistingTransformerBridge` and `TorchSequenceAdapter` | sequence plus causal and padding masks | context, model width, heads, layers, feed-forward width, dropout, optimization | seeded PyTorch softmax estimates; `torch.save`; attention cost is quadratic in context |

### TCN architecture

The TCN projects four features into channels with a `1 × 1` convolution, then applies residual
GELU blocks with dilation `1, 2, 4, ...`. Each block pads only on the historical side. Invalid
padded states are cleared after the input projection and every biased convolution so artificial
history cannot propagate to the target. Separate linear heads produce three current and three
anticipated logits at each position; the adapter selects the final target-candle logits. Projection,
convolution, padding, GELU, and linear algebra use PyTorch primitives, while Price Sanity defines
the causal residual arrangement, mask handling, and dual-head contract.

### GRU architecture

The GRU carries hidden state from older to newer candles and never runs backward. For padded
opening examples, the adapter compacts real history, packs true lengths, and reads the final valid
state. Separate linear heads classify both targets. Hidden size and layer count control capacity.
The recurrent operation is PyTorch `nn.GRU`; Price Sanity defines chronology, packing, final-state
selection, preprocessing, and the two-head task.

### Transformer architecture

The Transformer projects each four-value candle into `model_dimension` and adds a learned position
embedding before pre-normalized PyTorch Transformer encoder layers with GELU feed-forward blocks.
An upper-triangular Boolean mask prevents every query from reading future candles, while a separate
padding mask removes artificial history. Two linear heads emit class logits at each real candle.

Position semantics depend on the input workflow. The standalone Transformer receives complete
sessions whose zero position is the session open, so its learned indices are absolute
time-of-session positions. The benchmark bridge receives one fixed causal context window per
sample, so indices restart at zero for every window and represent relative positions inside that
context. Best-of-family padding remains right-aligned and masked; the target candle occupies the
final context position regardless of how much real opening history is available.

The adapter selects the final target-candle logits from each fixed causal window. Projection,
embedding, encoder layers, attention, normalization, dropout, and linear heads use PyTorch
primitives. Price Sanity defines their causal/padding masks, arrangement, feature contract, and
two-task output semantics.

The registered benchmark incumbent is context 16, model dimension 12, four layers, four attention
heads, feed-forward dimension 48, dropout 0.1, learning rate 0.001, weight decay 0.01, batch size 8,
and 50 epochs. It is guaranteed a tuning trial but is not treated as a result or presumed winner.

### Scaling safeguards

Exact RBF SVM, kNN, and degree-two polynomial logistic can become expensive on the full
candle-window corpus. Preflight estimates sample counts, dimensions, dense matrix memory, and
runtime risk. Exact RBF SVM at 100,000 or more samples requires a pilot and explicit
`--acknowledge-scaling-risk`. An approximation would be a different registered family and is never
silently substituted.

## Determinism and reproducibility

Scientific identity hashes the snapshot, annotation and normalized sources, model settings,
representation, feature order, label map, exact chronological session IDs, protocol, search
space, and seed. Changing one creates a different run. Partial-resume compatibility additionally
checks installed source bytes, Git state, Python and library versions, actual device/backend,
hardware, platform, and thread budget.

Deterministic families run once at final evaluation. Stochastic finalists run seeds 42, 137, and
271, and reports require the complete declared set before aggregation. PyTorch construction,
DataLoader ordering, and optimization use scoped seeds. CPU requests deterministic algorithms
where supported. MPS, CUDA, and ROCm may retain backend/version-specific floating-point or kernel
nondeterminism; metadata records rather than conceals that boundary.

Scikit-learn adapters pickle fitted estimator and standardizer state. PyTorch adapters save model,
architecture, optimization configuration, standardizer, and format metadata with `torch.save`.
Run-level SHA-256 checksums protect either format before loading. Only trusted local model files
should be deserialized.

## Artifact lifecycle

A benchmark run publishes:

```text
run_identity.json
run_progress.json
resume_environment.json
model.bin
predictions.parquet
metrics.json
benchmark_metadata.json
.run.lock
```

Model completion, prediction completion, and metric completion are separately checkpointed and
hash-linked. A crash after fitting resumes from the exact model; a crash after predictions derives
metrics without inference; a crash after metrics verifies and publishes final metadata. The final
metadata file is the commit marker. Read-only reports reject incompatible candle populations and
load large prediction tables only for selected detail views.

Uncertainty is typed. Ordinary classifiers provide probability estimates; neural families provide
softmax probability estimates; baselines provide deterministic distributions; RBF SVM provides
uncalibrated decision scores. The GUI never labels an arbitrary margin as probability or certainty.

## Public command reference

All commands are installed from `[project.scripts]` in `pyproject.toml`.

### `pricesanity-estimate`

Required: `--start YYYY-MM-DD`, `--end YYYY-MM-DD` (exclusive). Prints separate OHLC, status,
and total estimates without downloading. It reads remote metadata and does not mutate local data,
so it is safe before annotation or benchmark completion.

```bash
pricesanity-estimate --start 2015-11-01 --end 2026-09-12
```

### `pricesanity-download`

Required: `--start`, exclusive `--end`, and `--max-cost-usd`. Optional:

- `--raw-data-directory` (default `data/raw`);
- `--candlestick-chunk-years` (default 1);
- either `--status-chunk-months` (default 3) or `--status-chunk-years`;
- `--retries` (default 2 per transiently failing chunk);
- `--estimate-only`;
- `--repair` for adoption or repair-manifest resume.

The maximum cost applies to additional attempts in the invocation, including retries.
The command writes a request directory containing canonical candlestick/status CSV, condition
JSON, manifest/checkpoint state, checksums, and a log. It is safe before benchmark completion; a
paid request still requires a deliberate cost ceiling.

```bash
pricesanity-download --start 2015-11-01 --end 2026-09-12 --max-cost-usd 20
```

### `pricesanity-prepare`

Requires `--config`, raw `--candlesticks` CSV, `--status` CSV, `--conditions` JSON, normalized
`--output`, and OHLC `--interim-output`. `--csv-chunk-rows` defaults to 100,000. `--overwrite` is
required to replace an existing product.
This mutates only the two named derived outputs and is safe before benchmark completion.

```bash
pricesanity-prepare --config configs/default.yaml \
  --candlesticks data/raw/CORPUS/candlesticks.csv \
  --status data/raw/CORPUS/status.csv --conditions data/raw/CORPUS/condition.json \
  --output data/processed/CORPUS.parquet \
  --interim-output data/interim/CORPUS_ohlc.parquet
```

### `pricesanity-annotate`

Requires interim `--candlesticks`, aligned `--normalized`, and `--config`. `--database` defaults to
`data/annotations/pricesanity.sqlite3`.
It mutates that SQLite file through atomic upserts and is the intended operation while the corpus
is incomplete.

```bash
pricesanity-annotate --candlesticks data/interim/CORPUS_ohlc.parquet \
  --normalized data/processed/CORPUS.parquet --config configs/default.yaml
```

### `pricesanity-train`

Requires `--normalized` and `--config`; accepts `--database`, optional OHLC `--candlesticks`, and
single-run `--checkpoint`. `--walk-forward` uses `--output-directory` and can select `--run-index`
or `--start-run`. `--resume` skips verified complete runs; `--overwrite` explicitly replaces
selected historical artifacts. Split and optimization options are `--training-sessions`,
`--validation-sessions`, `--test-sessions`, `--model-dimension`, `--batch-size`, `--epochs`,
`--learning-rate`, `--weight-decay`, `--gradient-clip`, `--patience`, and
`--device auto|cpu|cuda|mps`.
The command reads Parquet and SQLite and writes checkpoints, predictions, metadata, and metrics.
It can support incremental research before the full benchmark corpus is complete, but its results
must be described as a standalone walk-forward study rather than final benchmark evidence.

```bash
pricesanity-train --normalized data/processed/CORPUS.parquet \
  --database data/annotations/pricesanity.sqlite3 --config configs/default.yaml \
  --walk-forward --device auto
```

### `pricesanity-test`

Requires `--run` and `--config`. Optional `--candlesticks` overrides the OHLC path recorded during
training. The Qt view reads saved predictions and never retrains.

```bash
pricesanity-test --run data/models/walk_forward/run_001 --config configs/default.yaml
```

### `pricesanity-benchmark`

Global `--config` defaults to `configs/benchmark/default.yaml`. Subcommands are:

| Subcommand | Required or important options | Effect | Example |
| --- | --- | --- | --- |
| `hardware` | none | read-only; print environment/backend evidence | `pricesanity-benchmark hardware` |
| `device-smoke` | device; optional compare | temporary synthetic fit/save/reload; no study mutation | `pricesanity-benchmark device-smoke --device mps --compare` |
| `models` | none | read-only registry list | `pricesanity-benchmark models` |
| `profile` | `--synthetic`; optional sizes/artifact root/output | profile representations; writes only optional report/output probes | `pricesanity-benchmark profile --synthetic` |
| `initialize` | normalized, database, OHLC, project config, study directory | write immutable snapshot and manifest; requires full configured corpus | `pricesanity-benchmark initialize --normalized N --candlesticks C --database A --project-config configs/default.yaml --study-directory STUDY` |
| `pilot` | study, model selection, track; optional device/search spaces | write one development diagnostic run; holdout remains sealed | `pricesanity-benchmark pilot --study-directory STUDY --model transformer --track controlled` |
| `tune` | study, model selection, track; optional device/search spaces/fold/risk acknowledgement | mutate development Optuna/fold artifacts and selected winner | `pricesanity-benchmark tune --study-directory STUDY --all-models --track controlled` |
| `learning-curve` | study, model selection, track; optional device/search spaces | write fixed-development-evaluation curve runs | `pricesanity-benchmark learning-curve --study-directory STUDY --model gru --track best_of_family` |
| `freeze-development` | study; optional search spaces | write global seal; permanently block study development mutation | `pricesanity-benchmark freeze-development --study-directory STUDY` |
| `final` | study, model selection, track, confirmation; optional device/search spaces | open sealed bytes and write final runs; only after global freeze | `pricesanity-benchmark final --study-directory STUDY --all-models --track controlled --confirm-final-holdout` |
| `plan` | session count | read-only split validation | `pricesanity-benchmark plan --session-count 2690` |
| `run` | model selection, track, mode, count, dry-run | read-only action/preflight validation; does not train | `pricesanity-benchmark run --all-models --track controlled --mode tuning --session-count 2690 --dry-run` |

For `pilot`, `tune`, `learning-curve`, and `final`, model selection is exactly one of `--model` or
`--all-models`; track is `controlled` or `best_of_family`; study devices are `cpu`, `mps`, or
`cuda`. A focused `tune --fold N` is diagnostic and cannot freeze a winner.

### `pricesanity-benchmark-report`

`--artifact-root` defaults to `data/models/benchmark`. The command verifies committed run summaries,
aggregates complete seed sets, adds baseline deltas, and prints an honest empty state when no final
runs exist. It is read-only and safe at every stage.

```bash
pricesanity-benchmark-report --artifact-root data/models/benchmark
```

### `pricesanity-compare`

`--artifact-root` defaults to `data/models/benchmark`. Opens the read-only Benchmark Explorer over
verified artifacts; it does not train or infer and is safe at every stage.

```bash
pricesanity-compare --artifact-root data/models/benchmark
```

## GUI and report behavior

The annotation, test-result, and comparison windows share a native Qt theme and semantic
Bull/Bear/Range colors. Qt inherits the platform's installed system font rather than requesting a
nonexistent family. The annotation view preserves keyboard focus rules, responsive chart sizing,
date eligibility, automatic atomic save, and progress. Read-only views say so explicitly.

The Benchmark Explorer contains leaderboard, aligned A/B session, confusion/metric,
learning-curve, and efficiency views. It rejects different ordered candlestick identities rather
than inner-joining them. It reads prediction Parquet lazily when detailed comparison is requested.
Notebook files use tested artifact helpers rather than duplicating metrics or training code.

## Development checks

Run the complete suite from the repository root:

```bash
QT_QPA_PLATFORM=offscreen python -m pytest -q
```

Create a source-only review archive from committed files with:

```bash
git archive --format=zip --output=pricesanity-source.zip HEAD
```

Before a scientific run, also inspect:

```bash
pricesanity-benchmark hardware
pricesanity-benchmark models
pricesanity-benchmark plan --session-count 2690
pricesanity-benchmark profile --synthetic
```

Treat a corpus-count mismatch, missing strict validation evidence, changed run identity, incomplete
seed set, or incompatible resume environment as a failure to investigate rather than a guardrail
to bypass.

## Specialized references

- [Benchmark protocol](benchmark_protocol.md): split and research design
- [Benchmark execution](benchmark_execution.md): study operations
- [Algorithm notes](algorithm_notes.md): conceptual model explanations and complexity
- [Leakage rules](leakage_rules.md): causal invariants
- [Artifact format](artifact_format.md): files, checksums, and resume stages
- [Reproducibility](reproducibility.md): identity, seeds, and environments
- [Statistical comparison](statistical_comparison.md): metrics and bootstrap design
- [GUI design](gui_design.md): interaction and report requirements
- [Efficiency](efficiency.md): resource policy
- [Accelerator setup](accelerator_setup.md): CPU, MPS, CUDA, and ROCm installation and diagnostics
- [Documentation index](README.md): scientific, architecture, operations, and verification map
