# Price Sanity

Price Sanity is an educational time-series machine-learning project that measures how well
causal classifiers reproduce a human interpretation of E-mini S&P 500 (ES) price action. It
prepares licensed Databento candles, records human annotations, trains models, and compares
fourteen model families under a chronological benchmark.

The outputs measure agreement with an annotator. They do not establish trading profitability
or market causality.

## Status and scientific task

Data preparation, annotation, training, benchmark execution, resumable artifacts, reports, and
GUI applications are implemented. Annotation is in progress. No final full-corpus benchmark
result is claimed.

Each completed five-minute candle has two Bull/Bear/Range targets:

- `current_regime`: the regime governing price action at the candle;
- `anticipated_regime`: the regime the annotator expects the candle to lead to next.

Models receive four causal measurements: opening gap, body, high relative to close, and low
relative to close, all scaled by the trustworthy preceding close. Human labels are targets,
not learned market-data inputs. The persistence reference is explicitly diagnostic: it predicts
from the preceding human label and is not deployable without annotations.

Eligibility requires trustworthy scheduled status evidence, an `available` dataset condition,
a complete configured candle grid, and the preceding-session reference chain. Official early
closes retain their shorter grids. Missing schedule evidence means ineligible; complete-looking
prices never justify a historical fallback. See the [data pipeline](docs/project_guide.md#data-pipeline).

## Benchmark design

Price Sanity uses a chronological **90/10 benchmark protocol** over the complete eligible
annotated corpus supplied to a run. The last `ceil(0.10 × N)` whole sessions are held-out test;
the earlier sessions support development, model selection, and final training. For 550 sessions
this is 495/55; for 2,690 it is 2,421/269. Two tracks answer different questions:

- **Controlled:** every learned family receives the same 16 candles × four values.
- **Best-of-family:** each learned family selects causal context on development folds while
  preserving identical scored candle IDs. Masked left padding represents missing opening history.

A **benchmark snapshot** freezes exact joined features, targets, and source identities. Its
**sealed holdout** is physically separate from development data. The **development freeze** binds
all declared family/track selections before explicit final access is allowed. Tuning never reads
holdout rows. Stochastic finalists use the full declared seed set; reports never select a lucky
seed. The standalone Transformer walk-forward workflow is a separate evaluation.

See the [benchmark protocol](docs/benchmark_protocol.md) and
[execution lifecycle](docs/benchmark_execution.md) before starting a study.

## Installation and accelerator portability

Requires Python >=3.11. Create a project-local environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,gui,training,benchmark]'
```

`dev` supplies tests/notebooks; `gui` supplies Qt/plotting; `training` requires PyTorch >=2.4;
`benchmark` supplies scikit-learn, Optuna, and resource controls. Install only the extras needed.

Price Sanity supports CPU execution, Apple Silicon through MPS, and compatible NVIDIA CUDA or
AMD ROCm environments through PyTorch. Platform-specific binaries are distinct from the project
API requirement. For an accelerator, follow [accelerator setup](docs/accelerator_setup.md) to
select the correct PyTorch build before installing other extras. WSL is one supported execution
context, not a project requirement.

```bash
pricesanity-benchmark hardware
pricesanity-benchmark device-smoke --device cpu
# Use mps for Apple Silicon, or cuda for a compatible NVIDIA/AMD environment.
pricesanity-benchmark device-smoke --device cuda --compare
```

A successful import alone does not prove accelerator availability. Keep diagnostic evidence
with the study rather than treating one tested machine as a universal compatibility guarantee.

## Quick workflow

1. Estimate and acquire OHLC, scheduled status, and condition records with
   `pricesanity-estimate` and the cost-guarded `pricesanity-download`.
2. Run `pricesanity-prepare` to publish aligned validated OHLC and normalized Parquet.
3. Use `pricesanity-annotate` to save atomic current/anticipated label pairs in SQLite.
4. Use `pricesanity-train` for the standalone Transformer workflow, or `pricesanity-benchmark run`
   for the full model suite on the currently available complete eligible annotations.
5. Inspect saved results with `pricesanity-test`, `pricesanity-benchmark-report`, and
   `pricesanity-compare`. Viewers never retrain or infer during navigation.

The [project guide](docs/project_guide.md#public-command-reference) contains executable command
examples and required arguments for every stage. `COMMAND --help` describes each CLI. Inspect
benchmark configuration without training:

```bash
pricesanity-benchmark models
pricesanity-benchmark plan --session-count 550
pricesanity-benchmark run --session-count 550 --dry-run
```

Run the complete suite on both tracks with the same command as annotation grows:

```bash
pricesanity-benchmark run \
  --normalized data/processed/ES-v-0_2010-06-06_2026-09-12.parquet \
  --candlesticks data/interim/ES-v-0_2010-06-06_2026-09-12_ohlc.parquet \
  --database data/annotations/pricesanity.sqlite3 \
  --project-config configs/default.yaml --device cuda \
  --artifact-root data/models/benchmark --acknowledge-scaling-risk --resume
```

Use the supported device for your machine. Add `--dry-run` to discover and print the actual
population without training or writing artifacts. An optional `--session-count N` selects the
first N complete eligible sessions; no cap uses all available. Incomplete annotations are omitted.
Every invocation freezes its exact population before training. Population fingerprints separate
studies beneath the artifact root; `--resume` continues a matching study. To continue an older
frozen population after annotations grow, use `run --study-directory STUDY --resume --device cuda
--acknowledge-scaling-risk` without source arguments.

Runs while the corpus grows are **interim development benchmarks**, recorded as `corpus_status:
interim`. Their inspected trailing blocks are not permanently untouched holdouts. The same command
and split rules apply when a terminal corpus is chosen; terminal interpretation does not create a
separate implementation. Full `run` completes selection and learning curves across both tracks,
seals development globally, then evaluates each selected model on the frozen test sessions.

Publish a completed run or sealed completed study as a deterministic, Git-safe summary:

```bash
pricesanity-benchmark publish-results \
  --run data/models/benchmark/generalized_550_POPULATION \
  --output results/benchmark/PUBLIC_RESULT
```

The exporter uses an explicit field allowlist. It includes protocol and population hashes,
configuration/tuning evidence, seed-aggregated metrics, both heads' exact / ±1 / ±2 transition
diagnostics, safe environment metadata, and learning curves when present. It never copies models,
prediction rows, exact session or candlestick IDs, annotations, databases, features, or licensed
market data. Repeating the same export is byte-identical; a different existing destination is not
overwritten.


## Development data sufficiency

Measure the value of more annotated history before completing the permanent benchmark:

```bash
pricesanity-benchmark data-sufficiency \
  --normalized data/processed/ES-v-0_2010-06-06_2026-09-12.parquet \
  --candlesticks data/interim/ES-v-0_2010-06-06_2026-09-12_ohlc.parquet \
  --database data/annotations/pricesanity.sqlite3 \
  --project-config configs/default.yaml \
  --model transformer --track controlled --train-sizes 100 200 300 400 500 \
  --device mps --output-directory data/models/data_sufficiency/transformer_500
```

This is **development only** and answers a different question: does more training history help
on one fixed later population? The complete prepared eligibility catalog establishes its 90/10
development boundary before feature or label queries. Known official snapshots under the configured
benchmark root further restrict allowable dates; use `--benchmark-snapshot PATH` for a snapshot
elsewhere. The command never opens `sealed_holdout.parquet`. This catalog-based diagnostic remains
separate from `run`, which splits the currently complete annotated population.

Only complete eligible annotated sessions are selected, chronologically. Training uses nested
prefixes of 100/200/300/400/500 sessions. Every point evaluates exactly the same later candles:
with 536 complete development sessions, evaluation is sessions 501–536, starting at candle
position 15 under the existing controlled 16×4 representation. Fewer than 20 later sessions
requires `--allow-small-evaluation` (at least two remain mandatory); 20–49 is preliminary.

The registry adapters and metrics are unchanged. Four-feature scaling is fitted on unique
training candles separately at each size. There is **no tuning or evaluation-based checkpoint
selection**. The Transformer uses the benchmark incumbent; other families use registry defaults.
`--fixed-config PATH` accepts a JSON mapping from every selected model name to its fixed
conceptual parameters (for example `{"transformer": {"epochs": 20}}`). Unspecified parameters
then use registry defaults, not incumbent values; the complete resolved settings are recorded.

`--model` accepts several names, including `majority_class previous_regime logistic_regression
 gradient_boosting mlp tcn gru transformer`; `--all-models` selects the full registry.
Stochastic families use declared seeds 42/137/271; `--seeds 42` is a faster screening run with
unmeasured seed stability. Deterministic families run once per size. Previous-regime predictions
use prior human labels and remain a non-deployable reference.

`curve.csv` records current, anticipated, and mean-head macro-F1, plus separate observed-seed
SDs. `marginal_gains.json` reports adjacent gains and gains per additional 100 sessions.
Paired 95% intervals resample **whole evaluation sessions**, using the same draws for both
sizes and every seed, recomputing pooled F1 before averaging seed scores. These intervals
measure session uncertainty conditional on the declared seeds, not combined seed uncertainty.

`assessment.json` reports STILL_RISING, PLAUSIBLY_PLATEAUING, or INCONCLUSIVE. The configurable
`--meaningful-gain 0.01` and `--small-gain 0.005` are practical absolute-F1 criteria per 100
additional sessions, not scientific laws. Rising requires positive paired support and a
material recent gain without a conflicting recent trend. Plateauing requires two small
recent gains in both heads and small latest upper bounds. Wide intervals, substantial seed
variation, fewer than 20 evaluation sessions, or one stochastic seed remain inconclusive.
Reference baselines cannot establish data sufficiency. No 1,000-session projection is fitted.

The output also contains an immutable development snapshot, source/configuration identities,
exact session/candle populations, per-size scalers, and checksummed seed-level model,
prediction, metric, software, hardware, and timing artifacts under `runs/`. `summary.json`
provides the complete machine-readable report. Add `--resume` to the identical command to
validate completed points and continue from an exact fitted checkpoint after interruption.
Resume uses frozen annotations; newly annotated data requires a new output directory.
Source/library/device changes require a new diagnostic as well. Completed evidence is never
silently replaced. Repeated inspection can overfit this development block, and its recent
market regimes cannot establish how performance will behave at 1,000 sessions.

## Models and results

The registry contains majority and previous-regime baselines; Gaussian Naive Bayes; logistic
and polynomial logistic regression; kNN; decision tree; random forest; histogram gradient
boosting; exact RBF SVM; MLP; TCN; GRU; and causal Transformer. See the
[model reference](docs/project_guide.md#model-reference) and [algorithm notes](docs/algorithm_notes.md).

Mean current/anticipated macro-F1 selects configurations. Reports retain individual heads,
confusion matrices, transition diagnostics, probabilities or uncalibrated scores, runtime,
model size, and learning curves. Confidence intervals resample whole sessions because
neighboring candle windows are dependent. Completed runs have checksummed models, predictions,
metrics, and metadata; interruptions resume from compatible persisted stages.

## Repository and private state

```text
configs/                 scientific and operational configuration
docs/                    project guide and categorized technical references
notebooks/               thin saved-artifact reports
src/pricesanity/
  annotation/            labels and SQLite storage
  benchmark/             protocol, execution, metrics, artifacts, reporting
  data/                  acquisition, evidence, validation, normalization
  features/              causal windows and equivalent representations
  gui/                   annotation and read-only result interfaces
  models/                common adapters and benchmark models
  training/              standalone Transformer workflow
tests/                   unit, integration, leakage, and artifact checks
```

Licensed data, annotations, trained weights, benchmark artifacts, caches, package metadata,
and installer downloads are local generated/private state. They do not belong in source control.
The [artifact format](docs/artifact_format.md) describes persisted study products.

## Verification and documentation

Run the maintained suite from the repository root:

```bash
QT_QPA_PLATFORM=offscreen python -m pytest -q
```

See [reproducibility](docs/reproducibility.md) for identity and resume rules, and
[efficiency](docs/efficiency.md) for resource checks before expensive fits.

The [documentation index](docs/README.md) groups scientific specification, implementation,
operations/environments, verification, and generated-state contracts. Start with the
[project guide](docs/project_guide.md) for detailed commands and data flow; use
[GUI design](docs/gui_design.md) for annotation and saved-result interactions.

## License

Copyright © 2026 Duncan Sims. All rights reserved.

Price Sanity is proprietary software. Source availability does not grant
permission to copy, modify, redistribute, republish, commercially exploit, or
create derivative works from this project.

See [LICENSE](LICENSE) for the complete terms.
