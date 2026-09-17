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

The configured corpus has **2,690 eligible sessions**: 2,190 for development and 500 later sessions
for final evaluation. Two tracks answer different questions:

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
4. Use `pricesanity-train` for the standalone Transformer workflow, or complete annotation
   and follow the benchmark lifecycle: initialize, pilot, tune both tracks, learning curves,
   global freeze, then explicitly confirmed final evaluation.
5. Inspect saved results with `pricesanity-test`, `pricesanity-benchmark-report`, and
   `pricesanity-compare`. Viewers never retrain or infer during navigation.

The [project guide](docs/project_guide.md#public-command-reference) contains executable command
examples and required arguments for every stage. `COMMAND --help` describes each CLI. Inspect
benchmark configuration without training:

```bash
pricesanity-benchmark models
pricesanity-benchmark plan --session-count 2690
pricesanity-benchmark run --all-models --track controlled \
  --mode tuning --session-count 2690 --dry-run
```

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
