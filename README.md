# Price Sanity

Price Sanity is an educational time-series machine-learning project for reproducing a human
price-action interpretation of E-mini S&P 500 (ES) intraday candlesticks. It turns licensed
Databento records into validated five-minute sessions, records two candle-level regime labels,
trains a causal Transformer, and provides a controlled benchmark across fourteen learning
families.

The project is a research system, not a trading product. Its outputs measure agreement with one
annotator's labels. They do not establish profitability, market causality, or investment
suitability.

## Current status

The data, annotation, training, benchmark, artifact, reporting, and GUI infrastructure is
implemented and tested. The benchmark protocol expects a fully annotated corpus of 2,690 strict
status-validated sessions. Annotation is still in progress, so no final multi-model benchmark or
sealed-holdout result is claimed here.

The current development configuration reserves:

- 2,190 chronological sessions for model development;
- 500 later sessions for one final holdout evaluation;
- two benchmark tracks: a controlled `16 × 4` comparison and a best-of-family comparison;
- Bull, Bear, and Range classes for both prediction heads.

## What the system predicts

Every eligible candle has two human labels:

- `current_regime`: the regime currently governing price action;
- `anticipated_regime`: the regime this candle is most likely to lead to next.

Both targets use the fixed class map `Bull = 0`, `Bear = 1`, `Range = 2`. A model receives only
causal candle geometry ending at the labeled candle. It never receives either human label as a
market-data feature.

The four normalized input features are:

```text
open_gap        = (open - previous_close) / previous_close
body            = (close - open) / previous_close
high_from_close = (high - close) / previous_close
low_from_close  = (low - close) / previous_close
```

Normalization requires a trustworthy preceding close. The first complete, available session in a
valid trust chain supplies reference history but is not annotation-eligible; a later session is
eligible only when its own candle grid and the immediately preceding evidenced session are both
trustworthy.

## Pipeline

```text
Databento OHLC, status, and condition records
                    |
                    v
chunked raw download with cost guard, resume, and checksums
                    |
                    v
RTH filtering -> five-minute aggregation -> strict session validation
                    |
                    +--> validated OHLC Parquet for charts and evidence
                    |
                    +--> normalized Parquet with four causal features
                                      |
                                      v
                          SQLite candle annotations
                                      |
                                      v
                    immutable benchmark snapshot
                  /                           \
        development Parquet             sealed holdout Parquet
                  |                           |
      tune, pilot, learning curves       explicit final command only
                  \___________________________/
                                      |
                                      v
                 checksummed models, predictions, metrics, reports
```

Session eligibility is deliberately strict. A session needs scheduled status evidence, an
`available` Databento condition, a complete scheduled candle grid, and a trustworthy preceding
session close. Official early closes are preserved from status data. Degraded, missing,
unavailable, truncated, or unscheduled sessions do not become annotation or training examples.

## Installation

The package supports Python 3.11 or newer. The current project test environment uses Python
3.14.7. Create a project-local virtual environment and install only the extras required for the
workflow:

```bash
cd pricesanity
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev,gui,training,benchmark]'
```

The extras are separated intentionally:

| Extra | Purpose |
| --- | --- |
| `dev` | pytest, notebooks, and plotting |
| `gui` | PySide6 annotation and result interfaces |
| `training` | PyTorch Transformer training |
| `benchmark` | scikit-learn, Optuna, resource controls, and profiling |

PyTorch accelerator wheels are platform-specific. Apple Silicon uses the `mps` device. A
supported AMD ROCm build on Linux/WSL is selected through PyTorch's `cuda` device name. Verify the
actual backend rather than assuming that an imported `torch` package has accelerator support:

```bash
pricesanity-benchmark hardware
pricesanity-benchmark device-smoke --device cpu
pricesanity-benchmark device-smoke --device mps --compare   # Apple Silicon
pricesanity-benchmark device-smoke --device cuda --compare  # NVIDIA CUDA or AMD ROCm
```

See [WSL and AMD training](docs/wsl_amd_training.md) for the ROCm workflow and device
interpretation.

## End-to-end workflow

### 1. Estimate and download source data

`--end` is exclusive for both commands. The API key is read through the Databento SDK's normal
environment configuration.

```bash
pricesanity-estimate --start 2015-11-01 --end 2026-09-12

pricesanity-download \
  --start 2015-11-01 \
  --end 2026-09-12 \
  --max-cost-usd 20.00
```

Downloads use yearly OHLC chunks and quarterly status chunks by default. Completed chunks are
committed incrementally. Repeating a matching managed request resumes it; `--repair` adopts an
interrupted unmanaged corpus or resumes its repair manifest. Licensed raw data and logs remain
under the ignored `data/` tree.

### 2. Prepare aligned OHLC and normalized artifacts

```bash
pricesanity-prepare \
  --config configs/default.yaml \
  --candlesticks data/raw/CORPUS/candlesticks.csv \
  --status data/raw/CORPUS/status.csv \
  --conditions data/raw/CORPUS/condition.json \
  --output data/processed/CORPUS.parquet \
  --interim-output data/interim/CORPUS_ohlc.parquet
```

The one-minute CSV is read in bounded chunks, but session validation and normalization retain a
global chronological reference chain. The two Parquet outputs have identical eligible timestamps.
Validation evidence stored in their metadata prevents a stale or incomplete pair from reaching
annotation or benchmark initialization.

### 3. Annotate eligible sessions

```bash
pricesanity-annotate \
  --candlesticks data/interim/CORPUS_ohlc.parquet \
  --normalized data/processed/CORPUS.parquet \
  --config configs/default.yaml \
  --database data/annotations/pricesanity.sqlite3
```

The Qt application shows full-session OHLC charts and the validated opening gap. It saves one
current/anticipated label pair atomically per candle. The progress display counts only complete,
eligible sessions. Annotation data is private local research data and is ignored by Git.

### 4. Train the standalone Transformer workflow

```bash
pricesanity-train \
  --normalized data/processed/CORPUS.parquet \
  --candlesticks data/interim/CORPUS_ohlc.parquet \
  --database data/annotations/pricesanity.sqlite3 \
  --config configs/default.yaml \
  --walk-forward \
  --device auto
```

This command is the expanding walk-forward Transformer workflow. It is separate
from the fixed multi-model benchmark. Inspect a saved run without retraining:

```bash
pricesanity-test --run data/models/walk_forward/run_001 --config configs/default.yaml
```

### 5. Initialize and run the multi-model benchmark

Initialization performs one validated join and freezes exact source identities:

```bash
pricesanity-benchmark initialize \
  --normalized data/processed/CORPUS.parquet \
  --candlesticks data/interim/CORPUS_ohlc.parquet \
  --database data/annotations/pricesanity.sqlite3 \
  --project-config configs/default.yaml \
  --study-directory data/models/benchmark/study_001
```

Development commands never load the sealed holdout:

```bash
pricesanity-benchmark pilot --study-directory data/models/benchmark/study_001 \
  --model transformer --track controlled --device cpu
pricesanity-benchmark tune --study-directory data/models/benchmark/study_001 \
  --all-models --track controlled --device cpu
pricesanity-benchmark tune --study-directory data/models/benchmark/study_001 \
  --all-models --track best_of_family --device cpu
pricesanity-benchmark learning-curve --study-directory data/models/benchmark/study_001 \
  --all-models --track controlled --device cpu
pricesanity-benchmark learning-curve --study-directory data/models/benchmark/study_001 \
  --all-models --track best_of_family --device cpu
```

After all declared development selections are complete, seal them and make the explicit one-time
final evaluation:

```bash
pricesanity-benchmark freeze-development \
  --study-directory data/models/benchmark/study_001

pricesanity-benchmark final --study-directory data/models/benchmark/study_001 \
  --all-models --track controlled --device cpu --confirm-final-holdout
pricesanity-benchmark final --study-directory data/models/benchmark/study_001 \
  --all-models --track best_of_family --device cpu --confirm-final-holdout
```

Do not run final evaluation until model families, features, search spaces, and reporting decisions
are frozen. The confirmation flag is a guardrail, not permission to iterate on the holdout.

## Model families

The registry contains fourteen concrete model identities:

| Identity | Implementation | Default representation |
| --- | --- | --- |
| `majority_class` | native frequency baseline | labels only |
| `previous_regime` | native persistence reference | prior human label; diagnostic only |
| `gaussian_naive_bayes` | scikit-learn GaussianNB | tabular |
| `logistic_regression` | scikit-learn LogisticRegression | tabular |
| `polynomial_logistic` | polynomial features + logistic regression | tabular |
| `knn` | scikit-learn KNeighborsClassifier | tabular |
| `decision_tree` | scikit-learn DecisionTreeClassifier | tabular |
| `random_forest` | scikit-learn RandomForestClassifier | tabular |
| `gradient_boosting` | scikit-learn HistGradientBoostingClassifier | tabular |
| `rbf_svm` | scikit-learn SVC | tabular |
| `mlp` | scikit-learn MLPClassifier | tabular |
| `tcn` | PyTorch causal dilated 1D convolution | sequential |
| `gru` | PyTorch gated recurrent unit | sequential |
| `transformer` | PyTorch causal Transformer adapter | sequential |

All learned families fit separate current and anticipated heads under one common adapter contract.
The controlled track gives every learner the same 16 completed candles and four features. The
best-of-family track may select a context of 16, 32, or 64 while preserving the same scored candle
IDs. Full preprocessing, architecture, tuning, determinism, persistence, and scaling details are
in the [project guide](docs/project_guide.md) and [algorithm notes](docs/algorithm_notes.md).

## Evaluation and artifacts

The primary selection metric is the mean of current-regime and anticipated-regime macro-F1.
Reports retain accuracy, fixed-class confusion matrices, class metrics, transition diagnostics,
probability or score semantics, runtime, model size, and learning curves. Confidence intervals
resample whole sessions because overlapping candles are not independent rows.

Each completed benchmark run contains a canonical identity, resumable progress, fitted model,
exact prediction rows, metrics, checksums, and a final metadata commit marker. Incomplete runs do
not masquerade as complete. The report and comparison GUI read verified artifacts and never train:

```bash
pricesanity-benchmark-report --artifact-root data/models/benchmark
pricesanity-compare --artifact-root data/models/benchmark
```

See [artifact format](docs/artifact_format.md), [reproducibility](docs/reproducibility.md), and
[statistical comparison](docs/statistical_comparison.md).

## Command reference

| Command | Purpose |
| --- | --- |
| `pricesanity-estimate` | estimate OHLC and status request cost |
| `pricesanity-download` | guarded, chunked, resumable Databento download or repair |
| `pricesanity-prepare` | validate sessions and write aligned OHLC/normalized Parquet |
| `pricesanity-annotate` | create or continue candle-level annotations |
| `pricesanity-train` | train the standalone causal Transformer workflow |
| `pricesanity-test` | inspect one saved Transformer test run |
| `pricesanity-benchmark` | inspect, initialize, execute, and seal benchmark studies |
| `pricesanity-benchmark-report` | print verified final benchmark summaries |
| `pricesanity-compare` | open the read-only Benchmark Explorer |

Use `COMMAND --help` for all options. Every benchmark subcommand and its guardrails are documented
in the [project guide](docs/project_guide.md#pricesanity-benchmark).

## Tests

```bash
python -m pytest
```

GUI tests run with Qt's offscreen platform in the maintained test configuration. Focused modules
can be run with `python -m pytest tests/test_NAME.py`.

## Repository layout

```text
configs/                  project, benchmark, and search-space configuration
docs/                     research protocol and engineering references
notebooks/                thin artifact-analysis notebooks
src/pricesanity/
  annotation/             labels and SQLite storage
  benchmark/              protocol, execution, metrics, artifacts, reports
  data/                   download, validation, resampling, normalization
  features/               causal windows and representations
  gui/                    annotation and read-only result interfaces
  models/                 common adapters and benchmark models
  training/               standalone Transformer training workflow
tests/                    unit, integration, leakage, and artifact tests
```

## Documentation map

- [Project guide](docs/project_guide.md): authoritative implementation and command reference
- [Benchmark protocol](docs/benchmark_protocol.md): research questions, splits, and holdout rules
- [Benchmark execution](docs/benchmark_execution.md): operational study lifecycle
- [Algorithm notes](docs/algorithm_notes.md): model-by-model educational explanations
- [Leakage rules](docs/leakage_rules.md): causal and chronological invariants
- [Artifact format](docs/artifact_format.md): persistence and resume contract
- [Reproducibility](docs/reproducibility.md): identities, seeds, and environment boundaries
- [Statistical comparison](docs/statistical_comparison.md): metrics and session bootstrap
- [GUI design](docs/gui_design.md): annotation and read-only visualization behavior
- [Efficiency](docs/efficiency.md): resource limits and scaling checks
- [WSL and AMD training](docs/wsl_amd_training.md): ROCm environment and diagnostics

Licensed CME/Databento data, annotations, trained weights, and generated experiment artifacts are
local inputs or outputs. They are intentionally not part of the source repository.
