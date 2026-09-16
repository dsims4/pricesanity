# Benchmark execution lifecycle

## 1. Initialize once

`pricesanity-benchmark initialize` joins the completed annotation database with the normalized
corpus and publishes one immutable, physically partitioned snapshot.
`snapshot/development.parquet` and `snapshot/sealed_holdout.parquet` are bound by
`benchmark_snapshot.json`. Development loading verifies only development bytes. The sealed file
is neither read nor checksummed by development commands. Legacy single-file snapshots remain
readable for archive inspection, but must be regenerated for the protected execution workflow.
Pass the paired clean OHLC Parquet with `--candlesticks`: initialization verifies its strict
scheduled-status evidence and reference chain against the normalized artifact before copying
any benchmark rows. The same recorded source supports price candles in the A/B Explorer. Source
paths and SHA-256 identities are recorded.

## 2. Pilot and tune development history

`pilot` runs one representative candidate on one fold and reports approximate time, samples,
dimensions, model bytes, and hardware. `tune` creates a persistent Optuna SQLite study seeded by
`tuning_seed`. One-fold executions live in isolated diagnostic studies; they cannot freeze a
winner. An all-fold run stores per-fold scores and freezes the selected configuration. The known
Transformer incumbent is queued exactly once.

The objective is mean validation mean-head macro-F1 over expanding chronological folds. The
final holdout is never constructed or read by tuning. Search cost is candidate count × fold count;
final random seeds apply only after selection.

## 3. Development learning curves

`learning-curve` refits the frozen family winner on growing development prefixes. Every point is
scored on zero-based sessions 2100–2189, a fixed block after every tuning fold. This keeps curve
changes attributable to training-history size rather than changing evaluation days.

## 4. Freeze the entire study, then explicitly evaluate

Run `pricesanity-benchmark freeze-development --study-directory STUDY` after completing tuning
and exploratory learning curves for **both** tracks. Every declared family needs an all-fold
selection. A one-fold diagnostic cannot freeze a winner or a study. The seal binds selected
configuration and representation hashes, protocol, effective search spaces, source, and snapshot.
Tuning and other development mutations are refused after the seal exists.

`final` refuses to reveal holdout indices without `--confirm-final-holdout`. It also refuses a
configuration selected from fewer than all development folds. Deterministic families fit once;
declared stochastic families fit seeds 42, 137, and 271. Seed is a run identity field, not part of
the configuration hash, so the three runs aggregate as one configuration.

## 5. Resume state machine

```text
reserved identity
      ↓
model + training timing ── crash → reload exact model; inference only
      ↓
predictions + inference timing ── crash → derive metrics without inference
      ↓
metrics ────────── crash → verify and publish metadata only
      ↓ coherent hashes verified
benchmark_metadata.json (commit marker)
```

A partial run compares Python, NumPy, pandas, scikit-learn, PyTorch, device/backend, and platform
against `resume_environment.json`. Scientific identity stays separate: changing a library does not
rewrite what the experiment means, but unsafe binary continuation is rejected. A completed run
remains readable wherever its artifact format is supported.

## Commands after annotation is complete

```zsh
pricesanity-benchmark initialize \
  --normalized data/processed/CORPUS.parquet \
  --candlesticks data/interim/CORPUS_ohlc.parquet \
  --database data/annotations/annotations.sqlite3 \
  --project-config configs/default.yaml \
  --study-directory data/models/benchmark/study_001

pricesanity-benchmark pilot --study-directory data/models/benchmark/study_001 \
  --model transformer --track controlled
pricesanity-benchmark tune --study-directory data/models/benchmark/study_001 \
  --all-models --track controlled
pricesanity-benchmark learning-curve --study-directory data/models/benchmark/study_001 \
  --all-models --track controlled
pricesanity-benchmark tune --study-directory data/models/benchmark/study_001 \
  --all-models --track best_of_family
pricesanity-benchmark learning-curve --study-directory data/models/benchmark/study_001 \
  --all-models --track best_of_family
pricesanity-benchmark freeze-development --study-directory data/models/benchmark/study_001
pricesanity-benchmark final --study-directory data/models/benchmark/study_001 \
  --all-models --track controlled --confirm-final-holdout
pricesanity-benchmark final --study-directory data/models/benchmark/study_001 \
  --all-models --track best_of_family --confirm-final-holdout
```

Both tracks must finish development before either final command. Exact RBF SVM, large kNN, or
large polynomial designs may require `--acknowledge-scaling-risk` during tuning after a pilot.
This acknowledgement accepts compute risk, not scientific leakage.
