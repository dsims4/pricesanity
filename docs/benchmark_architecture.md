# Benchmark architecture

```text
normalized candles + SQLite annotations
                 ↓ one validated join
checksummed immutable snapshot: development + sealed holdout
                 ↓
chronological session partitions
                 ↓
training-only candle preprocessing
                 ↓
causal windows + explicit validity masks
                 ↓
registered dual-head model adapter
                 ↓
persisted predictions → recomputed metrics
                 ↓
complete-seed aggregation
                 ↓
CLI report / Benchmark Explorer / notebooks
```

The snapshot owns copied Parquet rows containing only benchmark identity, four features, and two
targets. Live SQLite changes cannot alter it. Loaded sessions are owned data frames. Causal-window
arrays and masks are immutable after construction; fold selection makes owned copies so later
operations cannot modify a shared corpus accidentally.

Controlled preprocessing fits four statistics from unique training candles, transforms copied
training/evaluation sessions, and only then creates overlapping windows. Best-of-family adapters
may fit their own training-only preprocessing. A fitted standardizer, estimator, or neural state
belongs to exactly one training history and is never cached across incompatible folds.

Window context and evaluation population are distinct. Left padding lets contexts up to 64 score
the same early target candles and early-close sessions. A Boolean mask—not numeric zero—defines
real history. Sequential adapters consume the mask; tabular representations append it when
padding exists.

The registry is the only family-to-adapter construction boundary. The runner owns common timing,
prediction framing, and metrics. Artifacts own durable identity and resumption. Reports and Qt
views read completed artifacts only; they never fit or infer while a user navigates.


Executor representations use a 512 MiB bounded LRU cache keyed by immutable snapshot, exact
training/evaluation sessions, track, features, context contract, and target start. Controlled
folds share fitted candle statistics and arrays across families. Best-of-family construction
builds a maximum 64-candle masked representation and takes trailing views for 16/32 contexts.
Models and fitted model-specific preprocessing are never cached across folds.
