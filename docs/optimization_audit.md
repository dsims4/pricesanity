# Optimization and profiling audit

Run the maintained infrastructure profile without fitting a study model:

```zsh
pricesanity-benchmark profile --synthetic \
  --output data/models/benchmark/local_profile.json
```

The default deterministic synthetic shape is 2,690 sessions × 81 candles with context 16. It
measures snapshot load, unique-candle standardizer fit, causal-window construction, flattened
representation conversion, fold selection, and array byte counts. Supplying `--artifact-root`
also measures summary scans and one full artifact load. Values are explicitly machine-specific.

The initial local audit found individual window construction small enough to keep the direct
NumPy design: overlapping windows are stride views inside each session and concatenate once.
The preparation pass additionally measured repeated candidate requests and added the bounded,
fold-specific representation cache described below. Fold selection itself remains simple.

Repeated expensive operations remain the right optimization targets. Combined dual-head inference
already prevents running an estimator twice. kNN and polynomial logistic still use clear
scikit-learn estimators; sharing neighbor searches or polynomial matrices is deferred until the
maintained profile identifies either as an actual end-to-end bottleneck and exact semantics can be
preserved. These estimator-specific optimizations remain unimplemented; representation reuse
does not cache model-fitted state across folds.

Preflight computes degree-two transformed dimension and dense float32/float64 bytes for training
and evaluation rows. It warns before memory becomes large. Exact RBF SVM has an explicit policy:
100,000 or more estimated training samples requires a pilot and acknowledgement. If infeasible, it
is reported as an exact-kernel scaling limitation. An approximation would be a separately named
model, never an undisclosed substitution.


## September 15, 2026 preparation measurements

Synthetic inputs only; WSL2 / Ryzen 7 7800X3D / Python 3.14.7. These single-machine probes
are infrastructure measurements, not statistical estimates or predictive benchmark results.

| Operation | Before | After | Probe |
| --- | ---: | ---: | --- |
| Repeated controlled representation requests | 1.118 s | 0.018 s | 100 requests; 3 training sessions, 1 validation session, 6 candles/session |
| Pooled session bootstrap | 0.116 s | 0.011 s | 120 sessions × 81 candles; 200 replicates; same RNG draws |

The representation probe includes the first cache miss. The cache is bounded at 512 MiB and
keys exact training/evaluation histories; long studies may evict folds when this budget is
exceeded. Best-of-family 16/32 contexts are trailing views of a maximum-context representation.
This avoids repeating session logic without sharing model-fitted state across folds.

Ten degree-2 expansions of a 2,000 × 64 float32 matrix took 0.039 s. Ten kNN geometry queries
(500 queries against 2,000 rows, 64 features, five neighbors, one CPU thread) took 0.054 s.
Those small probes do not establish a meaningful end-to-end training improvement from replacing
scikit-learn's separate head implementations. They remain simple, with their duplicated work
explicitly acknowledged. Large-corpus pilot/preflight evidence should guide any later change;
no benchmark data are silently downsampled or exact SVM replaced.

Saved summaries already avoid prediction Parquet. The Explorer now also avoids eagerly opening
its default A/B predictions while the leaderboard is displayed. Full checksum/recompute audit
remains available separately. Inspect one selected comparison to load its prediction rows.
