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

The initial local audit found window construction small enough to keep the direct NumPy design:
overlapping windows are stride views inside each session and concatenate once. Fold selection is
also cheap. No cache was added because a leakage-sensitive cache key and invalidation system would
cost more complexity than the measured work.

Repeated expensive operations remain the right optimization targets. Combined dual-head inference
already prevents running an estimator twice. kNN and polynomial logistic still use clear
scikit-learn estimators; sharing neighbor searches or polynomial matrices is deferred until the
maintained profile identifies either as an actual end-to-end bottleneck and exact semantics can be
preserved. No unmeasured specialized cache was introduced.

Preflight computes degree-two transformed dimension and dense float32/float64 bytes for training
and evaluation rows. It warns before memory becomes large. Exact RBF SVM has an explicit policy:
100,000 or more estimated training samples requires a pilot and acknowledgement. If infeasible, it
is reported as an exact-kernel scaling limitation. An approximation would be a separately named
model, never an undisclosed substitution.

