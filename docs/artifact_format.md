# Benchmark artifact format

Each run directory contains:

```text
run_identity.json       canonical scientific identity
run_progress.json       checksum checkpoints for resumable components
resume_environment.json library/backend compatibility for an interrupted run
model.bin               family-specific fitted state
predictions.parquet     exact IDs, labels, classes, and probability estimates or scores
metrics.json            common dual-head, transition, probability, and efficiency metrics
benchmark_metadata.json final commit marker and checksums
.run.lock               advisory writer lock file
```

`uncertainty_kind` controls interpretation. `probability_estimate` and
`softmax_probability_estimate` contain rows summing to one but are not automatically calibrated.
`deterministic_distribution` is a one-hot baseline output. `uncalibrated_decision_score` contains
arbitrary-margin scores and blank probability columns; the GUI must never label these certainty.

Prediction rows are unique and chronological. Exact candlestick IDs, timestamps, session IDs,
candle positions, and human labels support strict A/B alignment. Incompatible evaluation
populations are rejected rather than inner-joined.

Progress also records the exact fitted-model state hash that produced predictions and the exact
prediction checksum that produced metrics. If only the model survived a crash, the executor loads
that checkpoint rather than constructing a new stochastic instance. Scientific metrics are
recomputed from the persisted prediction rows before the final metadata commit. This prevents a
directory from mixing an old model, new predictions, and unrelated metrics even when all settings
and seeds appear the same.
