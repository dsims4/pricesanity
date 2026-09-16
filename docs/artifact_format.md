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

## Partitioned snapshot and study records

New executor snapshots use format 2. `benchmark_snapshot.json` binds
`development.parquet`, `sealed_holdout.parquet`, their checksums, partition/total counts,
columns, and source identities. Loading development never opens or hashes the sealed file.
The old format-1 snapshot loader remains available for archived standalone inspection;
protected study execution requires the partitioned format.

`study_scope.json` declares families and tracks before tuning. Each Optuna fold scope has
`study_identity.json`. Selected configurations include parameter, representation, protocol,
search-space, snapshot, and selection-source identities. `development_frozen.json` binds
all declared selections together. It is the prerequisite for an explicitly confirmed final
holdout load; a per-model selected file alone is insufficient.

Candidate folds and final runs share the stage-aware artifact writer. `run_progress.json`
records `training_evidence` together with the fitted-model checksum, before inference begins.
Prediction publication records the full `efficiency` evidence and its source-model hash.
Metrics record their source-predictions hash. A metrics-complete interrupted run verifies
that chain and publishes metadata without loading an estimator or recomputing metrics.
Partial resume also verifies installed source bytes, Git state, dependencies, actual device,
hardware, and thread settings. Completed metadata includes each run's declared seed set.
Reports verify the exact set before averaging, and keep different protocol/source identities
separate even when model settings and session dates happen to match.
