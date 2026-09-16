# Reproducibility and artifact integrity

A run identity hashes model parameters, representation, feature order, label map, exact
training/validation/test sessions, annotation snapshot, normalized dataset, protocol, search
space, and seed. Changing a scientific input creates another run path instead of resuming stale
state. Metadata also records library versions, platform, Git revision, and dirty-tree state.

Model, prediction, and metric files have checksums. Publication writes a temporary file, flushes
and synchronizes it, validates it, atomically replaces the final path, and checkpoints completion.
Metadata is published last as the commit marker. One advisory lock allows one cooperating writer
per run while different run directories remain parallel.

Summary loading verifies metadata and metrics without reading prediction Parquet. Full loading
verifies every checksum. Full audit additionally recomputes scientific metrics from predictions.
Reproduction still requires the private market-data and annotation snapshots named by their
identities; licensed/private artifacts are intentionally excluded from Git.

## Seed scope

`tuning_seed` controls Optuna candidate sampling and repeatable tuning fits. It is deliberately
separate from `final_seeds`: the search runs once, then the frozen stochastic winner is fitted at
each final seed. For TCN, GRU, and Transformer, model construction happens inside a scoped PyTorch
seed, and training uses a scoped seed for DataLoader shuffling and optimizer randomness. CPU
training enables deterministic algorithms where PyTorch supports them. Tests require identical
initial parameters and identical tiny-run predictions for the same CPU seed, while different
seeds may differ.

The scope restores the caller's CPU PyTorch and NumPy RNG state. Apple MPS and CUDA/ROCm kernels
can have backend- or version-specific nondeterminism; Price Sanity records that environment and
does not claim cross-device bitwise equality. Reproducibility means same declared code,
dependencies, data identity, configuration, seed, and compatible backend—not that two unrelated
accelerators must emit identical floating-point bits.

## Scientific identity versus resume compatibility

Scientific identity describes the question being asked: data snapshot, sessions, representation,
model settings, seed, protocol, label map, and search space. Resume compatibility describes
whether it is safe to continue a partially serialized Python model. The latter records Python,
NumPy, pandas, scikit-learn, PyTorch, device/backend, and platform. A mismatch blocks partial
resume without pretending the scientific question changed. Fully committed checksummed artifacts
remain portable wherever their formats can be read.

## Source identity and study sealing

Partial execution records the installed package's canonical Python-source hash,
Git revision and dirty state, actual device, hardware, and thread budget. The source hash
identifies dirty source bytes; a Boolean dirty flag alone cannot distinguish two edits.
Completed artifacts remain readable; unfinished execution through changed source is refused.

Each Optuna study binds snapshot, protocol, effective track-specific search space, family,
track, exact fold scope, tuning seed, objective version, and selection source. Studies without
this identity are refused. Candidate-fold artifacts checkpoint completed fits and
predictions, so retrying an interrupted trial reuses its completed folds. A study-level advisory
lock serializes tuning, freezing, and final publication within the same study directory.

The declared models and tracks are stored in `study_scope.json`. The default scope contains all
configured families and both tracks. `freeze-development` verifies all selections and publishes
`development_frozen.json`, binding configuration/representation/search-space hashes, snapshot,
protocol, scope, and source. Tuning, pilots, and learning-curve mutation then refuse to run.
Final evaluation rechecks the seal and requires explicit confirmation before opening holdout.
