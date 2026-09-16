# Efficiency and scaling

Timing records training once and uses a warmup plus repeated combined inference passes, reporting
the median. Accelerator adapters synchronize around timing so queued MPS/CUDA work is included.
Reports include sample counts, samples per second, serialized bytes, meaningful parameter counts,
device, platform, and the explicit CPU worker budget.

Runtime comparisons are meaningful only on the same hardware and resource budget. A random forest
does not silently receive every CPU core while another model receives one.

Dry-run preflight prints training/evaluation samples, raw and transformed dimensions, candidates,
folds, seeds, and approximate jobs. Exact RBF SVM may scale worse than linearly. kNN moves much of
its cost to inference. Degree-two polynomial expansion grows to `C(d+2, 2)-1` terms. Ensembles
scale with trees/iterations; neural models scale with context, width, layers, epochs, and device.

Causal windows use NumPy stride views within each session and concatenate once into an immutable
corpus. Fold selection owns its rows. Standardizers are never reused across folds because their
values belong only to each fold's training history.

The preparation executor bounds its immutable representation cache at 512 MiB. Cache eviction
is safe and may reduce speedup in large studies. Best-of-family contexts share a maximum-context
window and exact target/mask identities. Training-fitted model state remains specific to each
fold. Use `pricesanity-benchmark profile --synthetic` to measure the same infrastructure on the
machine that will run the study.

Before tuning, preflight counts actual eligible training and validation windows from the
largest development fold. For best-of-family it uses the largest declared context and includes
explicit validity indicators in tabular dimension estimates. Polynomial memory is estimated for
both float32 and float64. These estimates exclude estimator workspaces and are lower bounds on
peak resident memory. No exact SVM substitution or silent downsampling is performed.

Use `pricesanity-benchmark hardware` and `device-smoke --compare` for device/build evidence and
small synchronized CPU/accelerator timing. CPU estimators record CPU even when a mixed-family
invocation assigns an accelerator to neural models. Fits and inference retain their original
measurements through model/prediction/metadata crash recovery.
