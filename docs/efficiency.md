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
