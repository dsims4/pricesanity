# Documentation

Start with the [project overview](../README.md) and use the [project guide](project_guide.md)
for commands, data flow, model reference, artifacts, and GUI workflows. Specialized documents
define the contracts below. A flat directory keeps these twelve references directly accessible;
the categories distinguish scientific choices from implementation and local environments.

| Category | Reference | Responsibility |
| --- | --- | --- |
| Scientific specification | [Benchmark protocol](benchmark_protocol.md) | Targets, strict eligibility, tracks, chronological splits, final holdout |
| Scientific specification | [Leakage rules](leakage_rules.md) | Causality, train-only transforms, label/reference boundaries |
| Scientific specification | [Algorithm notes](algorithm_notes.md) | Model assumptions, learning behavior, and scaling |
| Scientific specification | [Statistical comparison](statistical_comparison.md) | Metrics, seed interpretation, whole-session uncertainty |
| Implementation architecture | [Benchmark architecture](benchmark_architecture.md) | Module responsibilities and representations |
| Implementation architecture | [GUI design](gui_design.md) | Annotation state, read-only views, accessibility |
| Operations | [Project guide](project_guide.md) | Installation, data preparation, commands, workflows |
| Operations | [Benchmark execution](benchmark_execution.md) | Initialize, develop, freeze, explicitly evaluate, resume |
| Environment setup | [Accelerator setup](accelerator_setup.md) | Portable CPU, MPS, CUDA, and ROCm execution |
| Operations / verification | [Efficiency](efficiency.md) | Resource limits, pilots, timing, synthetic profiling |
| Verification / reproducibility | [Reproducibility](reproducibility.md) | Source/data identity, seeds, checksums, resume compatibility |
| Generated/private state | [Artifact format](artifact_format.md) | Snapshot, study, and run files; completeness and interpretation |

Scientific configuration belongs in `configs/`; implementation belongs in `src/pricesanity/`;
verification belongs in `tests/`. Notebooks consume saved artifacts rather than owning model
or metric implementations. Machine-specific setup is an execution concern and does not change
the scientific protocol.

Licensed raw/processed data, annotation databases, models, benchmark output, caches, and package
metadata are local state. Keep them outside source control in the ignored locations described
by the [project guide](project_guide.md#data-pipeline). Package metadata is regenerated from
`pyproject.toml` and README; installer downloads also remain local.
