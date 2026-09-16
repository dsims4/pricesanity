# Experiment design and reporting

## Three different evaluations

**Controlled benchmark:** same 16-by-4 information for every learner; isolates algorithmic
inductive bias.

**Best-of-family benchmark:** appropriate causal representation and validation-selected
context for each family; measures a practical representative.

**Production walk-forward evaluation:** the existing expanding Transformer workflow, where
historical test periods may become legal training history after time advances. It remains the
operational simulation and is not replaced by the fixed final benchmark.

Comparing numbers across these categories without naming the category is misleading.

## Tuning

Candidate budgets are modest and configured by family. Mean macro-F1 across chronological
development folds selects hyperparameters. The current and anticipated heads remain separately
reported even though selection averages their macro-F1 values. No tuning result may incorporate
the final holdout. Three final seeds measure stochastic variation after selection rather than
tripling the search.

## Artifacts

Completed runs live beneath:

```text
data/models/benchmark/
    controlled/MODEL/RUN/
    best_of_family/MODEL/RUN/
```

Each run reserves `run_identity.json`, then writes a model, predictions, metrics, and final
`benchmark_metadata.json`. The final metadata binds artifacts with SHA-256 checksums and records
track, model, representation, session ranges, seed, library versions, and dataset description.
Exclusive file creation prevents silent overwrite. An interrupted directory can resume only
when its stored identity exactly matches.

## Reporting

`pricesanity-benchmark-report` reads completed, checksummed runs. The comparison GUI does the
same and remains usable as an honest empty state before any benchmark exists. Notebook templates
call reusable loaders; they contain no training or metric implementation and no invented result.

The study predicts a person's regime labels from normalized ES candle geometry. It is an
educational comparison of generalization and data efficiency, not evidence of tradable return,
risk-adjusted performance, or market causality.
