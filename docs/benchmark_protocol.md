# Multi-model benchmark protocol

Price Sanity compares learning families on two human-labeled tasks: current market regime and
anticipated market regime. Macro-F1 is primary because Bull, Bear, and Range may not be equally
common. Accuracy, per-class metrics, confusion matrices, transition behavior, probability
quality, runtime, model size, and learning curves remain visible rather than being hidden inside
one opaque score.

## Corpus boundary

The accepted corpus policy is **strict status-validated eligibility**. Trustworthy scheduled
status evidence must support session boundaries, the dataset condition must be `available`,
and the configured candle grid must be complete. Reject degraded, missing, unavailable, or
otherwise invalid sessions. Complete-looking OHLC data cannot establish a schedule. Do not
guess early closes or apply a historical fallback before scheduled status coverage. Preserve
the data pipeline's trust chain for previous-session closing references. No historical fallback
is part of this policy.

The configured strict corpus begins in November 2015 and contains 2,690 sessions (214,791
candles, November 23, 2015 through September 11, 2026). Recompute and verify that count before
freezing the fully annotated benchmark. It is an expected validation result, not a reason to add
or discard sessions. If verification changes the count, investigate and explicitly update the
session accounting without weakening eligibility.

For that verified count, the configured plan requires 2,690 complete annotated sessions:

- sessions 1–2,190 are development history;
- sessions 2,191–2,690 are the untouched final holdout;
- expanding chronological training blocks are followed by future development validation blocks.

Configuration validation rejects a fold reaching the holdout and rejects any corpus count
different from the frozen plan. Tuning APIs do not reveal holdout indices without an explicit
final-evaluation call.

## Controlled and best-of-family tracks

Every controlled learner receives the same 16 completed five-minute candles by four normalized
features. Sequence input is `[sample, 16, 4]`; tabular input is the same 64 values flattened. The
label belongs to the final completed candle and no later candle enters the window.

Best-of-family models may use validation-selected causal context and appropriate preprocessing,
but still use only the four raw normalized measurements. Standardization is fitted from each
fold's training history. Polynomial interactions belong only to polynomial logistic regression.

Different context lengths must score the same target identities. Controlled evaluation begins at
candle position 15 because every input has sixteen real candles. Best-of-family evaluation begins
at candle position 1. A 16-, 32-, or 64-candle representation is left-padded and carries a separate
Boolean validity mask. Sequence models mask or pack absent positions; tabular models receive the
mask as explicit indicators. Padding is zeroed after preprocessing and never contributes to fitted
feature statistics. A context changes what a model may inspect, never which target candles appear
on its leaderboard. This keeps opening candles and scheduled early closes in the practical study.

For the controlled track, the four location/scale statistics are fitted from unique training
candles before overlapping windows exist. The same transformed candle values are then arranged as
`[sample, 16, 4]` for sequence models or flattened for tabular models. An interior candle therefore
gets one vote in preprocessing rather than one vote per overlapping window.

## Learning curves

Training prefixes contain 250, 500, 1,000, 1,500, 2,000, and 2,100 sessions (the largest valid
prefix before this fixed evaluation block). Curves use one fixed tuning seed and are explicitly
exploratory. Every point is evaluated
on the same future development block, sessions 2,101–2,190 (zero-based indices 2,100–2,189). This
block begins after the final tuning-validation fold ends. Preprocessing is re-fitted for each
prefix. The final holdout never enters this experiment.

## Frozen final procedure

1. Evaluate candidates on chronological development folds.
2. Select one configuration per family using development mean-head macro-F1.
3. Freeze search spaces, representation, and any training-duration rule.
4. Fit final models using legal development history.
5. Evaluate the final 500 sessions once per declared stochastic seed, or once for a deterministic
   model.
6. Aggregate seeds before ranking while retaining individual run artifacts.

Stochastic finalists report mean, standard deviation, and every configured seed result. The
luckiest seed is never selected. This fixed benchmark does not replace the expanding
walk-forward Transformer evaluation.

## Experiment design and reporting

### Three different evaluations

**Controlled benchmark:** same 16-by-4 information for every learner; isolates algorithmic
inductive bias.

**Best-of-family benchmark:** appropriate causal representation and validation-selected
context for each family; measures a practical representative.

**Production walk-forward evaluation:** the expanding Transformer workflow, where
historical test periods may become legal training history after time advances. It remains the
operational simulation and is not replaced by the fixed final benchmark.

Comparing numbers across these categories without naming the category is misleading.

### Tuning

Candidate budgets are modest and configured by family. Mean macro-F1 across chronological
development folds selects hyperparameters. The current and anticipated heads remain separately
reported even though selection averages their macro-F1 values. No tuning result may incorporate
the final holdout. Three final seeds measure stochastic variation after selection rather than
tripling the search.

### Artifacts

One initialized study has this layout:

```text
data/models/benchmark/STUDY/
    snapshot/
    tuning/
    selected/
    runs/TRACK/MODEL/RUN/
    learning_curves/TRACK/MODEL/RUN/
    study_scope.json
    development_frozen.json
```

The freeze file appears only after successful development sealing. Each run reserves
`run_identity.json`, then writes a model, predictions, metrics, and final
`benchmark_metadata.json`. The final metadata binds artifacts with SHA-256 checksums and records
track, model, representation, session ranges, seed, library versions, and dataset description.
Exclusive file creation prevents silent overwrite. An interrupted directory can resume only
when its stored identity exactly matches.

### Reporting

`pricesanity-benchmark-report` reads completed, checksummed runs. The comparison GUI does the
same and remains usable as an honest empty state before any benchmark exists. Notebook templates
call reusable loaders; they contain no training or metric implementation and no invented result.

The study predicts a person's regime labels from normalized ES candle geometry. It is an
educational comparison of generalization and data efficiency, not evidence of tradable return,
risk-adjusted performance, or market causality.
