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

The protocol identity is `generalized_chronological_90_10_v1`. For N complete eligible annotated
sessions sorted by trading date, test contains the last `ceil(0.10 × N)` sessions and development
contains all earlier sessions. No candle-level random split occurs. Examples:

| Total | Development | Test |
| --- | --- | --- |
| 550 | 495 | 55 |
| 551 | 495 | 56 |
| 1000 | 900 | 100 |
| 2690 | 2421 | 269 |

`run` discovers all complete eligible annotations by default. An optional positive session cap
selects the first N complete sessions and fails if N exceeds availability. Partly annotated sessions
are omitted as whole units. The existing strict preparation evidence is validated before selection;
apparently complete prices cannot supply missing schedule evidence. Exact ordered session and candle
identities, labels, source hashes, configuration, and the cap/auto choice are frozen before fitting.

Runs default to `corpus_status: interim`. An inspected trailing block from a growing corpus must
not be described later as permanently untouched. The same engine and command apply to a chosen
terminal corpus; finality is an interpretation of that population, not a different split algorithm.

### Generated development folds and minimum

Let D be the development count. Reserve its last `ceil(0.10 × D)` sessions for fixed learning-curve
evaluation; let L be the remaining prefix length. Start tuning with `floor(0.50 × D)` training
sessions. Divide the interval from that initial endpoint to L into five successive validation
blocks: endpoint k is `initial + floor(k × (L - initial) / 5)`, for k=0..5. Each fold trains on
all sessions preceding its validation block. These folds are shared across families and tracks.

The rules require at least 30 initial training sessions, five sessions in every tuning validation
block and the learning-curve evaluation block, five test sessions, and three sessions in the
smallest learning prefix (the fixed class universe has three classes). Every selected session must
also have scored candles for both representations. The smallest training prefix must support
every kNN neighbor count in the unchanged search space (currently up to 75 scored candles).
The derived session-count minimum is 70 complete eligible sessions; smaller corpora fail with the required counts. These are structural minimums, not a
promise of class balance or narrow uncertainty. No fold or curve is silently removed for small N.

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

Training prefixes are the unique sorted floors of 10%, 25%, 50%, 75%, and 100% of L. Here
100% means all history preceding the fixed development evaluation block, not all D sessions.
Rounded duplicates are collapsed. Curves use one fixed tuning seed and are exploratory. All points
evaluate the same last `ceil(0.10 × D)` development sessions, after every tuning-validation fold.
Preprocessing is fitted again for each prefix. Test sessions never enter this experiment.

At N=550, the training prefixes are 44, 111, 222, 333, and 445 sessions and evaluation is sessions
446–495. Tuning training endpoints are 247, 286, 326, 365, and 405; validation endpoints are 286,
326, 365, 405, and 445 (half-open zero-based intervals).

## Frozen final procedure

1. Evaluate candidates on chronological development folds.
2. Select one configuration per family using development mean-head macro-F1.
3. Freeze search spaces, representation, and any training-duration rule.
4. Fit final models using legal development history.
5. Evaluate the frozen trailing 10% once per declared stochastic seed, or once for a deterministic
   model.
6. Aggregate seeds before ranking while retaining individual run artifacts.

Stochastic finalists report mean, standard deviation, and every configured seed result. The
luckiest seed is never selected. Neural adapters train for the development-selected fixed epoch
count; test data is never passed to fitting, epoch selection, or early stopping. This benchmark
does not replace the expanding walk-forward Transformer evaluation.

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
track, model, representation, exact ordered development/test session IDs, protocol version, total/
development/test counts, rounding rule, seed, library versions, source/data hashes, hardware, timing,
and dataset description. Reports retain protocol version and population counts alongside metrics.
Exclusive file creation prevents silent overwrite. An interrupted directory can resume only
when its stored identity exactly matches.

### Reporting

`pricesanity-benchmark-report` reads completed, checksummed runs. The comparison GUI does the
same and remains usable as an honest empty state before any benchmark exists. Notebook templates
call reusable loaders; they contain no training or metric implementation and no invented result.

`pricesanity-benchmark publish-results` provides the public reproducibility layer: protocol and
population hashes, source identity, allowlisted hyperparameters and tuning budgets, declared seeds,
per-seed and aggregate metrics, transition diagnostics, and safe environment evidence. Full private
reproduction additionally requires the licensed market data, annotation database, exact ordered
session/candlestick identities, prediction rows, and trained models. Those private artifacts are
deliberately excluded rather than merely hidden by filename conventions.

The study predicts a person's regime labels from normalized ES candle geometry. It is an
educational comparison of generalization and data efficiency, not evidence of tradable return,
risk-adjusted performance, or market causality.

Benchmark launch configuration uses only rule-based format 2. Fixed absolute split plans are
rejected; archived run artifacts retain their independent reader compatibility.
