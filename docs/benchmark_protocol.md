# Multi-model benchmark protocol

Price Sanity compares learning families on two human-labeled tasks: current market regime and
anticipated market regime. Macro-F1 is primary because Bull, Bear, and Range may not be equally
common. Accuracy, per-class metrics, confusion matrices, transition behavior, probability
quality, runtime, model size, and learning curves remain visible rather than being hidden inside
one opaque score.

## Corpus boundary

The frozen plan requires exactly 2,690 complete annotated sessions:

- sessions 1–2,190 are development history;
- sessions 2,191–2,690 are the untouched final holdout;
- expanding chronological training blocks are followed by future development validation blocks.

No annotated session disappears because an earlier proposal used a round number. Configuration
validation rejects a fold reaching the holdout and rejects any corpus count different from the
frozen plan. Tuning APIs do not reveal holdout indices without an explicit final-evaluation call.

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

Training prefixes contain 250, 500, 1,000, 1,500, and 2,000 sessions. Every point is evaluated
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
luckiest seed is never selected. This additional study does not replace the existing expanding
walk-forward Transformer evaluation.
