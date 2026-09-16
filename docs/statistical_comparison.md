# Statistical comparison

Macro-F1 gives Bull, Bear, and Range equal importance. Price Sanity reports it separately for the
current and anticipated heads and uses their transparent mean as the declared selection objective.
Accuracy, class metrics, confusion matrices, transition timing, and probability quality remain
secondary evidence.

Overlapping candles inside a session are dependent. The primary confidence interval resamples
whole sessions, concatenates all candles from sampled sessions, and recomputes pooled current,
anticipated, and mean-head macro-F1. It is not bootstrap(mean(per-session F1)), which estimates a
different quantity. Mean per-session F1 remains useful descriptive evidence. Paired model
comparisons draw the same sessions for both models and recompute both pooled scores before taking
their difference. Exact ordered candle identity is required.

Macro-F1 always has the fixed Bull/Bear/Range class set. If a sampled session contains only one
class, absent classes contribute zero under the same rule used by the primary metric; they are not
silently removed to inflate a replicate.

Transition diagnostics remain separate: event timing matches the same current-regime from/to
change within zero, one, or two candles; neighborhood classification measures both heads exactly
at and within ±1/±2 candles of human transitions.

Stochastic finalists report every frozen seed plus mean and standard deviation. Chronological
folds share expanding training history and are not independent experiments; their mean is a
selection summary, not an independent-sample confidence interval.
