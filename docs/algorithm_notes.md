# Algorithm notes for Price Sanity

Every learned model receives the same four causal candle measurements: opening gap, body, high
from close, and low from close. One controlled example is sixteen completed five-minute candles,
ending at the candle whose current and anticipated regimes are targets. A sequence model sees a
`16 × 4` array; a tabular model sees the same 64 numbers in chronological order. The
best-of-family track can select longer causal context, but every family is scored on the same
target candle IDs.

“Fit” means learning only from an earlier chronological training partition. “Inference” means
applying that frozen state to later candles. Parametric models learn a fixed-size parameter set;
nonparametric models retain training examples or grow structures whose size depends strongly on
data. Linear means class evidence is additive in the supplied features. Nonlinear models can
represent interactions or curved boundaries.

## Majority-class baseline

**What it is and learns.** It counts Bull, Bear, and Range labels in training history, separately
for each head, and always predicts the most frequent class. The input candle vector is ignored.
Fit learns two integers; inference repeats them. It is parametric in the weakest sense and has no
linear/nonlinear geometry.

**Assumptions and controls.** Its only assumption is that future class frequency resembles
training frequency. It requires no scaling and has no tuned hyperparameters. Training is linear
in label count, inference is constant work per row, and memory is constant.

**Why include it.** A serious classifier must beat “always choose the common regime.” Failure to
do so reveals class imbalance or lack of generalization before architecture discussions begin.

## Previous-regime persistence reference

**What it is and learns.** It predicts the immediately preceding human label. It has no fitted
parameters. Persistence is nonlinear only in the irrelevant sense that it is a lookup; it is not
a deployable market-data model because live human labels would be required.

**Assumptions and cost.** It assumes regimes persist candle to candle. Training and inference are
linear with tiny constant work and memory. No preprocessing or tuning is needed.

**Why include it.** Regime labels naturally persist. A high raw F1 is unimpressive if it does not
improve on this reference. Reports name it as non-deployable and show a plain score delta rather
than folding it into an opaque rank.

## Gaussian Naive Bayes

**Idea.** For every class and input feature, it estimates a Gaussian mean and variance. Bayes’
rule combines those likelihoods with class frequency to choose the most likely regime. Fit learns
per-class feature statistics; inference evaluates Gaussian likelihoods.

**Type and assumptions.** It is parametric and produces nonlinear decision boundaries through
quadratic likelihood terms. Its “naive” assumption says features are conditionally independent
inside a class. Lagged candle measurements plainly correlate, so the assumption is intentionally
strong. Training-only standardization improves numerical geometry; `var_smoothing` prevents tiny
variances from dominating.

**Behavior and cost.** Too much smoothing underfits; too little can overreact to nearly constant
features. Fit and inference are roughly linear in rows × features × classes, with small memory.
It belongs because it is a fast probabilistic counterpoint. Matching the Transformer would imply
that simple class-conditional distributions capture much of the annotation rule.

## Multinomial logistic regression

**Idea.** Each class receives a weighted sum of the 64 controlled inputs plus an intercept.
Softmax turns the three scores into probability estimates. Fit adjusts weights to reduce
multiclass cross-entropy; inference is a matrix multiplication.

**Type and assumptions.** It is linear and parametric. It assumes useful evidence can be expressed
additively in the supplied coordinates. Training-only standardization prevents a feature’s numeric
scale from acting like importance. `C` controls inverse regularization strength; class weighting
can compensate for imbalance.

**Behavior and cost.** Strong regularization underfits; weak regularization can fit noise. Dense
training cost grows with rows, features, classes, and optimizer iterations; inference and stored
weights remain cheap. It is the clearest test of whether regime interpretation is close to a
linear combination of recent candle geometry.

## Degree-two polynomial logistic regression

**Idea.** It explicitly adds squared features and every pairwise interaction, then fits ordinary
multinomial logistic regression. The classifier stays linear in the expanded columns but is
nonlinear in the original candle values. An interaction can express, for example, that a large
body matters differently after a large opening gap.

**Preparation and controls.** Original features are standardized from training history before the
degree-two transform. `degree` is fixed at two and `C` controls regularization. With `d` inputs,
the no-bias expansion has `C(d + 2, 2) - 1` columns. Preflight reports dense float32 and float64
memory for train and evaluation matrices before allocation.

**Behavior and cost.** It can underfit patterns needing sequential abstraction and overfit when
thousands of interactions meet limited sessions. Both time and memory scale with the expanded
matrix. It belongs between linear and black-box nonlinear models: beating plain logistic shows
explicit interactions matter; matching a Transformer suggests attention may be unnecessary.

## k-nearest neighbors

**Idea.** Fit stores standardized training examples. Inference finds the `k` closest stored
windows and lets their labels vote, uniformly or by inverse distance. It is nonlinear and
nonparametric.

**Assumptions and controls.** Euclidean closeness must correspond to similar price-action meaning,
which makes scaling essential. `n_neighbors` sets locality and `weights` sets voting. Small `k`
has low bias and high variance; large `k` smooths toward common regimes.

**Cost and lesson.** Training is cheap, but exact inference compares evaluation rows with a large
stored corpus and memory grows with training data. Preflight requires acknowledgement for very
large comparison products. Strong results would suggest annotated patterns recur locally in raw
lag geometry; weak results would show Euclidean distance is a poor semantic measure.

## Decision tree

**Idea.** A tree repeatedly splits one feature at a threshold to make child nodes purer. Leaves
store class distributions. Fit chooses splits; inference follows one root-to-leaf path. It is
nonlinear, parametric after fitting, and invariant to monotonic feature scaling.

**Controls and behavior.** `max_depth` and `min_samples_leaf` control complexity. A shallow tree
underfits; a deep tree can memorize small variations and is unstable to training changes. Typical
fit cost is near rows × features × log(rows); inference is near tree depth; memory follows node
count.

**Why include it.** It provides readable threshold rules. If it approaches the Transformer,
human regimes may be reproducible with a short hierarchy of concrete candle conditions.

## Random forest

**Idea.** It fits many decision trees to bootstrap samples while randomly restricting candidate
features at splits, then averages their class distributions. This bagging lowers one tree’s
variance. It is nonlinear and parametric once its finite forest is stored.

**Controls and behavior.** Tree count improves stability at added time and memory; depth and leaf
size govern individual-tree complexity. Scaling is unnecessary scientifically, but the controlled
track uses the same transformed values so information is identical. Forests resist overfitting
better than one tree, though overly deep forests can still learn noise.

**Cost and lesson.** Trees parallelize, so the benchmark enforces the same CPU thread budget used
by competitors. Fit and model size scale approximately with trees and nodes; inference visits
every tree. A win would say diverse threshold interactions are sufficient without explicit
sequence state.

## Histogram gradient boosting

**Idea.** Small trees are added sequentially, each correcting residual errors from the ensemble
so far. Histogram binning makes candidate splits efficient. It is nonlinear and parametric.

**Controls and behavior.** Learning rate sets each tree’s contribution, iterations set ensemble
length, and maximum leaf count controls interaction complexity. Too few/small trees underfit;
too many aggressive trees overfit. Sequential boosting cannot parallelize across iterations as
freely as a forest. Memory and inference grow with the fitted ensemble.

**Why include it.** Boosted trees are a strong tabular standard. Beating sequence models would
show that explicit temporal architecture adds little beyond nonlinear functions of ordered lag
columns.

## Exact RBF support-vector machine

**Idea.** The radial-basis kernel measures similarity through squared standardized distance and
learns maximum-margin class boundaries represented by support vectors. It is nonlinear and its
stored size depends on training examples.

**Controls and assumptions.** `C` trades margin width against training errors; `gamma` sets how
local each influence is. Scaling is mandatory. Low `C` or `gamma` underfits; high values can build
fragmented boundaries around noise. Native SVC classes are authoritative; stored decision values
are explicitly uncalibrated scores, not probabilities.

**Cost and full-corpus policy.** Exact kernel training can approach quadratic or cubic time and
quadratic memory, with inference proportional to support vectors. At 100,000 or more samples,
Price Sanity requires a pilot and explicit scaling-risk acknowledgement. If the pilot is
infeasible, exact RBF SVM is reported as a scaling limitation and omitted; it is never silently
replaced. A future Nyström or random-feature approximation must be registered under a different
family name.

## Multilayer perceptron (MLP)

**Idea.** Dense layers repeatedly apply learned linear transforms and nonlinear activations to a
flattened causal window. Fit uses backpropagation; inference runs the transforms forward. It is a
nonlinear parametric model.

**Controls and behavior.** Width, layer count, learning rate, and training duration control
capacity and optimization. Training-only standardization is important. Random row-level early
stopping is disabled because overlapping time-series rows would create a leaky internal split;
duration is selected by chronological outer folds. Too little capacity/duration underfits, while
large networks or long training can memorize.

**Cost and lesson.** Dense layer cost follows rows × connected weights; stored memory follows
parameter count. It tests whether generic nonlinearity on ordered lag columns can match dedicated
sequence structure.

## Temporal convolutional network (TCN)

**Idea.** One-dimensional convolutions slide learned local filters through candle history.
Dilation increases receptive field without looking ahead. Price Sanity pads only on the left and
clears invalid padded states after each biased convolution. It is nonlinear and parametric.

**Controls and behavior.** Channel width, kernel size, layers, context, learning rate, batch size,
weight decay, and epochs are tuned chronologically. A small receptive field underfits long
dependencies; excessive width/depth can overfit. Convolution processes positions in parallel and
has bounded memory based on activations and weights.

**Why include it.** It tests whether reusable local motifs and multi-scale composition explain
regimes. Beating the Transformer would favor locality and simpler parallel structure over global
attention.

## Gated recurrent unit (GRU)

**Idea.** The GRU carries a hidden state from older to newer candles. Learned gates decide what to
retain, update, or forget. For padded early-session inputs, Price Sanity compacts only real history,
packs its true length, and classifies the final recurrent state. It is nonlinear and parametric.

**Controls and behavior.** Hidden width, layers, context, learning rate, batch size, weight decay,
and epochs control capacity. Small state underfits; large recurrent stacks can overfit and train
slowly. Recurrence limits parallelism across time, while parameter memory stays compact.

**Why include it.** It matches the intuition of carrying a changing market-state summary. A GRU
win would suggest compressed ordered memory matters more than directly revisiting all candles.

## Causal Transformer

**Idea.** Each candle builds queries, keys, and values. Causal self-attention lets its state weight
any available earlier candle but blocks future positions. Positional encoding preserves order;
feed-forward layers transform each attended state. Padding masks keep nonexistent opening history
out of attention. Two heads classify current and anticipated regimes.

**Controls and behavior.** Context, model dimension, layers, attention heads, feed-forward width,
dropout, learning rate, weight decay, batch size, and epochs all matter. The known incumbent
(`context=16`, `d_model=12`, four layers/four heads, feed-forward 48, dropout 0.1) is guaranteed a
tuning trial, but is not assumed optimal. Too small a model underfits; excessive capacity can
memorize and increases tuning sensitivity.

**Cost and lesson.** Standard attention grows roughly quadratically with context length and
linearly with layers/batch. At these intraday contexts it remains modest, but it has more design
choices than simpler families. A reliable win would support flexible long-range interactions. A
tie with logistic, trees, TCN, or GRU would be equally educational: it would show which simpler
inductive bias reproduces the human annotations with less complexity.

## Reading the comparison

No family wins because it sounds sophisticated. Interpret predictive metrics beside majority and
persistence deltas, chronological folds, complete final seed sets, parameter/model size, learning
curves, pooled session-bootstrap intervals, and compatible-hardware runtime. This benchmark asks
which learner reproduces the two annotation tasks from causal candle geometry; it does not by
itself prove trading profitability or market causality.

## Reading a best-of-family example

The controlled example stays at 16 × 4, with a single candle-level standardizer fitted to
unique training candles before window duplication. Best-of-family search adds a causal context
choice of 16, 32, or 64 for every learned family listed above. Baselines do not waste a context
trial, and controlled searches omit the context parameter entirely. The named Transformer
incumbent remains guaranteed rather than depending on a lucky random suggestion.

At the second session candle, a 64-candle example contains two real candles and 62 absent
history positions. The mask distinguishes them from real standardized zeros. Tabular examples
append 64 validity bits to 256 market values; their scaler fits each market lag using only its
real observations and leaves the bits unchanged. Sequence models consume masks or lengths.
The same target candle ID remains in every context, including short scheduled sessions.

This distinction matters for cost as well as fairness. A degree-2 expansion of 320 tabular
inputs has 51,680 nonconstant columns, so even a compact causal representation can produce a
large dense polynomial design. Read preflight's transformed dimension and float32/float64
memory estimates before allowing the search. Exact RBF SVM and kNN likewise retain their stated
algorithms: an approximation or a sampled corpus would answer a different question and must
not be substituted silently.

Histogram gradient boosting's training duration and the MLP epoch limit are selected on the
outer chronological folds. Their internal random-row early stopping is disabled. Model-family
comparisons therefore retain the same chronology contract despite different library defaults.
