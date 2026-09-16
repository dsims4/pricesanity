# Model families

The registry uses one representative per distinct learning family.

| Model | Family | Default representation | Status |
| --- | --- | --- | --- |
| Majority class | Frequency baseline | None | Ready |
| Previous regime | Persistence reference | Prior human label | Ready; diagnostic only |
| Gaussian Naive Bayes | Probabilistic | Standardized lags | Ready |
| Logistic / polynomial logistic | Linear / explicit interactions | Standardized tabular | Ready |
| kNN | Neighbor | Standardized tabular | Ready |
| Decision tree / random forest | Tree / bagged trees | Tabular | Ready |
| Histogram gradient boosting | Boosted trees | Tabular | Ready |
| RBF SVM | Kernel | Standardized tabular | Ready with native classes/scores |
| MLP | Feed-forward neural | Standardized tabular | Ready |
| TCN | Convolutional sequence | Sequence | Ready |
| GRU | Recurrent sequence | Sequence | Ready |
| Transformer | Attention sequence | Sequence | Existing architecture adapter ready |

The TCN pads only on the historical side, the GRU carries state forward, and the Transformer
adapter reuses the existing tested causal architecture. All use training-only standardization and
the same dual-head artifact contract. Existing `pricesanity-train` commands and checkpoint format
remain separate and compatible.

The RBF SVM keeps `SVC.predict()` as the authoritative class and stores uncalibrated decision
scores. It does not enable hidden libsvm probability calibration because row-level calibration is
inappropriate for overlapping time-series windows. The MLP likewise disables scikit-learn's
random-row early stopping; duration is selected by the outer chronological protocol.

Scikit-learn and Optuna remain optional under `.[benchmark]`. Downloading, preparation,
annotation, and the original Transformer do not import them.
