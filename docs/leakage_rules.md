# Leakage rules

Market candles are dependent observations, not interchangeable rows. Randomly assigning
candles to train and test would place nearby candles from the same session on both sides. Their
overlapping 16-candle windows could share 15 inputs, and later market conditions could affect
preprocessing or model selection. The resulting score would measure recognition of shared
history more than temporal generalization.

Price Sanity therefore enforces these rules:

1. Split complete sessions in chronological order before fitting preprocessing or models.
2. Never cross a session boundary while constructing a causal window.
3. End each input window at its target candle and exclude every later candle.
4. Fit standardizers, polynomial transforms, class frequencies, and other learned preparation
   on training rows only. Apply the frozen transform to validation and test.
5. Use development validation folds for hyperparameter choice. Do not inspect the final
   holdout while choosing features, context, thresholds, or search spaces.
6. Keep the two human labels as targets, never model inputs. The previous-regime baseline is a
   clearly marked reference exception: it reads only the immediately prior human label and is
   not deployable without live annotations.
7. Preserve candlestick IDs, timestamps, session indices, and feature order in artifacts so a
   prediction can be traced back to its exact source.
8. Generate GUI views and reports from persisted predictions. Navigation must not retrain or
   run inference.

Retrospective human labeling is a property of the annotation method: the annotator can see the
completed session. It does not permit a model input to see future candles. Results describe
agreement with those labels and temporal generalization within this dataset; they do not by
themselves establish a profitable trading strategy.
