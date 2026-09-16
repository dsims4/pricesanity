# GUI and report architecture

The Benchmark Explorer is read-only and artifact-only. It recursively discovers final metadata,
verifies checksums, and initially loads only small metadata/metric JSON. Prediction Parquet is read
only when A/B detail is selected. Navigation never trains or invokes a model.

The **Leaderboard** calls the same central seed aggregation used by the CLI report. Its grouping
includes track, model configuration, representation, and exact evaluation-session identity.
Stochastic rows require every expected seed. Prediction scores remain comparable across hardware;
runtime means appear only when hardware fingerprints match. Plain deltas against majority and the
non-deployable persistence reference remain secondary fields.

The **A/B Session Comparison** rejects mismatched ordered candle identities instead of using an
inner join. It shows six aligned regime bands, model uncertainty evidence, and—when `--candlesticks`
was recorded at snapshot initialization—a checksummed OHLC chart. The **Metrics / Confusion** tab
draws readable matrices. **Learning Curves** and **Efficiency** read their corresponding saved run
fields. All panels have an honest empty state before artifacts exist.

Notebook cells stay thin: load verified artifacts, call helpers in
`benchmark/notebook_reports.py`, and display. Metric, bootstrap, alignment, and grouping logic lives
in normal tested modules rather than being copied into notebooks or Qt callbacks.

