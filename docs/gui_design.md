# Price Sanity GUI design

`src/pricesanity/gui/theme.py` owns the shared light palette, semantic regime colors, typography,
spacing, buttons, focus states, table styling, and Qt-drawn icons. Annotation, saved results,
and Benchmark Explorer apply the same theme. Existing CLI launchers remain the entry points;
a new hub is unnecessary for these three independent workflows.

Use 20 px outer margins, 12 px main spacing, a 24 px screen heading, and readable 13 px control
text. Let Qt select an installed sans-serif system font; bundle no font or icon files. Primary
push buttons have a minimum 44 px height and horizontal padding. Calendar navigation keeps its
compact specialized sizing. Layouts expand the chart, whose minimum height is 280 px.
On screens below 900 logical pixels high (including 1080p at 150% scaling), outer margins
shrink to 8 px, main spacing to 4 px, and the chart minimum to 240 px. The 44 px buttons
remain unchanged so the complete annotation workflow fits without clipping.

Current and anticipated annotation choices occupy separate cards. Each Bull/Bear/Range button
has text, a direction/range icon, a visible shortcut, and a checked border; color alone never
communicates selection. The original two-key chart workflow and automatic advance after a
successful atomic pair save remain intact. Clicking a regime button follows that same save
path and returns keyboard focus to the chart. Date inputs retain their own key handling.

Progress counts only eligible sessions supplied by the validated loader. A complete session
requires every candle's pair to be saved. The header shows completed/total sessions, percentage,
remaining sessions, active date, and candle position; the session bar counts saved candles.
Unsaved first choices are distinct from a committed pair. Database errors retain the existing
retry behavior. No success modal interrupts repeated annotation.

Read-only windows say READ ONLY in their title area. Tables cannot edit stored values. They
consume persisted artifacts, with full prediction loading delayed until A/B comparison opens.
The leaderboard aggregates declared stochastic seeds before display and includes explicit
baseline deltas. Track/model filters reduce clutter. Learning curves show one selected family
with separate track lines and the fixed development-evaluation interpretation.

Candlestick geometry, candle IDs, timestamps, and regime alignment remain unchanged. Bull,
Bear, and Range share the green/red/ochre mapping in charts and controls. Labels identify spans;
transition ticks mark changes. A/B panels remain symmetric. Probability curves use a [0,1]
axis; if either model supplies uncalibrated scores, each model receives its own labeled evidence
axis. Confusion matrices display the three named classes and raw counts.

Keep focus borders visible, disabled controls readable, and Tab traversal logical. Icons are
painted at the device pixel ratio and do not depend on a symbol font. Offscreen tests cover
selection/save behavior, button dimensions, progress, read-only state, exact A/B identity,
uncertainty semantics, and empty artifacts. Visual review must additionally inspect real
rendering at 1080p, 1440p, and scaled layouts; tests are not a substitute for checking clipping.


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
