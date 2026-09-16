# Final preparation report — September 15, 2026

## A. Benchmark correctness

Disabled histogram boosting's implicit internal validation. Best-of-family tabular scaling now
excludes padded positions from fitted market statistics, neutralizes absent values, and preserves
explicit validity indicators. Controlled preprocessing still fits unique training candles before
constructing overlapping windows. Current and anticipated heads now have independent transition
diagnostics. Seed aggregation checks the exact declared seed set and separates incompatible
protocol/data identities. Incomplete configurations appear separately without partial rankings.
Fixed a populated leaderboard bug where sorting during insertion mixed cells between rows.

## B. Final-holdout protection

New snapshots physically separate `development.parquet` from `sealed_holdout.parquet`.
Development loaders verify and read only development bytes; tuning never materializes final rows.
Initialization validates paired OHLC/normalized evidence before freezing the corpus. Existing
single-file snapshots remain inspectable, but protected execution requires the partitioned format.

`freeze-development` requires all-fold winners for every family and track in the immutable study
scope. The global `development_frozen.json` binds those selections, source, protocol, representations,
search spaces, and snapshot. Development mutations are rejected after sealing. Final evaluation
requires this global seal and explicit `--confirm-final-holdout`; a per-model winner cannot satisfy
the lower-level holdout loader. The seal is a workflow boundary, not filesystem encryption.
No real final-holdout labels, predictions, or metrics were inspected or evaluated in this pass.

## C. Study / resume identity

Snapshot identity binds both partition checksums, columns, row/session counts, and provenance.
Selections bind protocol, effective search space, representation semantics, and source. Optuna
studies separately bind snapshot, folds, tuning seed, objective version, model, track, and source;
changed or unidentified studies cannot silently resume. Installed package source is hashed in
addition to recording Git state. Active executors reject source changes.

Partial continuation additionally checks environment, hardware, device, and thread compatibility.
Model and training evidence are checkpointed before inference. Prediction checkpoints preserve
inference timing; metric checkpoints permit metadata publication without repeating earlier work.
Synthetic crash tests exercise all three boundaries. Completed artifacts remain readable without
requiring their original environment to be installed exactly. Study locks serialize mutations.

## D. Best-of-family

All twelve learned families can select context: Gaussian Naive Bayes, logistic regression,
polynomial logistic regression, kNN, decision tree, random forest, histogram gradient boosting,
RBF SVM, MLP, TCN, GRU, and Transformer. Default choices are 16, 32, and 64; controlled context
stays 16. Baselines do not tune context.

Best-of-family contexts retain identical target candle IDs starting at candle position 1.
Left padding carries a validity mask: sequence models mask/pack missing positions, while tabular
models receive validity indicators. Scheduled early-close sessions retain their actual shorter
lengths. Controlled sequence and flattened inputs contain the same standardized values. Exact
identity and numerical-equivalence tests cover context slicing and representation reuse.

## E. Optimization

Measured synthetic probes on this machine:

| Operation | Before | After |
| --- | ---: | ---: |
| 100 repeated controlled representation requests | 1.118 s | 0.018 s |
| 200 pooled bootstrap replicates, 120 × 81 candles | 0.116 s | 0.011 s |

The first probe has three training sessions and one validation session, six candles each, and
includes the first cache miss. A 512 MiB bounded cache keys exact fold histories and representation
settings. Smaller contexts use trailing views of maximum-context arrays. Bootstrap sums per-session
confusion matrices with replacement multiplicities; single and paired results match reference
implementations using identical RNG draws.

Ten degree-two expansions of 2,000 × 64 values took 0.039 s; ten kNN searches of 500 against
2,000 rows took 0.054 s. These probes did not establish an end-to-end benefit sufficient to replace
the clear separate-head sklearn implementations. Their duplicated work remains acknowledged.
Preflight reports actual eligible sample counts, transformed dimensions, and scaling risks before
expensive execution. Nothing silently downsamples data or substitutes approximate SVM.

Summary views avoid prediction Parquet. A/B predictions load only when that view opens and the
selected pair is reused during session navigation. These infrastructure measurements are neither
full-corpus runtime promises nor predictive results. See [optimization audit](optimization_audit.md).

## F. WSL / RX 7800 XT

Tested WSL2 kernel `6.18.33.2-microsoft-standard-WSL2`, Python 3.14.7, PyTorch
`2.14.0+cu130`, Ryzen 7 7800X3D (8 physical / 16 logical cores), and 15.2 GiB RAM exposed to WSL.
The installed wheel reports CUDA 13.0, no HIP build, and no available accelerator. **ROCm
acceleration did not run; RX 7800 XT support is unverified in this environment.**

Tiny synthetic CPU measurements after warm-up, two epochs, batch 8, one CPU thread:

| Model | Fit seconds | Training samples/s | Inference samples/s |
| --- | ---: | ---: | ---: |
| TCN | 0.0108 | 5,939 | 24,452 |
| GRU | 0.0148 | 4,310 | 26,522 |
| Transformer | 0.0390 | 1,640 | 8,539 |

Training throughput counts processed epoch samples. All three passed prediction/save/reload
checks. GPU throughput and CPU/GPU ratios are unavailable. New `hardware` and `device-smoke`
commands make this reproducible and fail clearly for explicitly unavailable devices. A compatible
AMD WSL driver, ROCm PyTorch wheel, and supported Python version still need to be installed and
verified before choosing `--device cuda` on this Radeon machine. See the
[WSL/AMD guide](wsl_amd_training.md) for authoritative installation references.

## G. GUI

A shared theme defines typography, spacing, focus states, regime colors, and resolution-aware
Qt icons across annotation, session results, and Benchmark Explorer. Annotation has separate
current/anticipated cards, 44 px buttons, candle navigation, saved-pair feedback, and eligible-session
progress. The original keyboard pair workflow and automatic advance remain intact. Compact spacing
fits 1080p at 150% scaling without reducing button size.

Read-only result windows show saved evidence. Explorer adds track/model/completion filters,
seed-complete rankings, symmetric A/B comparison, independently labeled score/probability axes,
candle evidence inspection, confusion matrices, learning-curve plotting, and hardware/thread details
in efficiency views. Existing launch commands remain sufficient; no extra hub was introduced.

Manual image inspection covered synthetic annotation, session results, empty and populated
leaderboards, A/B comparison, and confusion matrices. Captures exercised 1920 × 1080 and
2560 × 1440 layouts; a 150%-scale annotation capture verified a 1280 × 700 logical window fits
inside 1080p. A real WSL graphical launch also succeeded. Synthetic acceptance artifacts lack a
price-source path, so their A/B panel displays an explicit unavailable-price state; annotation and
session-results captures exercised actual synthetic candlesticks. GUI tests supplement this review.

## H. Comments / documentation

Reasoning comments were added around snapshot isolation, study/source identity, resume stages,
fold representation reuse, padding statistics, bootstrap sufficient statistics, lazy result loading,
and scaled annotation layout. Expanded protocol, execution lifecycle, architecture, leakage rules,
artifact format, reproducibility, statistical comparison, efficiency, optimization audit, and
algorithm educational notes. Added [GUI design](gui_design.md) and [WSL/AMD](wsl_amd_training.md)
guides. Report notebooks remain artifact-driven templates, with development-only and single-session
empty states handled explicitly; no real benchmark conclusions or outputs were invented.

## I. Data policy

The user's clarified policy is implemented unchanged: trustworthy scheduled-status session
boundaries, `available` dataset condition, a complete configured candle grid, and the existing
prior-close trust chain are mandatory. Invalid sessions are rejected. Complete-looking OHLC cannot
establish historical hours; early closes are never guessed; no pre-status fallback is permitted.

Recomputing the current preparation pipeline from local raw candles, status, and condition evidence
produced **2,690 eligible sessions / 214,791 candles**, **November 23, 2015–September 11, 2026**.
Raw data beginning June 2010 do not justify expanding the corpus backward. Dates before trustworthy
scheduled coverage in November 2015 are excluded; November 20, 2015 supplies the first trusted
reference close and is not itself annotation-eligible. Later invalid sessions remain excluded.
This matches the explicitly confirmed intended policy. Documentation now calls the historical
fallback retired, rather than unresolved. No data-pipeline eligibility rule was weakened.

## J. Tests

Final full suite: **433 passed, 0 failed, 5 skipped, 0 dependency-blocked**, in 40.96 seconds.
All five skips require unavailable Apple MPS. Nineteen warnings comprise deliberately tiny MLP
non-convergence checks and an sklearn deprecation from a legacy probability-semantic test.
ROCm validation is separately unavailable as described above; it is not counted as a passing test.

The synthetic acceptance study exercises six families, both tracks, two chronological folds,
learning curves, global freeze, final seeds, report generation, and artifact-preserving resume.
Additional tests cover leakage, source/study mismatches, crash recovery, exact seed sets, padding,
early closes, identical targets, bootstrap equivalence, read-only GUI behavior, and score semantics.
Existing original Transformer command/artifact tests pass. CPU neural fit/save/reload smokes pass.
`compileall` and `git diff --check` pass.

Archive hygiene inspection of `git archive --worktree-attributes ... HEAD` found 146 members and
zero prohibited data, environment, secret, cache, or archive members. This was a hygiene check of
HEAD, not delivery of the uncommitted working changes. Added export exclusions for future archives.
The user's pre-existing `Archive.zip` deletion and untracked `.python-version` were preserved.

## K. Deferred work

Remaining human annotation; expensive full-corpus hyperparameter tuning; real final-holdout
evaluation after global freezing; and scientific conclusions from those real results. No further
core benchmark subsystem is deferred. Local accelerator installation/verification is the explicit
environment caveat in section F. All fitting and final-evaluation acceptance work here used
synthetic fixtures.

## L. Verdict

**READY WITH MINOR CAVEATS**

The annotation and benchmark workflow is implemented and CPU-validated, including protected final
access, fair family comparisons, interruption recovery, and saved-artifact inspection. The installed
environment cannot yet substantiate RX 7800 XT throughput; configure and verify ROCm before relying
on GPU execution. Full-corpus exact-kernel and polynomial resource requirements still warrant their
documented pilot/preflight checks. No real benchmark was run.
