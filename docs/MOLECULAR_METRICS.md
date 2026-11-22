# Offline molecular metrics

## Current consolidated results (2026-11-21)

The November 17/19 experiment folders and November 20/21 cap-50,000 reruns are
consolidated in `results/cowboys/merged_20261121/`. Its `RUN_SUMMARY.md`,
`run_summary.csv`, `attempts.csv` and `merge_manifest.json` identify the selected
whole attempt for every task/seed and retain failed/interrupted-run provenance.
The selected working set has 23 completed and two partial runs. Only albuterol
seeds 1 and 2 remain incomplete, after 281 and 197 BO steps respectively; both
failed at the 50,000-step sampling cap. Seventeen completed runs use a
10,000-step sampling cap and six use 50,000; this is a
progress collection, not a homogeneous five-seed algorithm comparison. Use
`configuration_summary.csv` for cap-separated completed-run statistics.

The former source folders, previous copied merge, verification/smoke/environment
checks and their older metrics are preserved in the self-contained recovery
archive `results/cowboys/_archive/cleanup_20261121.zip`. Its internal manifest maps
every original path to SHA256-addressed contents, deduplicating byte-identical
copies. Historical generated-result paths quoted below refer to that snapshot;
the tracked verification JSON files in `docs/verification/` remain in place.
The later albuterol source folder and superseded merged artifacts are preserved
in `results/cowboys/_archive/albuterol_update_20261121_085359.zip` with the same
self-contained recovery format.
`results/cowboys/merged_20261121/restore_sources.py --archive <archive.zip>` can reconstruct the original
source tree into a fresh directory. `cleanup_result.json` records the cleanup,
and `verification.json` records archive/selected-copy and metric validation.
The canonical `metrics/<run-stem>/` exports are regenerated against the new
paths without evaluating objectives or changing original checkpoint bytes.

## Artifact audit (2026-11-17)

`scripts/experiments/run.sh` invokes root `run.py`, which forwards to
`hdbo_benchmark.experiments.run`. The runner calls the unchanged registered
COWBOYS solver and `_save_solver_progress` writes
`results/cowboys/<RUN_ID>/cowboys_<task>_<seed>{.npy,_history.npz,_iteration_results.json,_status.json}`.
The fresh sweep directory and `sweep_status.tsv` identify this experiment;
`preflight/` is a separate child directory. Filenames alone do not identify
configuration: the status JSON includes the corrected solver configuration and
VAE checkpoint hash. Planned `max_iter` is **not** saved in these status files.

The corrected histories contain aligned `structures` (original padded SELFIES),
`y` scores, and latent `x`, including initialization. Status gives initialization
size and completed iterations; configuration/iteration JSON gives batch size.
Iteration JSON contains selected structures/scores, sampling counts and ranked
candidate **log-EI selection scores**. These candidate records are not extra
observations. Historical latent-only histories must not be decoded as if they
contained the original evaluated molecules. Bare `.npy` files contain final best
performance, not chronological evaluations.

Reusable formats and analysis: `utils/logging/iteration_metrics.py` already saves
best-so-far, decoded molecule counts, sampling counts and nearest-previous
distance. Its radius-3 folded **count** fingerprints, exclusion of initial
molecules from distance references, and candidate Top-K objective diagnostics
differ from the requested metrics. `evaluation/plot_iteration_metrics.py` plots
those legacy metrics; its adjusted candidate quality is not observed Top-10.
`compare_cowboys_paper.py` compares protocol-matching final scores. Both remain
unchanged. The new adapter and metrics extend the existing `evaluation` package;
they never import a solver, VAE, poli, torch, or an oracle.

The audited artifacts have no physical-call/cache flags, proposal chronology,
failed evaluation records, pre-evaluation GP mean/std, timestamps, component
times, oracle version, historical git revision or dependency snapshot. History
row counts are available; **physical oracle calls are not measured**. Exporter
versions describe the export environment, not the optimization environment.

The non-invasive integration is one post-run export command, before the existing
paper comparison, scoped to the exact sweep directory (nonrecursive), passing
the shell's planned BO budget. No optimization invocation changes. Export
failures warn and preserve the script's existing optimization exit behavior.

## Commands and definitions

Run in the established Python 3.10 environment, from the repository root:

```bash
python -m hdbo_benchmark.evaluation export --results-dir results/cowboys/<RUN_ID> --planned-bo-steps 300
python -m hdbo_benchmark.evaluation aggregate --results-dir results/cowboys/<RUN_ID>
```

Export writes `metrics/<source-stem>/run.json`, a readable `summary.json`, a flat
`evaluations.csv`, a lossless `diagnostics.json` sidecar when original iteration
diagnostics exist, and `metrics/export_manifest.json`. Large candidate/latent
arrays are linked from `run.json` instead of duplicated inside it; observations,
trajectories, configuration and all scientific metrics remain in `run.json`.
It is deterministic and atomically replaces only its own generated files.
Discovery is nonrecursive unless `--recursive` is explicitly supplied; preflight
directories are always excluded. Missing/unreadable artifacts and failures before
the first checkpoint are listed in the manifest. Historical NPZ/NPY without a
status file are reported, not silently interpreted. `--task-config` accepts a
JSON object keyed by task, for example:

```json
{"albuterol_similarity": {"hit_threshold": 0.3, "delta": 0.7, "score_direction": "maximize"}}
```

There is no default hit threshold. Thresholds use the **metric score** scale.
Direction defaults to maximizing the recorded solver objective (identity, no
normalization), not a claim about chemical desirability. In particular `sa_tdc`
retains the historical maximizing convention despite raw SA being lower-is-easier.
An explicit task `score_direction: minimize` negates recorded scores for metrics
only; no oracle values or optimization outcomes change. No upstream transform is
invented. Unknown oracle version/range remains null. Task configurations may
also specify `bo_steps_planned`, `oracle_version`, and `initialization_policy`.
Supply version and sampling policy only with evidence from that run's original
environment/protocol. A hard interruption can prevent the shell's post-run hook
from executing; use the standalone export command on the saved checkpoints.

Initialization observations have inferred step 0; a snapshot follows the whole
initial design, then every completed batch/BO step. These boundaries come from
status plus batch size and are labeled inferred, not logged events. Mismatched
checkpoint lengths retain raw observations but truncate the metric horizon to
whole completed steps, mark partial, and warn. Missing initialization/batch
metadata prevents inventing a trajectory. A completed status without a planned
budget is a completed artifact with **unverified budget completion**.
If only iteration JSON survives, its selected scores are retained as BO records,
but absent initialization prevents full-prefix trajectories and AUCs. Structures
in historical iteration JSON may be latent reconstructions; only the corrected
configuration's actual-observation tracking establishes them as observations.

Identity is RDKit canonical isomeric SMILES with all fragments retained, no
neutralization or tautomer normalization. Valid finite observations contribute to
best-so-far; unique molecules retain the maximum score seen **so far** on repeat
evaluation. Top-K averages `min(K, n)` molecules; score ties use first-seen order.
Top-1 equals best-so-far under this policy. Without original structures only
finite-score best-so-far is supported, explicitly labeled unverified validity;
Top-K and structural metrics stay null. No latent reconstruction or oracle call
is performed. Invalid/nonfinite observations remain in raw records with reasons.

Morgan **bit** fingerprints use radius 2, 2048 bits, no chirality. Diversity is
`1 - Tanimoto`. Online packing includes initialization hits in observation order;
a first threshold crossing is considered once per molecular identity and a
representative is accepted iff every distance is **strictly greater** than delta.
Representatives retain observation index, score and step. Empty Bemis-Murcko
scaffolds (acyclic molecules) are excluded from scaffold counts. Top-10 internal
diversity is null below two eligible molecules. Novelty is minimum distance to
the initial valid structures; history duplicate rate is labeled separately from
unavailable physical-call and proposal duplicate rates. Unique-hit rate per
physical oracle call stays null; per-observation yield is a separately named field.

AUC is trapezoidal over integer BO steps `0..B`, including initialization state,
divided by the **observed** horizon B. B=0 or missing values gives null with a
reason. AUC scope is `full_budget` only when completed status, saved horizon and
explicit planned budget agree; otherwise `partial` or `unverified_budget`.
This is not the official PMO 10,000-call leaderboard AUC. Optional approximate
Circles and calibration diagnostics are not implemented; their nulls have reasons.

The bundled `run_schema.json` adapts the supplied reference schema: it permits
null for unrecorded validity/call counts, supports partial/status-only exports,
and uses a map of explicit missing reasons. `validate` checks JSON Schema when
jsonschema is available and semantic trajectory/AUC/monotonicity invariants.

## Aggregation

Aggregation writes `metrics/aggregate/experiment_summary.json`, CSV tables,
`plot_manifest.json`, and PNG figures. It accepts additional roots via repeated
`--results-dir`. It partitions by task, oracle version, direction/transform,
metric policy, planned budget, initialization policy/count and diagnostic-call
setting. Unknown planned budgets, oracle versions and initialization sampling
policies are reported and excluded from full-budget statistics. Solver
configuration defines separate algorithm variants;
different variants may overlay within a compatible comparison group.

Only completed, full-budget runs with unique seed per algorithm/configuration
contribute to full-budget means; duplicate seed/configuration entries are
reported and excluded instead of counting reruns as independent replicates.
Partial/unverified curves and scalars appear individually in tables and plots,
with explicit exclusion labels. Curves are never padded, interpolated or
forward-filled. Sample SD and
deterministic percentile bootstrap 95% CIs resample independent runs, never
molecules; with one run SD/CI are null. Pointwise curve bands have the same
bootstrap convention and report n at each step. A plot manifest records skipped
figures and reasons (e.g. no threshold or no compatible full-budget runs).

The three current real verification artifacts each have seed 1: two completed
one BO step, and one failed before any completed BO step (initialization only).
They demonstrate export and plots, not independent-seed uncertainty or a 300-step
performance comparison. See the generated manifest for source hashes.

## Verification

The tracked [verification evidence](verification/molecular_metrics_verification.json)
records source hashes and the original one-step checks. Their generated outputs
are in the recovery archive described above. Regression tests cover the real
checkpoint fixture, byte preservation, deterministic export, hand-computed
prefix metrics, 10+300 indexing, strict Tanimoto boundaries, partial/missing data,
schema/semantic validation, compatibility groups and independent seed handling.

```bash
python -m unittest tests.test_molecular_metrics tests.test_structure tests.test_cowboys_paper_report tests.test_cowboys_fidelity -v
```

## Pre-evaluation GP predictions and readable views

For newly started COWBOYS processes, the runner attaches
`utils/logging/gp_predictions.py` to the candidate-return boundary. It calls the
original `next_candidate()` first, records the existing fitted GP posterior for
the selected molecule, and returns the exact same candidate object. This occurs
before `step()` calls the oracle and updates history. It does not refit a GP,
change acquisition values, call an oracle, or change evaluation order. Python,
NumPy and Torch RNG states are restored after diagnostic calls. Diagnostic errors
are saved as missing reasons and do not abort optimization.

`cowboys_<task>_<seed>_predictions.json` is an additive source sidecar, saved at
the existing checkpoints and final save, even if legacy iteration JSON is
disabled. Status advertises `predictions_file` and
`pre_evaluation_predictions_enabled`. Failed evaluations can leave an unmatched
prediction attempt; it is retained in the source sidecar, not invented as an
observation. The exporter only joins matching observation index, canonical
identity, BO step and a training history ending before that observation.

Recorded fields use the original recorded objective scale (BoTorch's outcome
standardization is undone by `posterior`):

| Field | Meaning |
|---|---|
| `mean` | Pre-evaluation GP posterior mean |
| `latent_std` | Posterior standard deviation excluding observation noise |
| `predictive_std` | Predictive standard deviation including GP likelihood noise |
| `incumbent_for_acquisition` | The same historical maximizing threshold, including the existing NaN penalty |
| `analytic_expected_improvement` | Analytic Gaussian EI from the latent posterior, relative to that incumbent |

Analytic EI is an additional diagnostic, distinct from the existing Monte Carlo
`qLogExpectedImprovement` acquisition. The latter remains the recorded
`recorded_acquisition.selection_score` / CSV `recorded_log_ei`; it is never
replaced in candidate selection. GP predictions and observed scores are paired
in `run.json` under `evaluations`, with `prediction_residual = observed - mean`.
Summary diagnostics include prequential RMSE, Gaussian NLL using predictive
variance, and coverage of the central Gaussian 90% predictive interval. These
are diagnostics on adaptively selected points, not independent test-set metrics.

Open `metrics/<source-stem>/summary.json` for final metrics/AUCs/completion and
GP diagnostic summaries. Open `evaluations.csv` to compare actual score,
predicted mean, both uncertainty definitions, analytic EI, recorded log-EI and
realized improvement on one row. All original scientific metric fields and raw
observations remain in `run.json`; only the duplicated candidate diagnostic
payload moves to `diagnostics.json`, linked with its SHA-256 hash. Source
`*_iteration_results.json` and all legacy plots retain their existing format.
Aggregation adds predicted-versus-observed and per-run prediction/uncertainty
figures; existing molecular metric figures are preserved.

Already running Python processes do not acquire the recorder retroactively.
Subsequent task/seed processes started by a running shell sweep load the updated
runner, so a sweep may contain mixed prediction availability. Old runs retain
null predictions and explicit missing reasons. No historical GP is refitted to
fill missing records, and no running sweep needs to be interrupted.

Additional checks live in `tests/test_gp_predictions.py`: before-oracle timing,
candidate/oracle-count/chain/RNG parity, prediction failure isolation, a real
BoTorch posterior on the original score scale, saving with legacy JSON disabled,
causal record matching, missing historical predictions, clean-view repeatability
and byte-identical existing plot images. No solver file is changed.

The current extension passed 49 checks. A bounded real CPU albuterol run also
recorded a pre-evaluation mean and both uncertainty definitions, with candidate,
score and structure histories exactly equal to the existing uninstrumented
seed-1 verification run. See
[`verification/gp_prediction_logging_verification.json`](verification/gp_prediction_logging_verification.json)
for the prediction values, source paths and validation evidence. Clean views were
also generated for the already completed DRD2 seed-1 run without changing any of
its original inputs; its unrecorded historical GP values remain null.

## EI versus realized improvement diagnostics

The offline `improvement` command adds a focused four-panel analysis without
changing existing plots or optimization:

```bash
python -m hdbo_benchmark.evaluation improvement --status-file results/cowboys/<RUN_ID>/cowboys_<task>_<seed>_status.json
```

It writes `improvement_analysis.{png,pdf,csv,json,md}` under that run's
`metrics/<source-stem>/figures/` (or `--output-dir`). Panels compare per-step
expected/realized gain, cumulative conditional forecasts/achieved record gain,
the best-score/Top-10 trajectories, and the EI uplift above positive mean advantage.
Initialization supplies the first incumbent. Every observed improvement uses
the best eligible score strictly **before** evaluation; gains telescope to
final best minus initial best. Tied scores create no new record. Expected gain
is comparable only when the logged acquisition incumbent matches that prefix.
Missing increments are not zero-filled or interpolated in cumulative curves.

Analytic Gaussian EI and exp(recorded qLogEI) are labeled separately: the latter
is a smoothed Monte Carlo proxy. The uncertainty uplift is
`EI - max(mean - incumbent, 0)`, a Jensen gap, not the two standard EI formula
terms or a change to the acquisition. Cumulative conditional EI is descriptive,
not a guaranteed/feasible final-score forecast. One selected adaptive run is not
a collection of independent replicates or proof that the policy ranks candidates
optimally. The command currently supports the recorded maximizing scale and one
evaluation per BO step. Source hashes are checked after plotting; the original
artifacts and mean/std plot are preserved. Hand calculations, prefix causality,
ties, incompatible thresholds and missing EI are covered by
`tests/test_improvement_analysis.py`.
