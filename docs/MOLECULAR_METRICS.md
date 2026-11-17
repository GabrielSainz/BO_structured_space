# Offline molecular metrics

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

Export writes `metrics/<source-stem>/run.json` and `metrics/export_manifest.json`.
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

## Handoff and generated artifacts

New source files:

- `src/hdbo_benchmark/evaluation/__main__.py`
- `src/hdbo_benchmark/evaluation/cowboys_artifacts.py`
- `src/hdbo_benchmark/evaluation/molecular_metrics.py`
- `src/hdbo_benchmark/evaluation/export_runs.py`
- `src/hdbo_benchmark/evaluation/aggregate_runs.py`
- `src/hdbo_benchmark/evaluation/metric_io.py`
- `src/hdbo_benchmark/evaluation/run_schema.json`
- `tests/test_molecular_metrics.py`
- `tests/fixtures/cowboys_checkpoint/README.md`
- `tests/fixtures/cowboys_checkpoint/cowboys_albuterol_similarity_1.npy`
- `tests/fixtures/cowboys_checkpoint/cowboys_albuterol_similarity_1_history.npz`
- `tests/fixtures/cowboys_checkpoint/cowboys_albuterol_similarity_1_iteration_results.json`
- `tests/fixtures/cowboys_checkpoint/cowboys_albuterol_similarity_1_status.json`
- `docs/MOLECULAR_METRICS.md`
- `docs/verification/molecular_metrics_verification.json`

Modified files: one additive exporter command in `scripts/experiments/run.sh`,
and documentation links in `README.md` and `docs/architecture.md`. No solver,
runner, objective, GP, logging utility or old plot was edited. No optimization,
oracle rescore, training, download, sweep or GPU execution was launched.

Existing real artifacts were exported under:

- `results/cowboys/verification_conda_pcn_20261117T090618/metrics/cowboys_albuterol_similarity_1/run.json` (completed, one BO step)
- `results/cowboys/verification_corrected_pcn_20261117T090506/metrics/cowboys_albuterol_similarity_1/run.json` (partial, initialization only; source failed)
- `results/cowboys/verification_script_pcn_20261117T090734/metrics/cowboys_albuterol_similarity_1/run.json` (completed, one BO step)

The real report is `results/cowboys/metrics/aggregate/experiment_summary.json`,
with CSVs and `plot_manifest.json` alongside it. Fourteen available quality and
internal-diversity figures are in its `figures/` directory. Missing hit metrics
produce skipped entries in the manifest instead of empty/imputed figures.

| Metric/data | Current real artifacts |
|---|---|
| Best-so-far, unique Top-1/10/100 | Computable from original structures and recorded scores, including t=0 |
| Quality AUCs | Computable for the two positive-horizon runs; null for initialization-only run |
| Top-10 internal diversity, novelty, observation duplicate rate | Computable offline with the stated identity/fingerprint policies |
| Unique hits, online diverse hits, scaffolds and their AUCs | Implemented; null in these exports because no task hit threshold was supplied |
| Recorded ranked candidate counts/log-EI | Preserved where present; never treated as additional observations |
| Physical calls, call-based hit/repeated-evaluation rate | Null: no physical-call/cache flags |
| GP predictive moments, calibration, timings, full proposal rates | Null/unavailable: not recorded |
| Independent-seed mean/SD/CI on these artifacts | Excluded: same seed reruns and unrecorded oracle version/sampling policy |
| Approximate Circles | Optional, not implemented; explicitly null |

Verification:

```bash
python -m unittest tests.test_molecular_metrics tests.test_structure tests.test_cowboys_paper_report tests.test_cowboys_fidelity -v
python -m hdbo_benchmark.evaluation export --results-dir results/cowboys/verification_script_pcn_20261117T090734 --planned-bo-steps 1
python -m hdbo_benchmark.evaluation aggregate --results-dir results/cowboys
```

The one-step budget above applies only to the existing verification run; normal
`run.sh` exports use the actual `$MAX_ITER`. The tracked verification JSON records
source hashes and generated paths. Tests include an exact real fixture, byte
preservation and deterministic repeat export, hand-computed prefix metrics,
10+300 indexing, strict Tanimoto boundaries, partial/missing data, schema and
semantic validation, compatibility groups, seed replication and offline plots.
