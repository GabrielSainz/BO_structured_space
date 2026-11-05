# Migration and verification record

Phase 1, 2026-11-05. Work remains uncommitted on `refactor/agent-repo`.

## Implemented structural changes

The original repository combined a src-layout package with root method files,
a root experiment runner, root shell sweeps, local analysis in reports, and a
root integration-test module. Canonical methods, benchmark adapters, the runner
and current local analysis now live in the existing `hdbo_benchmark` package.
Configuration values remain at their original sources of truth, indexed under
`configs/`. See the complete [20-file migration map](MIGRATION_MAP.md).

Additional changes:

- Four `src/` aliases declared in `setup.cfg` preserve installed root imports.
- Existing solver class exports are lazy to avoid unrelated scientific imports.
- Pytest defaults to `tests/`, avoiding duplicate collection of the root shim.
- The generated-output ignore rule is anchored to `/results/`, exposing package
  analysis source to ordinary file searches.
- Internal references to migrated loader/transform modules use canonical paths.
  The registry imports COWBOYS variants through their package modules. The test
  imports the canonical runner; eight older runner/compiler files now reference
  `solvers.registry`. The five local algorithm bodies needed no import changes.

## Source and artifact preservation

- Every original tracked text file was snapshotted before migration. 150 files
  match their originals after only declared import substitutions, comparing
  normalized text line endings. Five explicitly edited organizational files are
  excluded: README, two packaging files, `.gitignore` and solver package exports.
- All five local algorithm implementations remain intact. Registry names,
  settings, benchmark definitions, acquisition code, hyperparameters, model
  selection, seeds and initialization code are unchanged.
- A snapshot of 1,902 resource/output file paths, sizes and modification times
  taken after the baseline is unchanged after migration. This includes four
  newly generated baseline files. This is a metadata check, not a full hash of
  more than ten GB of historical data.
- Baseline/after outputs use fresh distinct suffixes. Existing results, datasets,
  checkpoints, analysis and obsolete-looking implementations were retained.
- No model training, pretrained downloads, sweeps, GPU runs, commits, pushes or
  merges were performed. Both COWBOYS attempts reach the existing GP-fitting call
  and fail at the same external BLAS operation.

## COWBOYS before/after attempt

Environment: existing Conda `hdbo_benchmark`, Python 3.10.20. Observed versions:
NumPy 1.26.4, PyTorch 2.10.0, BoTorch 0.16.1, GPyTorch 1.15.2, SciPy 1.15.2,
threadpoolctl 3.6.0, poli-core 1.3.1, poli-baselines 1.0.2.dev1.

Both attempts used `HDBO_TORCH_DEVICE=cpu`, `WANDB_MODE=disabled`,
`HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, `OMP_NUM_THREADS=1`,
`MKL_NUM_THREADS=1`, and `PYTHONDONTWRITEBYTECODE=1`, with a 120-second process
limit. Only the output `--sufix` differs to avoid overwriting the baseline.

```bash
python run.py --function-name albuterol_similarity --solver-name cowboys --n-dimensions 128 --max-iter 1 --seed 1 --no-strict-on-hash --wandb-mode disabled --tag phase1-structure --sufix phase1_before_20261105
python run.py --function-name albuterol_similarity --solver-name cowboys --n-dimensions 128 --max-iter 1 --seed 1 --no-strict-on-hash --wandb-mode disabled --tag phase1-structure --sufix phase1_after_20261105
```

The loader uses ten initial evaluations; reducing to five would change
initialization. One additional iteration was requested to bound cost. Both
attempts loaded the existing VAE and reached GP fitting. Both failed with
`OSError: [WinError -1066598273] Windows Error 0xc06d007f` in
`threadpoolctl.get_num_threads`, called through BoTorch/SciPy. The cause was not
investigated or repaired. Baseline elapsed time: 29.57 s; after: 15.15 s. These
timings are not a performance comparison.

| Invariant | Before | After | Result |
|---|---|---|---|
| Exit code | 1 | 1 | Same pre-existing failure |
| Initial latent history | `(10, 128)` | `(10, 128)` | Arrays exactly equal |
| Initial objective history | `(10, 1)` | `(10, 1)` | Arrays exactly equal |
| Requested additional iterations | 1 | 1 | Preserved |
| Completed iterations | 0 | 0 | No full optimization regression established |
| Best recorded value | 0.32941176470588235 | 0.32941176470588235 | Exactly equal |
| Files | NPY, history NPZ, metrics JSON, status JSON | Same four names | Outputs produced |
| Content | Initial history, empty BO series, failed status | Same | All arrays and JSON exactly equal |

Artifacts remain under root
`results/new_vae_10_chain_100_steps_with_stoch_sampling_phase1_{before,after}_20261105/`.
[Machine-readable evidence](verification/phase1.json) records commands, shapes,
status and artifact hashes. The [baseline log](verification/phase1_baseline.log)
and [after log](verification/phase1_after.log) preserve output with local machine
paths replaced by placeholders. NPZ container bytes may differ because of ZIP
timestamps; their arrays were compared exactly.

## Other checks

| Check | Outcome / limitation |
|---|---|
| In-memory Python compilation | 131 files compile without executing analysis/training code |
| `python -m unittest tests.test_structure -v` | Three tests pass: lightweight namespace imports, packaged methods, shell targets |
| Legacy module imports | Eight old/canonical module pairs resolve to the same object, preserving private helpers and module state |
| Local registry | All five local classes resolve without running optimization |
| Legacy serialization references | Three old root-module pickle class references resolve; full old objects not tested |
| Existing package exports | Diffusion and LSBO class exports resolve lazily |
| CLI help | Root runner help succeeds before/after; package runner and both old/new plotting CLI forms pass; source comparison confirms unchanged Click options |
| Package discovery | setuptools discovers 29 packages including new namespaces; four `src/` aliases import outside checkout with `PYTHONPATH=src` |
| Shell syntax | 20 current/compatibility launchers parse using Git Bash `-n` on normalized text; no sweep executed |
| `git diff --check` | Passed; ordinary LF/CRLF checkout warnings remain |
| Existing pytest/tox integrations | Not run: pytest absent in the research environment; integration selection executes real benchmarks and optional stacks |
| Wheel build | `pip wheel --no-deps --no-build-isolation --no-index` fails during metadata with local setuptools 80.10.2; identical `AttributeError` reproduced from original metadata in a temporary directory |

The declared backend is `setuptools<=68`. A supported isolated build was not
performed because it would require obtaining another build environment. No
wheel installation is claimed. Package discovery and outside-checkout imports
validate layout, not distribution completeness or external data availability.
The existing editable installation was retained unchanged.

## Documented problems and future proposals

[Technical debt](TECHNICAL_DEBT.md) separates user-reported zero objectives,
pre-existing Windows runtime/build failures, missing integration coverage,
legacy APIs, duplicated scientific code, configuration drift and provenance
risks. None was scientifically fixed here.

[Roadmap](ROADMAP.md) proposes five milestones: benchmark diagnostics; unified
experiment management; algorithm verification; benchmarking/analysis; and future
methods including GenTuRBO-inspired local/global search, acquisition-guided
generative optimization and protein/surrogate research. No new solver,
configuration framework, experiment runner or scientific feature was added.

Documentation includes README, AGENTS, architecture, audit, complete original
file inventory, technical debt, roadmap, configuration/script indexes and this
migration record. Remaining compatibility limits include full serialized
objects, external `__file__` consumers, standalone wheel/data paths, older
optional environments and expensive historical workflows. These require
separately scoped checks, not a claim that full benchmarks passed.
