# Post-migration repository cleanup

This cleanup supersedes the compatibility-launcher inventory in the historical
Phase 1 migration report. It changes file organization and documented entry
points, not scientific behavior. Work remains on `refactor/agent-repo`.

## Removed redundant files

Seventeen tracked files were removed:

- Three root method aliases: `cowboys.py`, `cowboys_flow.py`,
  `cowboys_flow_2.py`. The installed aliases under `src/` remain for old imports
  and pickle class references; the scientific implementations remain in
  `src/hdbo_benchmark/solvers/`.
- Ten root `run*.sh` forwarding scripts. Their actual experiment bodies remain
  under `scripts/experiments/`.
- Two report aliases: `reports/plot_iteration_metrics.py` and
  `reports/table_results.py`. Use the package modules below.
- Root `test_main_entry_point.py`. The actual integration tests remain under
  `tests/`; default pytest discovery already uses that directory.
- Inactive `gitignore` (without a leading dot). It contained only `*.npy` and
  was never interpreted by Git; no new global ignore rule was introduced.

One repeated processed-data ignore pattern was removed from `.gitignore` with
no change in effective ignore behavior.

## Canonical commands

Run from the repository root after the existing editable installation:

| Task | Current entry point |
|---|---|
| Main experiment CLI | `python run.py ...` or `python -m hdbo_benchmark.experiments.run ...` |
| Existing COWBOYS sweep | `bash scripts/experiments/run.sh` |
| Other method sweeps | `bash scripts/experiments/<script>.sh` |
| Iteration plots | `python -m hdbo_benchmark.evaluation.plot_iteration_metrics ...` |
| Existing table script | `python -m hdbo_benchmark.evaluation.table_results` |
| Integration tests | `python -m pytest tests/test_main_entry_point.py ...` |
| Lightweight checks | `python -m unittest tests.test_structure -v` |
| SLURM submission | `sbatch scripts/slurm/batch.sh ...` or `sbatch scripts/slurm/batch_gpu.sh ...` |

The two SLURM files moved intact from the root. Submit the actual files so SLURM
reads their existing directives. Their arguments, resource requests and
working-directory assumptions remain unchanged. Shell comments and report
docstrings now show canonical commands; executable experiment logic is unchanged.

## Retained deliberately

- Root `run.py` is still referenced by existing experiment and scheduler scripts.
- Installed aliases under `src/` and the old loader/transform aliases provide
  import/serialization compatibility without duplicate scientific code.
- All methods, older experiments, historical VAE implementations, datasets,
  checkpoints, generated results, figures, logs and chemistry caches remain.
- Environment files and dependency metadata still have active consumers and
  working-directory assumptions. They are not safe duplicate-file deletions.
- Historical audit/verification evidence and inactive website workflow are kept
  as historical material; absence of a current caller alone does not establish
  that a historical artifact is disposable.

## Generated-cache limitation

Eighty-six `.pyc` files were identified in root/source/test/report cache
locations. Automatic approval review blocked the cache-deletion command with
“blocked by policy”; those ignored files remain. Subsequent verification uses
`PYTHONDONTWRITEBYTECODE=1` to avoid creating more cache files.

## Verification

- Three structural tests passed with `python -m unittest tests.test_structure -v`.
- Root and package experiment CLI help, and canonical plotting CLI help, passed
  in the existing Python 3.10 research environment.
- The three installed method aliases and their old pickle class references
  resolved; retained runner and loader aliases also resolved.
- All solver package sources match the pre-cleanup Git revision. Both moved
  SLURM files match their originals, allowing checkout line-ending differences.
- All twelve shell launchers passed Git Bash syntax parsing on normalized text;
  Python sources compiled without execution; documentation links resolved.
- All 1,902 recorded resource/output files retained their sizes and modification
  times. This is a metadata check, not a complete content hash of historical data.
- `git diff --check` passed. No experiments, training, dependency installation,
  commits, pushes or merges were performed.
