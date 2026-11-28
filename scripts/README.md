# Existing experiment launch scripts

Objective diagnostics are under `scripts/diagnostics/`; see
[the PMO report](../docs/PMO_DIAGNOSTICS.md) for their commands and scope.

Run these from the **repository root** after activating the appropriate existing
environment. Invoke the scripts here directly; redundant root `run*.sh`
forwarders have been removed. The corrected COWBOYS launcher is described in
[the implementation notes](../docs/COWBOYS_IMPLEMENTATION.md); other historical
launchers retain their settings.

| Scripts under `experiments/` | Existing role |
|---|---|
| `run.sh` | Corrected single-chain COWBOYS, preflight checks, fresh outputs and paper comparison |
| `run_iterations.sh` | Historical COWBOYS iteration-metric sweep |
| `run_flow.sh`, `run_flow_2.sh`, `run_iterations_nf.sh` | Two flow implementations and flow iteration runs |
| `run_diffusion.sh`, `run_diffusion2.sh`, `run_diffusion_iterations.sh`, `run_diffusion_guidance_sweep.sh` | Diffusion experiments and guidance sweeps |
| `run_lsbo_poli.sh` | LSBO PMO runs, environment-variable settings and local logs |

Example: `bash scripts/experiments/run_lsbo_poli.sh`. This is an existing sweep,
not a lightweight check. Inspect scripts before running them: some run many
seeds or hundreds of iterations.

SLURM entry points are `scripts/slurm/batch.sh` and
`scripts/slurm/batch_gpu.sh`. Submit them directly from the repository root with
`sbatch scripts/slurm/batch.sh ...` (or the GPU variant), passing the same
function, solver, dimensions, iterations, tag and environment arguments as
before. Their `#SBATCH` directives and working-directory assumptions are
unchanged. Older benchmark-specific batch and compiler scripts remain beside
their protocols in `src/hdbo_benchmark/experiments/`.

`scripts/slurm/hendrix_afgg.sh` runs sequential AFGG with SeqFlow on one GPU
per task. The [AFGG protocol](../docs/AFGG_SEQFLOW.md) gives separate smoke and
five-task array commands.
