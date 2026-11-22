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

## First job on Hendrix

Use `scripts/slurm/hendrix_cpu.sh` for a single CPU run. It uses the Hendrix
`gpu` partition and requests one CPU, 6 GB RAM and 15 minutes. GPU allocation
requires a separate resource request. [DIKU Slurm guide](https://diku-dk.github.io/wiki/slurm-cluster#using-slurm)

In your Hendrix shell, use your existing checkout or clone the current branch
if you need a checkout.

```bash
git clone --depth 1 --branch refactor/agent-repo https://github.com/GabrielSainz/BO_structured_space.git ~/ROTLSC
```

Make Conda available on Hendrix. If you use a cluster module, find it with
`module avail` and load it before submitting. The launcher uses the Linux
`hdbo_benchmark` Conda environment by default. Check it with
`conda run -n hdbo_benchmark python --version`.
If it needs to be created, follow the Python 3.10 and molecular PMO installation
steps in the [repository README](../README.md#installation). Set `HDBO_CONDA_ENV`
when submitting if the environment has a different name.

The 128-dimensional Zinc VAE checkpoint is tracked in the repository under
`data/trained_models/training_vae_on_zinc_250k/`. Use a checkout containing the
checkpoint and the launcher. If the launcher is still a local change, copy it
from your local PowerShell after creating the remote checkout. These examples
assume the remote checkout is `~/ROTLSC`. Substitute your KU ID and path as
needed.

```powershell
scp scripts/slurm/hendrix_cpu.sh KU_ID@hendrixgate:ROTLSC/scripts/slurm/
```

From the repository root on Hendrix, submit albuterol similarity with seed 1.
This runs ten initial evaluations and one COWBOYS iteration with W&B disabled.

```bash
cd ~/ROTLSC
mkdir -p slurm_logs
sbatch scripts/slurm/hendrix_cpu.sh albuterol_similarity cowboys 128 1 1
```

Use the job ID printed by `sbatch` in place of `JOB_ID` below. The log appears
when the job starts. The COWBOYS result folder contains the status JSON, best
curve NPY, history NPZ and iteration metrics. Check for `"status": "completed"`
and `"completed_iterations": 1` in the status JSON.

```bash
squeue -j JOB_ID
sacct -j JOB_ID --format=JobID,State,ExitCode,Elapsed,MaxRSS
tail -n 50 slurm_logs/hendrix-JOB_ID.log
cat results/cowboys/hendrix-JOB_ID/cowboys_albuterol_similarity_1_status.json
```

The files stay in the Hendrix checkout. To copy the run and log back, run this
from your local repository in PowerShell.

```powershell
New-Item -ItemType Directory -Force results/cowboys, slurm_logs | Out-Null
scp -r KU_ID@hendrixgate:ROTLSC/results/cowboys/hendrix-JOB_ID results/cowboys/
scp KU_ID@hendrixgate:ROTLSC/slurm_logs/hendrix-JOB_ID.log slurm_logs/
```

If Slurm restarts the job, the log records the new run suffix
`hendrix-JOB_ID-rN`. For other registered CPU solvers, submit the same launcher
with `FUNCTION SOLVER DIMENSIONS BO_STEPS SEED` and the appropriate environment.
The runner keeps its [solver-specific output directories](../docs/architecture.md#outputs-and-resources).
