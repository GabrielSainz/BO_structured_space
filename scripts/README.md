# Existing experiment launch scripts

Run these from the **repository root** after activating the appropriate existing
environment. Root `run*.sh` entry points forward to the corresponding scripts
here, forwarding arguments and exit status without changing the working
directory. Their experiment commands, settings and loops are preserved.

| Scripts under `experiments/` | Existing role |
|---|---|
| `run.sh`, `run_iterations.sh` | COWBOYS PMO sweeps and iteration metrics |
| `run_flow.sh`, `run_flow_2.sh`, `run_iterations_nf.sh` | Two flow implementations and flow iteration runs |
| `run_diffusion.sh`, `run_diffusion2.sh`, `run_diffusion_iterations.sh`, `run_diffusion_guidance_sweep.sh` | Diffusion experiments and guidance sweeps |
| `run_lsbo_poli.sh` | LSBO PMO runs, environment-variable settings and local logs |

Example: `bash scripts/experiments/run_lsbo_poli.sh`. This is an existing sweep,
not a lightweight check. Inspect scripts before running them: some run many
seeds or hundreds of iterations. No sweep was executed during Phase 1.

`batch.sh` and `batch_gpu.sh` remain at the root because SLURM must read their
`#SBATCH` directives directly. Older benchmark-specific batch and compiler
scripts remain beside their protocols in `src/hdbo_benchmark/experiments/`.
