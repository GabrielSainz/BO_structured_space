# Repository architecture

`hdbo_benchmark` uses a `src/` package layout. Root `run.py` forwards to the
canonical experiment CLI. Scientific implementations, older experiment
protocols and import/serialization aliases have distinct roles.

## Canonical modules

| Responsibility | Location |
| --- | --- |
| Local methods and upstream solver factory | `src/hdbo_benchmark/solvers/` and `registry.py` |
| poli problems and representation adapters | `src/hdbo_benchmark/benchmarks/` |
| VAEs, protein AE, one-hot and diffusion components | `src/hdbo_benchmark/generative_models/` |
| Current CLI | `src/hdbo_benchmark/experiments/run.py` |
| Historical benchmark and training protocols | Other directories under `experiments/` |
| Offline molecular metrics, plots and tables | `src/hdbo_benchmark/evaluation/` |
| Historical analysis | Package `results/`, `utils/results/`, `utils/visualization/` |
| Shared GP fitting | `utils/experiments/training_gps.py` |
| Objective/iteration/prediction recording | `utils/logging/` |
| Sweeps and scheduler entry points | Root `scripts/experiments/` and `scripts/slurm/` |
| Environment and configuration index | Root `configs/README.md` |

Locations without a full path are relative to `src/hdbo_benchmark/`; scripts
and configuration paths marked root are relative to the repository.

| Solver key | Canonical implementation |
| --- | --- |
| `cowboys` | `solvers/cowboys.py:COWBOYS` |
| `cowboys_flow` | `solvers/cowboys_flow.py:COWBOYSFlow` |
| `cowboys_flow_2` | `solvers/cowboys_flow_2.py:COWBOYSFlow` |
| `cowboys_diffusion` | `solvers/cowboys_diffusion.py:COWBOYSDiffusion` |
| `lsbo` | `solvers/latent_space_bo.py:LatentSpaceBayesianOptimization` |

The registry also integrates upstream `poli_baselines` methods. GP/acquisition
and flow components remain within their methods; the two flow variants are
distinct. Plotting aliases `dgbo` and `nflow3` mean diffusion and flow variant 2.

## Execution and scientific behavior

The runner loads the poli objective and supervised initialization, attaches its
observer, loads an existing representation model when needed, transforms the
problem, constructs the solver and records progress. PMO uses SELFIES and ten
initial molecules. Budgets, seeds and objective conventions belong to their
existing loaders/runners; older experiment entry points retain older protocols.

Corrected COWBOYS samples in raw VAE coordinates, fits its GP to actually
evaluated structures, selects canonically novel molecules and keeps the
incumbent threshold fixed. Its current default is one chain, 100 minimum pCN
steps and a 50,000-step cap, explicitly raised from 10,000 on 2026-11-20.
Diagnostic objective calls are disabled by default. Read
[COWBOYS_IMPLEMENTATION.md](COWBOYS_IMPLEMENTATION.md) before changing these
settings; [PMO_DIAGNOSTICS.md](PMO_DIAGNOSTICS.md) explains oracle/runtime limits.

## Outputs and resources

- COWBOYS writes `results/cowboys/<sufix>/` with best-performance NPY, aligned
  history NPZ, status/iteration JSON and available GP prediction sidecars.
- Other non-diffusion methods use
  `results/new_vae_10_chain_100_steps_with_stoch_sampling_<sufix>/`; diffusion
  uses `results/diffusion/` with parameter-derived subdirectories.
- Offline exports live in a selected run directory's `metrics/`. See
  [MOLECULAR_METRICS.md](MOLECULAR_METRICS.md) for provenance, schema, missing
  data and aggregation rules. Preflights are excluded from ordinary discovery.
- Root `results/`, `logs/`, `wandb/`, generated report directories and local
  scratch/agent files are ignored. Package `src/hdbo_benchmark/results/` is
  analysis source and must remain tracked.
- Datasets/checkpoints are in root `data/` and package `assets/`. Molecular VAE
  lookup checks `data/trained_models/training_vae_on_zinc_250k` before package
  assets. Preserve bytes, precedence and legacy serialized class references.

## Compatibility and checks

Edit canonical package modules, not forwarding aliases. Root `run.py`, installed
aliases under `src/`, and old loader/transform module aliases remain compatibility
boundaries. Historical training/benchmark commands are not lightweight checks.
Submit actual SLURM files from the repository root so their headers are read.

Use the established Python 3.10 environment. On this Windows machine, Conda
activation is necessary: `conda run -n hdbo_benchmark ...` passed a real bounded
COWBOYS check, while invoking its Python directly from the base shell could
fail in BLAS/threadpool handling. The original comparison is recorded in
[verification/phase1.json](verification/phase1.json); the successful corrected
check is documented in [COWBOYS_IMPLEMENTATION.md](COWBOYS_IMPLEMENTATION.md).
Clean-install and full historical serialization guarantees remain open.

```bash
python -m unittest tests.test_structure -v
python run.py --help
python -m hdbo_benchmark.evaluation --help
```

Integration tests may execute real objectives and require optional dependencies;
select them deliberately. Current unresolved work is in
[TECHNICAL_DEBT.md](TECHNICAL_DEBT.md).
