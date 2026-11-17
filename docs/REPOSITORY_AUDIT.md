# Repository audit — Phase 1

Inventory recorded before moving source files on 2026-11-05, on
`refactor/agent-repo`, from an initially clean working tree. Scope: architecture
and compatibility only; no scientific validation or zero-score diagnosis.
The companion [file inventory](FILE_INVENTORY.md) lists every originally tracked
file, including analysis sources hidden by `.gitignore`.

## Inventory and migration design

Locations in the second column describe the original checkout. “Unchanged” is
an intentional decision to preserve public imports, resources, or provenance.

| Component | Current location | Purpose | Dependencies | Proposed location |
|---|---|---|---|---|
| COWBOYS | `cowboys.py` | GP, fingerprint acquisition and MCMC candidate proposals | BoTorch, GPyTorch, gauche, RDKit, SELFIES, poli-baselines, VAE utilities | `src/hdbo_benchmark/solvers/cowboys.py`; old module alias |
| Flow variants | `cowboys_flow.py`, `cowboys_flow_2.py` | Two distinct COWBOYSFlow implementations; embedded RealNVP, mixture proposals and diagnostics | Same scientific stack, PyTorch | `src/hdbo_benchmark/solvers/`; both retained separately; old aliases |
| Diffusion COWBOYS | `src/hdbo_benchmark/solvers/cowboys_diffusion.py` | Diffusion proposals, critic, acquisition guidance, GP fitting | Latent diffusion, VAE, BoTorch, gauche, poli-baselines | Unchanged |
| LSBO | `src/hdbo_benchmark/solvers/latent_space_bo.py` | Latent-space GP/EI optimization | BoTorch, GPyTorch, SELFIES, RDKit, poli-baselines | Unchanged |
| Solver selection | `src/hdbo_benchmark/utils/experiments/load_solvers.py` | Names, environment mapping, constructor settings and problem-to-solver adaptation | Local solvers, poli-baselines, constants | `src/hdbo_benchmark/solvers/registry.py`; old alias |
| External methods | Branches in `load_solvers.py` | Directed evolution, hill climbing, genetic algorithm, CMA-ES, random/coordinate LineBO, BAxUS, TuRBO, Hvarfner BO, ALEBO, Bounce, PR, SAASBO, LaMBO2 | Optional poli-baselines environments | Registry only; implementations remain upstream |
| Additional registry branches | `load_solvers.py` | `vanilla_bo`, `vanilla_bo_with_lognormal_prior`; supported by factory but absent from `SOLVER_NAMES` | poli-baselines, GPyTorch | Preserve distinction in registry |
| Older wrappers | `utils/latent_space_solver.py`, `utils/ax/` under package | Generic latent wrapper and AxSolver adapter | poli-baselines, Ax, generative model | Unchanged |
| Benchmark loaders | `utils/experiments/load_problems.py` | PMO, RASP, FoldX, Ehrlich and pest-control-equivalent setup, initial data, budgets, observers | poli, TDC compatibility, metadata, W&B | `src/hdbo_benchmark/benchmarks/load_problems.py`; old alias |
| Representation adapters | `utils/experiments/problem_transformations.py` | Latent/one-hot black-box wrappers and diagnostic forwarding | poli, generative models, normalization | `src/hdbo_benchmark/benchmarks/problem_transformations.py`; old alias |
| GP utilities | `utils/experiments/training_gps.py`; inline GP code in solvers | Exact GP fitting and point filtering; method-specific kernels | GPyTorch, PyTorch, BoTorch | Unchanged; do not extract method internals |
| Acquisition functions | Inline in five solvers; external baselines | EI/log-EI/log-PI and method-specific wrappers | BoTorch, local solver state | Unchanged; no independent acquisition package invented |
| Generative components | `src/hdbo_benchmark/generative_models/` | Molecular/Mario VAEs, RNN VAE, historical VAE, protein AE, one-hot adapter, latent diffusion and dataset helpers | PyTorch, Lightning, ESM, RDKit, data utilities | Unchanged; flow components stay inside their methods |
| Main experiment CLI | `run.py` | Click arguments, construction, iteration loop, checkpoint and metric writes | Loaders, registry, observer, diagnostic utilities | `src/hdbo_benchmark/experiments/run.py`; root CLI/module alias |
| Historical experiment entry points | `src/hdbo_benchmark/experiments/benchmark_on_{pmo,rasp,foldx,ehrlich,toy_objective}/` | Older protocols and batch/compiler scripts | Legacy poli APIs, W&B, model/data files | Unchanged except migrated import references |
| Training entry points | `experiments/training_*` under package | Existing VAE, protein AE, diffusion training commands | Training utilities, datasets, checkpoints | Unchanged; not executed |
| Batch launchers | Root `run*.sh` | Existing method sweeps, seeds, budgets and parameter settings | Bash, root CLI, optional Conda/SLURM | `scripts/experiments/`; root forwarding scripts |
| Scheduler entry points | Root `batch.sh`, `batch_gpu.sh`; experiment-local batch scripts | SLURM resource requests and commands | SLURM, Conda | Unchanged to preserve scheduler header handling |
| Configuration | Root `environment*.yml`, `envs/*.yml`, CLI defaults and shell settings | Existing environment and experiment configuration | Conda, pip, Click, shell | Existing locations preserved; indexed by `configs/README.md` |
| Dataset handling | `data_preprocessing/zinc250k/`, `utils/data/`, `utils/selfies/`, `utils/mario/`, `utils/download/` | Download/conversion/tokenization/loading | TorchDrug, pandas, NumPy, PyTorch, SELFIES, requests | Unchanged; relative-depth paths retained |
| Checkpoints and metadata | `data/`, package `assets/` | Trained models, molecular metadata, Mario data, PDBs, ESM embeddings | Factories and loaders | Unchanged, including duplicate/historical checkpoints |
| Current local analysis | `reports/plot_iteration_metrics.py`, `reports/table_results.py` | Plotting/aggregation and five-seed tables | NumPy, matplotlib | `src/hdbo_benchmark/evaluation/`; old entry points retained |
| Historical analysis | Package `results/`, `utils/results/`, `utils/data/experiments/` | W&B download, paper/website tables, comparisons, plots | W&B, pandas, plotting stack | Unchanged; source is tracked despite broad ignore rule |
| Visualization | Package `utils/visualization/`, `utils/selfies/visualization.py` | Figures, molecules, Mario rendering | matplotlib, RDKit, image libraries | Unchanged |
| Results and runtime outputs | Root `results/`, `reports/figures/`, `reports/iteration_plots/`, `logs/`, `wandb/`, `oracle/` | Historical trajectories, figures, logs and chemistry caches | Existing execution/analysis scripts | Unchanged; no deletion or overwriting |
| Existing tests | `test_main_entry_point.py` | Environment-marked integration tests across benchmark/solver combinations | pytest, poli, optional solver stacks | `tests/test_main_entry_point.py`; root compatibility entry point |
| Sanity experiment | `src/hdbo_benchmark/sanity_checks/bayesian_optimization_in_2d.py` | BO demonstration | poli and poli-baselines | Unchanged; not treated as a cheap unit test |
| Notebooks | None found (tracked or local `.ipynb`) | No notebook organization required | — | No empty directory created |
| Packaging/dependencies | `pyproject.toml`, `setup.cfg`, `requirements*.txt`, `tox.ini`, Conda YAMLs | Existing src-layout build, extras and checks | setuptools, pip, Conda, tox | Preserve dependency settings; add installable legacy module shims |
| Documentation | Root README, preprocessing/experiment readmes, `data/rfp_pdbs/readme.md` | Setup, old experiments, data context | — | Update README; add architecture, roadmap, debt and migration notes |
| Automation | `.github/workflows/`, `.pre-commit-config.yaml`, `workflows/website.yml` | CI and historical website deployment definition | GitHub Actions, tox, Hugo | Unchanged; inactive website workflow not activated |
| Other organizational residue | `gitignore` alongside `.gitignore`, root `.syntax.pyc`, caches | Inactive ignore fragment and generated bytecode | — | Retained; no cleanup of historical/unclear files |

## Architecture findings and risks

- Root implementations were unavailable in a normal wheel using only src package
  discovery. Moving them into the existing `solvers` package removes that source
  layout inconsistency. Keep the package name and existing two solver paths.
- Solver files intentionally mix GP fitting, acquisition, sampling, decoding and
  diagnostics. Extracting these would exceed a safe file migration.
- The main CLI mixes execution and saving, and older runners implement independent
  protocols. Move the CLI as a whole; do not unify those protocols in this phase.
- `solvers/__init__.py` eagerly imports diffusion and LSBO. Adding root COWBOYS to
  that package must not force unrelated solver imports: retain the old exported
  class names through lazy lookup.
- Shared normalization, GP training and VAE helpers have public utility paths;
  leaving them in place avoids needless aliases and depth-dependent path edits.
- There are three analysis organizations: root reports, package results and
  utilities. Package the two current local scripts; preserve old paper/website
  scripts with their original paths and import-time behavior.
- Bare `results/` ignores package analysis sources as well as runtime outputs.
  Anchor the runtime ignore rule to `/results/` so future source is discoverable.
- Checkpoints are duplicated under data and package assets. Factories have
  explicit search order; moving or consolidating them could change model choice.
- `ROOT_DIR` and several data paths count parents of `__file__`. Keep those
  modules and their data at their current depths. CLI outputs and shell scripts
  remain relative to the working directory; use the repository root.
- `envs/*.yml` uses editable `..[extra]`, while root `environment.yml` uses `.`.
  Preserve these files and document their working-directory assumptions instead
  of creating conflicting copies or a new configuration system.
- Root module aliases must expose private helpers and module state as well as
  public classes. New pickles use canonical package paths; old imports/pickles
  require retained aliases. Full historical serialized-object compatibility
  cannot be inferred from import checks alone.
- SLURM reads directives before running a shell; replacing scheduler entry
  points with ordinary forwarding scripts could change resource allocation.
  Keep scheduler scripts where they are.
- Naming inconsistencies (`sufix`, `uncommited`, `vae_selfies-old`, `_2`) are
  preserved because they may be public flags, imports or distinct experiments.

## Baseline plan

The available `hdbo_benchmark` Conda environment uses Python 3.10.20; the default
shell Python is 3.13.12. The research environment has the scientific libraries
but no pytest. `python run.py --help` succeeds before migration. The PMO loader
hardcodes ten initial molecules, so reducing initialization to five would alter
the existing protocol. Attempt one CPU COWBOYS iteration, seed 1, 128 latent
dimensions, `albuterol_similarity`, W&B disabled, without training or downloads,
with a 120-second process timeout. Record the outcome even if it fails and do
not diagnose the scientific cause. See [verification](MIGRATION_VERIFICATION.md)
for final evidence and limitations.

## Historical storage inventory

Before baseline execution: `results/` contained 1,647 files (1,028,071,525 bytes),
`data/` 35 files (10,204,920,804 bytes), `reports/` 82 files (21,639,774 bytes,
including analysis source/cache), `logs/` 125 files (10,604,754 bytes), `wandb/`
8 files (36,309 bytes), and `oracle/` 4 files (83,147,391 bytes). These are storage
counts, not a scientific inspection of objective values. The user-reported
all-zero results are recorded as known issues, without investigation.
