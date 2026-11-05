# COWBOYS and structured Bayesian optimization

Research code for Bayesian optimization over discrete and structured sequences
using **poli** and **poli-baselines**, including molecular and protein objectives,
latent representations, Gaussian processes and generative candidate proposals.
The existing Python package is `hdbo_benchmark`.

Phase 1 reorganizes existing implementations and entry points. It does not
change algorithms, defaults, evaluation budgets or benchmark definitions.
Scientific diagnostics and new research methods are future work.

## Existing methods

| Local method / CLI key | Implementation |
|---|---|
| COWBOYS ? `cowboys` | `src/hdbo_benchmark/solvers/cowboys.py` |
| Flow COWBOYS ? `cowboys_flow` | `src/hdbo_benchmark/solvers/cowboys_flow.py` |
| Second flow variant ? `cowboys_flow_2` | `src/hdbo_benchmark/solvers/cowboys_flow_2.py` |
| Diffusion COWBOYS ? `cowboys_diffusion` | `src/hdbo_benchmark/solvers/cowboys_diffusion.py` |
| LSBO ? `lsbo` | `src/hdbo_benchmark/solvers/latent_space_bo.py` |

The [registry](src/hdbo_benchmark/solvers/registry.py) also integrates upstream
directed evolution, hill climbing, genetic algorithm, CMA-ES, random/coordinate
LineBO, BAxUS, TuRBO, Hvarfner BO, ALEBO, Bounce, PR, SAASBO and LaMBO2. Optional
methods require their original environments. Factory-only vanilla BO variants
and older Ax/latent wrappers are retained; see the [audit](docs/REPOSITORY_AUDIT.md).

## Installation

Use the established Python 3.10 environment. From the repository root, the
existing setup procedure is:

```bash
conda create -n hdbo_benchmark python=3.10
conda activate hdbo_benchmark
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install gauche selfies
python -m pip install -e .
```

For molecular PMO objectives, the existing chemistry setup is:

```bash
conda install -c conda-forge "rdkit<2024.03" -y
python -m pip install --upgrade PyTDC "huggingface_hub<1"
python -c "import rdkit; print(rdkit.__version__)"
python -c "from tdc import Oracle; print('tdc ok')"
```

These are retained setup instructions, not a newly validated dependency lock.
The build system declares `setuptools<=68`; a local no-isolation build with
setuptools 80.10.2 failed with the original and migrated metadata. The available
Windows research environment also fails during COWBOYS GP fitting. See
[verification and limits](docs/MIGRATION_VERIFICATION.md) before interpreting
these commands as a verified clean installation or successful experiment.
Colab remains experimental because of chemistry/TDC and model-stack dependencies.

Existing Conda definitions, extras and runtime settings are indexed in
[configs/README.md](configs/README.md). Keep preprocessing separate from research
execution; its environment targets Python 3.8/TorchDrug. No environments or
scientific dependencies were installed or upgraded during this restructuring.

## Run existing entry points

Run from the repository root after installation:

```bash
python run.py --help
python -m hdbo_benchmark.experiments.run --help
```

Both invoke the same existing CLI. An example of the bounded CPU check used in
Phase 1 (Bash) is:

```bash
HDBO_TORCH_DEVICE=cpu OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python run.py --function-name albuterol_similarity --solver-name cowboys --n-dimensions 128 --max-iter 1 --seed 1 --no-strict-on-hash --wandb-mode disabled --tag phase1-structure --sufix your_unique_check_name
```

In PowerShell, set `$env:HDBO_TORCH_DEVICE = 'cpu'`,
`$env:OMP_NUM_THREADS = '1'`, and `$env:MKL_NUM_THREADS = '1'`, then invoke the
same `python run.py ...` command. The existing PMO loader uses ten initial
molecules; `--max-iter 1` requests one additional optimization iteration. The
attempt failed before completing that iteration in the recorded environment;
partial output creation and identical initial histories were checked.
Use a fresh `--sufix` to avoid reusing existing output filenames.

The previous long-run command remains valid in a working scientific environment:

```bash
python run.py --function-name albuterol_similarity --solver-name cowboys --n-dimensions 128 --max-iter 300 --seed 1 --no-strict-on-hash --wandb-mode disabled --tag local-test --sufix unique_run_name
```

It loads an existing molecular VAE checkpoint; it is not a lightweight check.
Full sweeps run through `bash scripts/experiments/run.sh`. Flow, diffusion and LSBO launchers are described
in [scripts/README.md](scripts/README.md). Use Git Bash/WSL on Windows for shell
scripts. Inspect budgets before launching sweeps. SLURM batch files are under `scripts/slurm/`; submit them from the repository
root. Historical package experiment entry points retain their locations.

## Structure and outputs

```text
src/hdbo_benchmark/
  solvers/            local methods and external solver registry
  benchmarks/         poli loading and representation adapters
  generative_models/  existing VAEs, protein AE, one-hot and diffusion
  experiments/        current CLI and historical/training protocols
  evaluation/         current local plotting and tables
  results/            historical analysis source (not generated results)
  utils/              existing shared/domain helpers
  assets/             existing package resources
  data_preprocessing/ existing dataset preparation
scripts/experiments/  existing shell experiment configurations
scripts/slurm/        scheduler entry points
configs/              configuration index (no new config framework)
tests/                lightweight structure and existing integration checks
docs/                 architecture, audit, verification and future work
data/                 datasets and pretrained checkpoints
results/              generated optimization results
reports/              figures and legacy report entry points
```

Root `run.py` remains the main convenience launcher. Redundant root method,
sweep, report and test launchers have been removed. Installed module aliases in
`src/` and old loader imports remain for import/serialization compatibility.
Edit canonical implementations under `src/hdbo_benchmark/`. See the
[cleanup record](docs/CLEANUP.md) for replacement commands.

The current runner saves best-performance `.npy`, available history `.npz`,
status JSON and requested iteration-metric JSON. Non-diffusion output goes to
`results/new_vae_10_chain_100_steps_with_stoch_sampling_<sufix>/`; diffusion
outputs use the existing parameter-derived subdirectories under
`results/diffusion/`. Filenames include solver, function and seed. Older runners
retain their own data/W&B output formats. Existing data and historical results
were not moved or deleted.

For local analysis:

```bash
python -m hdbo_benchmark.evaluation.plot_iteration_metrics --help
python -m hdbo_benchmark.evaluation.table_results
```

The table module retains its existing hardcoded result selection and top-level
printing. W&B settings remain in `utils/constants.py`; use the existing
`--wandb-mode disabled` for local verification.

## Checks and documentation

Dependency-free structural checks:

```bash
python -m unittest tests.test_structure -v
```

Existing integration tests now live under `tests/test_main_entry_point.py`.
With pytest and the relevant scientific dependencies installed, select them
explicitly, for example `python -m pytest tests/test_main_entry_point.py -m hdbo_base`.
This runs real benchmark combinations and is not a cheap smoke test. The redundant root test path has been removed; pytest discovers `tests/`.
The existing tox environments remain available and can install dependencies/run
benchmarks. They were not launched during Phase 1.

- [Architecture and compatibility](docs/architecture.md)
- [Repository audit](docs/REPOSITORY_AUDIT.md) and [complete original file inventory](docs/FILE_INVENTORY.md)
- [Migration map and verification](docs/MIGRATION_VERIFICATION.md)
- [Known technical debt](docs/TECHNICAL_DEBT.md), including user-reported zero-valued objectives
- [Future milestones](docs/ROADMAP.md): benchmark diagnostics, unified experiments,
  algorithm verification, benchmarking/analysis, and new research methods
- [Codex guidance](AGENTS.md)
