# Applying COWBOYS to the benchmark of High-Dimensional Bayesian Optimization for discrete sequence optimization

[![Link to Project website](https://img.shields.io/badge/GitHub-Project_Website-100000?logo=github&logoColor=white)](https://machinelearninglifescience.github.io/hdbo_benchmark)
[![Link to Project website](https://img.shields.io/badge/GitHub-poli_docs-100000?logo=github&logoColor=white)](https://machinelearninglifescience.github.io/poli-docs)
[![Tests on hdbo (conda, python 3.10)](https://github.com/MachineLearningLifeScience/hdbo_benchmark/actions/workflows/tox-lint-and-pytest.yml/badge.svg)](https://github.com/MachineLearningLifeScience/hdbo_benchmark/actions/workflows/tox-lint-and-pytest.yml)

This repository contains the code to apply COWBOYS to a benchmark of **high-dimensional Bayesian optimization** over discrete sequences using [poli](https://github.com/MachineLearningLifeScience/poli) and [poli-baselines](https://github.com/MachineLearningLifeScience/poli-baselines).


### Recommended setup

The most reliable setup is a fresh Conda environment with Python 3.10.

```bash
conda create -n hdbo_benchmark python=3.10
conda activate hdbo_benchmark
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install gauche selfies
python -m pip install -e .
```

For the PMO tasks used by `run.py`, install the extra chemistry/runtime dependencies in the same environment:

```bash
conda install -c conda-forge "rdkit<2024.03" -y
python -m pip install --upgrade PyTDC huggingface_hub
```

These two quick checks should work before you launch the benchmark:

```bash
python -c "import rdkit; print(rdkit.__version__)"
python -c "from tdc import Oracle; print('tdc ok')"
```

If you want Weights & Biases logging, set `WANDB_PROJECT` and `WANDB_ENTITY` in `src/hdbo_benchmark/utils/constants.py`. For a simple local run, use `--wandb-mode disabled`.

### Run one PMO benchmark

This reproduces a single COWBOYS run on the `albuterol_similarity` task using the pretrained 128-dimensional molecular VAE:

```bash
python run.py \
  --function-name albuterol_similarity \
  --solver-name cowboys \
  --n-dimensions 128 \
  --max-iter 300 \
  --seed 1 \
  --no-strict-on-hash \
  --wandb-mode disabled \
  --tag local-test
```

Important notes:

- this does not train a VAE; it loads the pretrained checkpoint shipped with the repository
- the first PMO run can take a bit longer because `poli` prepares the underlying task machinery
- the final output is written to `results/new_vae_10_chain_100_steps_with_stoch_sampling/<function_name>_<seed>.npy`

### Run the full PMO sweep

To run COWBOYS across all molecular PMO tasks and seeds:

```bash
./run.sh
```

On Windows, use Git Bash, WSL, or translate the loop in `run.sh` into PowerShell commands.

### What gets saved

The main entry point `run.py` saves one local `.npy` file per run containing the best objective value found by the solver.

If you run with WANDB enabled, the observer also logs:

- `x`: the evaluated sequence or latent point
- `y`: the objective value of that evaluation
- `best_y`: the best score seen so far

The repository also contains separate post-processing scripts under `src/hdbo_benchmark/results/` for creating tables and figures after many runs have finished.

### Google Colab (experimental)

Colab can work, but it is less reliable than a local Conda environment because the PMO stack depends on chemistry packages and TDC/poli integrations.

The PMO tasks in this repo need `tdc` available in the active Python
environment. On pip, that package is distributed as `PyTDC`. If `tdc` is
missing, `poli` falls back to a Conda-based isolation path, which is exactly
the failure mode you see on Colab.

Suggested Colab workflow:

```python
!git clone <your repo url>
%cd ROTLSC
!python -m pip install --upgrade pip
!python -m pip install -r requirements.txt
!python -m pip install gauche selfies PyTDC huggingface_hub
!python -m pip install "rdkit<2024.03"
!python -m pip install -e .
```

Then run:

```python
!python run.py --function-name albuterol_similarity --solver-name cowboys --n-dimensions 128 --max-iter 300 --seed 1 --no-strict-on-hash --wandb-mode disabled --tag colab-test
```

Before launching the benchmark, these checks should pass:

```python
!python -c "import rdkit; print(rdkit.__version__)"
!python -c "from tdc import Oracle; print('tdc ok')"
```

If Colab dependency resolution fails, prefer a Linux or Windows Conda environment instead. For reproducibility, the local Conda route is the recommended one.
