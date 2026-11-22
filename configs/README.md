# Configuration index

Configuration remains in its existing executable forms. There is no new YAML
experiment schema or configuration loader. This index connects the files without
duplicating settings or changing relative paths.

| Responsibility | Source of truth | Working directory / notes |
|---|---|---|
| Main environment | [`../environment.yml`](../environment.yml) | Repository root; editable install uses `.` |
| Preprocessing environment | [`../environment.data_preprocessing.yml`](../environment.data_preprocessing.yml) | Repository root; separate Python 3.8/TorchDrug stack |
| Optional solver environments | [`../envs/`](../envs/) | YAML pip entries use `..[extra]`; run from `envs/` |
| Package dependencies and extras | [`../pyproject.toml`](../pyproject.toml), [`../setup.cfg`](../setup.cfg) | Root; existing dependency versions preserved |
| Additional requirements | [`../requirements.txt`](../requirements.txt), [`../requirements-dev.txt`](../requirements-dev.txt) | Root; overlaps with metadata remain technical debt |
| Current experiment settings | [`../scripts/experiments/`](../scripts/experiments/) | `run.sh` uses the corrected COWBOYS setup; other historical launchers retain their settings |
| COWBOYS paper reference | [`cowboys_paper_table1.json`](cowboys_paper_table1.json) | Table 1 task list, reported values, protocol and verified checkpoint digest; see [implementation notes](../docs/COWBOYS_IMPLEMENTATION.md) |
| COWBOYS pending runs | [`cowboys_pending.tsv`](cowboys_pending.tsv) | Only albuterol seeds 1 and 2 remain incomplete after the latest November 21 merge. Pass as `RUNS_FILE` to `scripts/experiments/run.sh`; remove completed pairs after reviewing new results. |
| Scheduler settings | [`../scripts/slurm/`](../scripts/slurm/) | Submit from the repository root; original SLURM headers and arguments preserved |
| CLI defaults | [`../src/hdbo_benchmark/experiments/run.py`](../src/hdbo_benchmark/experiments/run.py) | Click options and `_main`; retain the existing `--sufix` spelling |
| Method defaults | [`../src/hdbo_benchmark/solvers/`](../src/hdbo_benchmark/solvers/) | Constructor defaults and `registry.py`; no extraction to config files |
| Benchmark setup | [`../src/hdbo_benchmark/benchmarks/load_problems.py`](../src/hdbo_benchmark/benchmarks/load_problems.py) | Existing initialization and objective conventions |
| Device and W&B | [`../src/hdbo_benchmark/utils/constants.py`](../src/hdbo_benchmark/utils/constants.py) | Existing `HDBO_TORCH_DEVICE` override and W&B constants |
| Checkpoint configuration | `data/trained_models/**` and package assets | Existing JSON/YAML metadata, including diffusion and protein AE settings; kept alongside checkpoints |
| Checks and CI | `tox.ini`, `.pre-commit-config.yaml`, `.github/workflows/` | Root; integrations may be expensive |

Do not move checkpoint metadata independently of checkpoints. A future unified
experiment configuration format belongs to Roadmap milestone 2.
