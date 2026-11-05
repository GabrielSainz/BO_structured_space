# Research repository guidance

This repository studies Bayesian optimization over discrete/structured
sequences with poli: molecular and protein tasks, latent models, GP surrogates,
acquisition functions and generative proposals. Preserve scientific behavior.

## Map

- `src/hdbo_benchmark/solvers/`: COWBOYS, two flow variants, diffusion COWBOYS,
  LSBO and `registry.py` for local/upstream methods. GP/acquisition and flow
  components remain inside methods; shared GP fitting is in
  `utils/experiments/training_gps.py`.
- `benchmarks/` inside the package: poli loaders and representation adapters.
- `generative_models/`: VAEs, protein AE, one-hot and diffusion components.
- `experiments/run.py`: current CLI; other `experiments/` folders retain older
  protocols and training commands. Root `run.py` is a compatibility entry point.
- `scripts/experiments/`: existing sweeps. `configs/README.md` indexes actual
  settings in scripts, Python defaults, root environments and `envs/`.
- `scripts/slurm/`: scheduler entry points; submit from the repository root.
- `evaluation/`: current local analysis. Package `results/`, `utils/results/`
  and `utils/visualization/` retain older analysis and plotting.
- `tests/`: structural checks and existing environment-specific integrations.
- `docs/`: audit, architecture, verification, roadmap and technical debt.
- Root `results/`, `reports/`, `logs/`, `wandb/`: generated artifacts;
  `data/` and package `assets/`: datasets, metadata and checkpoints.

## Workflow and principles

1. Read `docs/architecture.md` and relevant debt/roadmap items.
2. Locate canonical modules, callers, configuration and compatibility aliases.
3. Read the current implementation before proposing a focused change.
4. State the intended change and implement within the user's authorized scope.
5. Run appropriate lightweight checks; update relevant documentation.

Keep changes focused and modular; avoid new duplicated utilities. Never silently
change equations, kernels, objectives, proposal distributions, defaults or
experimental protocols. Document any authorized behavior change. Preserve seeds,
budgets, objective conventions, configuration and result provenance. Retain
reasonable import/CLI compatibility; edit canonical modules, not forwarding
shims. Root sweep, report, method and test duplicates were removed in the
post-migration cleanup; use canonical paths. Installed aliases in `src/` and
the old loader aliases still support existing imports and serialized classes.
Never delete historical experiments, checkpoints or results.

## Computational constraints

Do not automatically train networks, download pretrained models, use GPUs,
launch sweeps or run full benchmarks. Prefer CPU COWBOYS for an appropriate
bounded end-to-end check; its recorded Phase 1 run fails in the existing Windows
BLAS environment. Do not interpret that as a new regression or fix it without
scope. See `docs/MIGRATION_VERIFICATION.md` for the exact attempt.

Use the established Python 3.10 environment. Dependency-free checks:
`python -m unittest tests.test_structure`. Existing pytest integration tests
need optional dependencies and can execute real objectives; select deliberately.

Phase 1 is architecture-only: zero-score diagnostics and scientific fixes belong
to future work. Stay on the authorized refactoring branch; do not change `main`,
commit, push or merge without explicit user instruction.
