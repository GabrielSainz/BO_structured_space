# File migration map

> Historical Phase 1 record. The subsequent [cleanup](CLEANUP.md) removed redundant compatibility launchers; see that record for current entry points.

Every original entry point is retained as a compatibility shim. Canonical implementations are edited at their new locations.

| Original path | Canonical path |
|---|---|
| `cowboys.py` | `src/hdbo_benchmark/solvers/cowboys.py` |
| `cowboys_flow.py` | `src/hdbo_benchmark/solvers/cowboys_flow.py` |
| `cowboys_flow_2.py` | `src/hdbo_benchmark/solvers/cowboys_flow_2.py` |
| `run.py` | `src/hdbo_benchmark/experiments/run.py` |
| `src/hdbo_benchmark/utils/experiments/load_solvers.py` | `src/hdbo_benchmark/solvers/registry.py` |
| `src/hdbo_benchmark/utils/experiments/load_problems.py` | `src/hdbo_benchmark/benchmarks/load_problems.py` |
| `src/hdbo_benchmark/utils/experiments/problem_transformations.py` | `src/hdbo_benchmark/benchmarks/problem_transformations.py` |
| `reports/plot_iteration_metrics.py` | `src/hdbo_benchmark/evaluation/plot_iteration_metrics.py` |
| `reports/table_results.py` | `src/hdbo_benchmark/evaluation/table_results.py` |
| `test_main_entry_point.py` | `tests/test_main_entry_point.py` |
| `run.sh` | `scripts/experiments/run.sh` |
| `run_diffusion.sh` | `scripts/experiments/run_diffusion.sh` |
| `run_diffusion2.sh` | `scripts/experiments/run_diffusion2.sh` |
| `run_diffusion_guidance_sweep.sh` | `scripts/experiments/run_diffusion_guidance_sweep.sh` |
| `run_diffusion_iterations.sh` | `scripts/experiments/run_diffusion_iterations.sh` |
| `run_flow.sh` | `scripts/experiments/run_flow.sh` |
| `run_flow_2.sh` | `scripts/experiments/run_flow_2.sh` |
| `run_iterations.sh` | `scripts/experiments/run_iterations.sh` |
| `run_iterations_nf.sh` | `scripts/experiments/run_iterations_nf.sh` |
| `run_lsbo_poli.sh` | `scripts/experiments/run_lsbo_poli.sh` |
