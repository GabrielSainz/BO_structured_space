# Existing technical debt

These observations are not fixes. Scientific causes were not investigated in
Phase 1. Severity describes impact; priority describes suggested order, not a
deadline. Roadmap items below provide scoped follow-up work.

| ID | Observation and evidence | Severity | Priority | Follow-up / acceptance |
|---|---|---|---|---|
| TD-01 | User reports objectives staying zero across iterations on some poli tasks. No root cause established and no scan of historical scores performed. | High: research conclusions may be affected | P0 | M1.1–M1.3: reproduce independently, identify representation/normalization/sparse-reward causes, add tests only for confirmed defects |
| TD-02 | CPU COWBOYS fails before and after migration in `threadpoolctl` while BoTorch/SciPy fits the GP: Windows error `0xc06d007f`. Initial data and partial saving succeed. | High: blocks local end-to-end validation | P0 | M1.1: establish a supported environment without changing solver logic; bounded CPU run must complete |
| TD-03 | `pyproject.toml` and `setup.cfg` split metadata; requirements files and Conda definitions overlap. Build requirement is `setuptools<=68`, but local research environment has 80.10.2. No-isolation wheel build hits metadata failure with both original and migrated metadata. | High for installation; not a proven supported-build failure | P1 | M1.1: verify declared build backend in isolation; then consolidate metadata under a separately scoped packaging change |
| TD-04 | Scientific dependencies are incompletely constrained; Git dependencies track moving upstream revisions. gauche/SELFIES/PyTDC/chemistry installation relies partly on README instructions. | High: reproducibility | P1 | M1.1 / M2.2: record compatible versions and benchmark-specific requirements, test clean environment |
| TD-05 | Current CLI, benchmark-specific runners, shell sweeps and compiler scripts maintain distinct protocols. Older modules reference `poli.objective_repository`. | Medium–high | P1 | M2.1: inventory behavioral differences before proposing a common runner; preserve historical adapters |
| TD-06 | GP/acquisition/fingerprint/diagnostic code is duplicated across COWBOYS variants; flow v1/v2 have overlapping classes but distinct implementations. | Medium | P2 | M3.1–M3.2: verify equations first, then extract only behaviorally equivalent utilities with regression evidence |
| TD-07 | No small pre-existing unit suite; the single pytest file constructs real benchmark combinations at collection and runs integrations. pytest is absent from the available research environment. LaMBO2 filtering asks for `foldx_stability`, absent from the generated setups. | Medium–high | P1 | M3.2: separate unit/integration collection, add meaningful algorithm tests; confirm LaMBO2 coverage |
| TD-08 | CLI default `foldx_stability` differs from the loader's `rfp_foldx_stability`; factory-only vanilla BO branches are missing from `SOLVER_NAMES`. | Medium: interface inconsistency | P1 | M2.1: inventory accepted names, test intended aliases/defaults before changing behavior |
| TD-09 | Root-relative and `__file__`-depth paths, package/data checkpoint duplication, environment YAML relative editable paths, and long Windows result names complicate relocation/standalone installs. | Medium | P1 | M2.2: explicit resource resolution with precedence tests; no checkpoint deduplication without provenance review |
| TD-10 | Result formats are split between current JSON/NPZ/NPY, legacy data directories and W&B. Output names can be reused across runs. | High for provenance | P1 | M2.2–M2.3: preserve/import old artifacts; define collision policy and resumability without overwriting history |
| TD-11 | Diagnostic helpers may perform objective evaluations during final metric construction; exact budget/observer consequences require scientific verification. This is a review question, not a confirmed bug. | Potentially high | P1 | M1.3 / M3.2: count objective calls and verify state/RNG restoration independently |
| TD-12 | Table/legacy analysis modules contain top-level execution and hardcoded selections. GP, acquisition and model code share large solver modules. | Medium | P2 | M3.2 / M4: separate reusable components only after tests; preserve existing tables and interpretation |
| TD-13 | Historical VAE source/checkpoints, standalone `gitignore`, root bytecode and stale documentation remain. `workflows/website.yml` is outside active Actions configuration; CI targets `master`. | Low–medium | P2 | Separate maintenance review; retain history, do not activate deployment or remove files implicitly |
| TD-14 | Old serialized payloads may refer to legacy module paths; class alias lookups are checked but full saved objects are not. External programs may inspect moved modules' `__file__`. | Medium | P1 | M2.2: test representative trusted historical objects in the exact original dependency environment |

Structural corrections already made: canonical package locations for root
methods/runner/local analysis, compatibility aliases, lazy solver exports,
integration-test location, and an anchored runtime-results ignore rule. None of
these establishes scientific validity or resolves the debt above.
