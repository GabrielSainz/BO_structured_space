# Future development roadmap

All work here is **proposed, not implemented**. Each row is an independently
scopable task with its own acceptance criteria. No deadlines are assigned.
Begin each milestone by reading the current implementation and the Phase 1
verification record; do not silently include scientific fixes in maintenance.
P0 = blocking diagnosis, P1 = research foundation, P2 = subsequent improvement.

## Milestone 1 — Benchmark diagnostics

Goal: explain suspicious objective values and establish a trustworthy small
baseline before comparing optimization methods. Preserve failing evidence and
do not assume zero scores necessarily indicate a bug.

| Item / objective | Motivation | Proposed components | Dependencies | Validation strategy | Priority | Acceptance criteria |
|---|---|---|---|---|---|---|
| M1.1 Establish a reproducible CPU environment and baseline | Current Windows BLAS failure prevents complete COWBOYS validation; build/dependency versions differ from declarations | Environment definitions, packaging metadata, `tests/`, verification docs | Phase 1 evidence, no earlier milestone | Reproduce recorded failure; test declared build backend separately; run one bounded existing PMO/COWBOYS configuration with no training/downloads | P0 | Document compatible versions and commands; installation and bounded CPU run complete; scientific settings remain unchanged |
| M1.2 Diagnose all-zero tasks independently | Distinguish sparse rewards from representation or initialization errors | `benchmarks/load_problems.py`, `benchmarks/problem_transformations.py`, generative loaders; new narrowly scoped tests | M1.1 and identification of affected task/seed/configuration | Evaluate known valid and intentionally invalid candidates directly; inspect initial candidates, SELFIES/protein token shapes, padding, decoding and objective return shapes | P0 | Each investigated task has a minimal reproduction and evidence classifying expected sparsity, benchmark/runtime issue or confirmed implementation bug |
| M1.3 Verify transformations and evaluation accounting | Normalization, penalties, wrappers and diagnostic calls can affect interpretation and budgets | Benchmark transformations, normalization helpers, `utils/logging/candidate_diagnostics.py`, iteration metrics | M1.2 | Compare independent objective values before/after wrappers; count calls including diagnostics; verify budget, observer and RNG-state effects | P1 | Objective conventions and counts documented; confirmed bugs have regression tests and separately reviewed fixes; unexplained cases remain explicitly open |

Milestone completion: affected benchmark behavior is explained with reproducible
evidence, and at least one bounded COWBOYS baseline can serve as a meaningful
future regression check. Fixes require their own scoped scientific review.

## Milestone 2 — Unified experiment management

Goal: design a reproducible framework from actual protocols, then implement it
in a separately authorized change. Phase 1 did not introduce this framework.

| Item / objective | Motivation | Proposed components | Dependencies | Validation strategy | Priority | Acceptance criteria |
|---|---|---|---|---|---|---|
| M2.1 Specify consistent method/runner/configuration contracts | Current and historical runners differ; registration/default names are inconsistent | `experiments/`, `solvers/registry.py`, `configs/`, a documented interface proposal | M1 completed for representative tasks | Map every runner's initialization, bounds, seeds, budgets, objective direction and settings; prototype adapters against preserved entry points | P1 | Written contract and migration matrix approved; each old protocol has an explicit preservation or deprecation path; no accidental default changes |
| M2.2 Implement configuration, seed and provenance handling | Results need auditable model/environment/benchmark identity | Proposed configuration schema, experiment metadata storage, resource resolver and trusted-serialization checks | M2.1, M1.3 | Round-trip configuration and metadata; replay a bounded run; test resource precedence and existing artifact imports; exercise name collisions | P1 | Seeds, initialization and total evaluation budgets recorded; checkpoint identities and dependency versions recoverable; existing data imports without overwriting history |
| M2.3 Add resumability and comparative orchestration | Longer evaluations need safe recovery and comparable budgets | Proposed runner/checkpoint adapters and comparative benchmark commands under `experiments/` | M2.2, verified solver state from M3 where needed | Interrupt/resume a small run; compare state, RNG and objective call counts with uninterrupted reference; dry-run sweeps | P2 | Supported methods resume with declared guarantees; unsupported methods are explicit; comparable runs preserve budgets/configurations and avoid duplicate evaluations |

Milestone completion: one documented runner/configuration path supports selected
verified methods, while historical commands remain interpretable and accessible.
Do not assume every third-party solver can implement identical resume semantics.

## Milestone 3 — Algorithm verification

Goal: establish what existing implementations compute before refactoring shared
scientific internals. Paper review and algorithm changes belong to this milestone.

| Item / objective | Motivation | Proposed components | Dependencies | Validation strategy | Priority | Acceptance criteria |
|---|---|---|---|---|---|---|
| M3.1 Review equations and implementation details | Method names do not establish mathematical correctness | Five local solver files, generative models, acquisition/kernel settings; method notes in `docs/` | M1; M2.1 useful but not mandatory | Map code to original papers/reference algorithms; inspect proposal densities, acceptance ratios, guidance, GP fitting, bounds and objective direction | P1 | Each local method has a documented equation-to-code review; confirmed deviations distinguished from intentional variants and open questions |
| M3.2 Add numerical and behavioral regression tests | Existing integration coverage is too broad and missing small invariants | `tests/`, local solvers, GP/normalization/diagnostic utilities | M3.1 and M1.3 | Synthetic objectives, finite/shape/bounds checks, deterministic replay where supported, proposal/acquisition reference comparisons, pathological inputs | P1 | Tests catch confirmed algorithmic bugs and numerical failures; authorized fixes include before/after evidence; budgets and initialization are explicitly checked |
| M3.3 Consolidate proven shared scientific utilities | Duplicated code makes fixes diverge across variants | Potential GP, acquisition, fingerprint and diagnostics modules; flow-specific code stays near methods unless equivalent | M3.2; M2.1 for interfaces | Compare trajectories or justified statistical invariants before/after extraction; verify old class/import compatibility | P2 | Only demonstrated-equivalent code is shared; intentional differences are documented; no unexplained numerical/protocol changes |

Milestone completion: documented scientific intent and tests exist for each
reviewed method, with a clear list of remaining uncertainties. Do not report
untested methods as verified.

## Milestone 4 — Benchmarking and analysis

Goal: build on the already existing iteration plotting and table scripts to
produce reproducible, fair comparisons; do not treat plotting as absent today.

| Item / objective | Motivation | Proposed components | Dependencies | Validation strategy | Priority | Acceptance criteria |
|---|---|---|---|---|---|---|
| M4.1 Standardize artifact reading and optimization curves | Old/local/W&B formats and objective conventions differ | `evaluation/`, legacy result loaders; proposed format adapters | M1 conventions and M2.2 provenance | Small hand-computed fixtures covering best-so-far values, initial evaluations, partial/failed runs and different objective directions | P1 | Existing formats load without altering originals; curves use documented evaluation axes and explicit inclusion rules |
| M4.2 Validate multi-seed comparisons and uncertainty | Missing seeds and unequal budgets can distort averages | `evaluation/plot_iteration_metrics.py`, table utilities, report metadata | M4.1 and M3-verified methods | Known synthetic seed arrays; missing/censored-run scenarios; uncertainty and ranking calculations checked against independent calculations | P1 | Best-so-far curves, seed counts, aggregation rules and statistical uncertainty are reproducible; performance claims include limitations |
| M4.3 Compare evaluation efficiency, runtime and cost | Oracle calls and generative/GP computation have different costs | Proposed timing/provenance fields and analysis reports | M2.2, M4.2 | Separate objective-call count from iteration count, wall time and model overhead; repeated small controlled comparisons | P2 | Reports expose hardware/environment, oracle budgets, runtime and computational cost without hiding failed runs or amortized training assumptions |

Milestone completion: a documented small comparison is regenerable from retained
artifacts; larger benchmark campaigns require an explicit compute plan.

## Milestone 5 — Future research methods

Goal: develop new methods only after scientific baselines and evaluation are
trustworthy. The following are research proposals, not existing implementations.

| Item / objective | Motivation | Proposed components | Dependencies | Validation strategy | Priority | Acceptance criteria |
|---|---|---|---|---|---|---|
| M5.1 Explore GenTuRBO-inspired local/global generative optimization | Combine incumbent-centered search and global exploration with trust-region adaptation | New method under `solvers/`, proposal/trust-region components, existing generative models | M1, M3 and M4; M2 runner when available | Synthetic/small discrete tasks; ablate trust-region adaptation, incumbent-local proposals and global exploration under equal budgets | P2 | Algorithm and adaptation rules specified; reproducible ablations against verified TuRBO/COWBOYS/LSBO baselines; no unqualified superiority claims |
| M5.2 Explore acquisition-guided generative sampling | Study how guidance changes proposal quality, diversity and cost | New or explicitly versioned flow/diffusion method, guidance components, diagnostics | M3 review of current diffusion/flow methods; M4 comparisons | Matched oracle budgets; guidance-strength and proposal-mixture ablations; diversity, feasibility and cost measurements | P2 | New equations and implementation choices documented; controlled evidence isolates guidance effects from extra compute/evaluations |
| M5.3 Extend protein optimization and surrogate/kernel choices | Structured sequences require representation and uncertainty models suited to the task | Protein benchmark adapters, AE/ESM integration, candidate new GP/kernel modules | M1 protein objective checks, M3 and M4; explicit model/compute authorization | Small protein tasks, representation round trips, kernel validity/numerical checks and fixed-budget comparisons; quantify training cost separately | P2 | Supported protein representations and objective conventions tested; surrogate/kernel choices have documented motivation, reproducible experiments and compute accounting |

Milestone completion is a research result or a documented negative finding for
the selected direction, not a requirement to implement every proposal. New model
training, pretrained downloads, GPU use and large sweeps require explicit scope.
