# Open technical work

Only unresolved work is listed here. Current behavior and completed scientific
changes are documented in [architecture](architecture.md),
[COWBOYS](COWBOYS_IMPLEMENTATION.md), [PMO diagnostics](PMO_DIAGNOSTICS.md) and
[molecular metrics](MOLECULAR_METRICS.md). These items are not authorization to
change experimental protocols or launch computation.

## Benchmark interpretation and runtime

- **Classifier compatibility:** JNK3 and GSK3beta need a compatible production
  oracle runtime. Old cached classifiers failed with scikit-learn 1.7.2;
  evaluator failures are now exposed. Preserve affected-run provenance.
- **Objective conventions:** raw `sa_tdc` is lower-is-easier while the runner
  maximizes recorded objectives. Decide transformations explicitly before
  scientific comparisons. `drd2_docking` is a classifier objective; the installed
  25-task set includes two properties beyond the original 23-task PMO suite.
- **Sampler failures:** the COWBOYS cap is 50,000 pCN steps. Investigate acceptance,
  decoder validity, novelty and beta when sampling stalls; a higher cap does not
  establish successful completion. Keep 10,000- and 50,000-cap attempts distinct.
- **Environment reproduction:** use Conda activation on Windows. Directly calling
  the environment Python from the base shell can trigger the historical BLAS
  failure. A clean environment and supported isolated wheel build still need
  verification; the recorded local build used setuptools 80.10.2 against the
  declared `setuptools<=68` backend.

## Packaging and compatibility

- Consolidate overlapping packaging/environment definitions only after recording
  compatible chemistry, poli and model-stack versions. Moving Git dependencies
  and partially constrained scientific packages limit reproducibility.
- Repository-relative resources, duplicated checkpoint locations and old
  environment working-directory assumptions complicate relocation. Preserve
  checkpoint bytes and lookup precedence when improving resource resolution.
- Import aliases are checked, but full historical serialized solver objects and
  consumers of module `__file__` have not been certified.
- CLI default `foldx_stability` differs from loader key `rfp_foldx_stability`;
  factory-only vanilla BO variants are missing from `SOLVER_NAMES`. Confirm
  intended compatibility before changing names/defaults.
- `workflows/website.yml` is outside active Actions configuration and some CI
  definitions target `master`; review separately before enabling deployment.

## Experiment management and validation

- Current and historical runners use different protocols and poli APIs. Specify
  initialization, objective direction, budgets and result contracts before
  unifying them. Record planned budgets, oracle versions, environment/revision,
  timings and physical oracle calls; some are unavailable in existing artifacts.
- Exact optimizer/RNG resumption is unverified. Observation NPZ files alone are
  not resumable solver checkpoints. Preserve attempts and reject output collisions.
- Integration tests can construct real objectives at collection. Separate their
  selection from unit tests; confirm optional environments and LaMBO2 coverage.
- Extend equation/numerical verification to the flow, diffusion and LSBO variants.
  GP/acquisition and flow implementations contain duplicated code; extract only
  demonstrated-equivalent pieces after checking trajectories and RNG behavior.
- Legacy analysis uses hardcoded selections and import-time execution. Preserve
  historical interpretation while improving reusable interfaces and multi-seed
  comparisons. Never treat partial runs or duplicate seeds as independent,
  full-budget replicates.
