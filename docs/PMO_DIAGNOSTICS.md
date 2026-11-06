# PMO objective diagnostics

Audit date: 2026-11-06. Scope: the **25 tasks returned by the installed poli
`PMOBenchmark`**, covering every molecular task in this repository's PMO loader.
This does not claim to audit all protein, external-software or other problems
available throughout the poli ecosystem.

## Findings

**23 task integrations pass the checks below. JNK3 and GSK3beta are blocked in the
current research environment by incompatible cached classifier pickles.**
Valsartan SMARTS is a valid sparse objective; zero on most ordinary molecules
is expected. Integration checks do not prove every upstream scientific formula
or every possible input correct.

The active environment has Python 3.10.20, poli-core 1.3.1, poli-baselines
1.0.2.dev1, PyTDC 1.1.14, scikit-learn 1.7.2, NumPy 1.26.4, SciPy 1.15.2,
RDKit 2023.9.6 and SELFIES 2.2.0. Evidence records cached model SHA-256 hashes.
The original solver code, checkpoints and historical optimization results were
not changed. No BO iterations, neural-network training or full optimization
campaigns were run.

## The reported problems

### JNK3 and GSK3beta: model loading, then hidden errors

The cached `jnk3_current.pkl` and `gsk3b_current.pkl` identify their training
scikit-learn version as **0.23.0**. Loading the trees under 1.7.2 raises an
incompatible node-dtype `ValueError` involving `missing_go_to_left`.

JNK3 constructs its model eagerly, so initialization raises. GSK3beta loads lazily
inside evaluation. TDC's scalar call catches evaluator exceptions and returns
its default `0.0`, causing an apparently successful all-zero experiment. Calling
the raw evaluator or TDC's list API exposes the same model-loading error. This
behavior matches the upstream [TDC issue #238](https://github.com/mims-harvard/TDC/issues/238).
Scikit-learn documents the limits of loading models across versions in its
[model persistence guide](https://scikit-learn.org/stable/model_persistence.html).

To verify the cause without replacing or editing the models, the same files were
loaded in a separate **diagnostic-only** Python 3.8.20 process with scikit-learn
0.23.0, NumPy 1.19.5 and SciPy 1.5.4. Packages were installed into a temporary
`--target` directory, used with `python -S`; the existing Conda environments were
not downgraded. Identical RDKit fingerprints were passed to the original models.

| Cached model | poli reference molecule score in original runtime | Current runtime |
|---|---|---|
| JNK3 | 0.01 | Loading error |
| GSK3beta | 0.03 | Previously hidden as 0; now an explicit evaluation error |
| DRD2 | 0.0015465365340340924 | Same score, also matching the original runtime |

The reference molecule is
`CC(C)(C)[C@H]1CCc2c(sc(NC(=O)COc3ccc(Cl)cc3)c2C(N)=O)C1`.
Additional controls produced JNK3/GSK3beta scores of 0.04/0.11 for valsartan.

**JNK3 and GSK3beta are still unavailable for optimization in the current runtime.**
The diagnostic worker establishes that the existing models can produce valid
scores; it is not a production integration. A production repair needs a verified
oracle environment/bridge or upstream-compatible model artifact with provenance
and reference-score validation. Do not simply rewrite tree fields, replace
pickles, change scores, or downgrade the entire research environment blindly.

### Valsartan SMARTS: legitimate zeros and narrow score terms

The oracle requires the SMARTS motif `CN(C=O)Cc1ccc(c2ccccc2)cc1`. Its score is the
geometric mean of that binary substructure term and three Gaussian descriptor
terms: logP (sigma 0.2), TPSA (sigma 5), and Bertz complexity (sigma 30), each
centered on sitagliptin's descriptor value. A missing motif forces the score to
zero; even a motif match may score very poorly on the descriptor terms. See
[TDC oracle implementation](https://tdc.readthedocs.io/en/latest/_modules/tdc/chem_utils/oracle/oracle.html#valsartan_smarts).

| Control | Score |
|---|---|
| Ethanol, aspirin and poli's generic example | 0 |
| Minimal matching motif | 1.5554200758111907e-19 |
| Valsartan | 9.540221553711118e-11 |
| Matching polar control below | 0.0003957924316773945 |

The stronger control is `CN(C=O)Cc1ccc(-c2ccc(C(N)=O)cc2)cc1O`. Its score agrees
with an independent calculation of the descriptor Gaussian/geometric-mean
formula. The control uses only the existing VAE vocabulary. The full valsartan
control also uses `[NH1]`, which is absent from that vocabulary: the raw objective
can score it, but that particular SELFIES encoding cannot be represented by the
trained VAE. This does not make the SMARTS objective unreachable; the stronger
control above remains representable.

All ten initial molecules at seed 1 have zero Valsartan SMARTS scores. Three
fixed latent points are checked separately against direct scoring. This is
consistent with sparse rewards, not evidence that the optimizer has found a
substructure match. No objective smoothing, modified rewards or initialization
changes were introduced.

## Complete task inventory

| Task | Seed-1 initial score range (10 molecules) | Result |
|---|---|---|
| `albuterol_similarity` | 0.150538 to 0.329412 | Pass |
| `amlodipine_mpo` | 0.00607158 to 0.459763 | Pass |
| `celecoxib_rediscovery` | 0.0740741 to 0.198276 | Pass |
| `deco_hop` | 0.502646 to 0.550999 | Pass |
| `fexofenadine_mpo` | 1.47151e-05 to 0.502269 | Pass |
| `isomer_c7h8n2o2` | 9.34906e-62 to 0.119433 | Pass |
| `isomer_c9h10n2o2pf2cl` | 1.1325e-30 to 0.0432558 | Pass |
| `median_1` | 0 to 0.161569 | Pass |
| `median_2` | 0.0564815 to 0.132746 | Pass |
| `mestranol_similarity` | 0.0586081 to 0.276423 | Pass |
| `osimetrinib_mpo` | 2.88299e-05 to 0.71412 | Pass |
| `perindopril_mpo` | 0.0379439 to 0.419524 | Pass |
| `ranolazine_mpo` | 0.00488631 to 0.0635639 | Pass |
| `rdkit_logp` | 0.7928 to 3.3852 | Pass |
| `rdkit_qed` | 0.491934 to 0.917764 | Pass |
| `sa_tdc` | 2.16653 to 4.4139 | Pass; raw SA direction needs care |
| `scaffold_hop` | 0.337331 to 0.410399 | Pass |
| `sitagliptin_mpo` | 9.90812e-15 to 0.0877733 | Pass |
| `thiothixene_rediscovery` | 0.0686275 to 0.233871 | Pass |
| `troglitazone_rediscovery` | 0.0679612 to 0.165468 | Pass |
| `valsartan_smarts` | 0 to 0 | Pass (sparse objective) |
| `zaleplon_mpo` | 4.20002e-12 to 0.345656 | Pass |
| `jnk3` | Unavailable | Blocked: incompatible cached classifier |
| `gsk3_beta` | Unavailable | Blocked: incompatible cached classifier |
| `drd2_docking` | 8.42848e-05 to 0.246232 | Pass |

The original [PMO benchmark repository](https://github.com/wenhao-gao/mol_opt)
describes 23 tasks. This installed poli benchmark includes the extra
`rdkit_logp` and `sa_tdc` properties. Do not present their inclusion as exactly
reproducing the original 23-task PMO suite.

`drd2_docking` is poli's name here, but its implementation uses the **DRD2
classifier**, not a docking engine. Raw logP is also not the penalized-logP
oracle. Preserve these distinctions in experiment labels and publications.

**SA direction requires a research decision:** `sa_tdc` returns raw SA scores,
where lower values mean easier synthesis ([TDC SA documentation](https://tdcommons.ai/functions/oracles/#synthetic-accessibility-sa)).
The current solver/runner convention uses maxima (`get_best_performance()` and
saved `best_value`). Maximizing raw SA rewards synthetic difficulty. The property
calculation passes integration checks, but it must not be interpreted as an
accessibility-maximization experiment without an explicit, documented direction
or transformation. This audit does not silently negate historical objectives.

## Repository fixes

1. `utils/tdc_compat.py` retains proper package `__spec__` metadata in its fallback
   modules. Previously, a second PMO initialization in the same process could
   fail with `ValueError: tdc.__spec__ is None`. The fallback is now discoverable
   and repeatable.
2. The PMO loader wraps an in-process TDC oracle so scalar inputs use TDC's
   one-item list path. This preserves valid scores, invalid-molecule defaults
   and the number of oracle calls while allowing evaluator exceptions to
   propagate. GSK3beta now fails during initialization instead of creating bogus
   zero-valued supervised data. The guard does not add probe evaluations.

The error guard applies to the repository loader; direct external calls to
`poli.create` or TDC without the helper retain their upstream behavior. Existing
historical runners have not been rewritten, and inaccessible externally isolated
oracles cannot be wrapped in-process. Do not assume all external entry points
inherit the guard.

## Verification and reproducibility

For each task the audit attempted seed-1 initialization (ten molecules), five
valid molecular probes, SMILES/SELFIES-string/padded-token representations, raw
TDC calls, list calls, batching, repeatability and invalid inputs. Passing tasks
have initial shapes `(10, 70)` and `(10, 1)`, ten valid molecules and ten accounted
initial evaluations. The diagnostic's requested 100 additional evaluations give
an unchanged total budget of 110. That larger diagnostic budget accommodates
probes; it is not a change to the production experiment defaults.

For all 23 available tasks, valid outputs are identical before and after the
error guard. Every cached oracle model hash is unchanged. Invalid RDKit returns
may be NaN; evidence serializes nonfinite values as JSON null with `finite:false`.
Invalid-input conventions differ between upstream tasks and are recorded rather
than forced to one universal score.

Five similarity/rediscovery reference molecules score one in both SMILES and
SELFIES. Valsartan's negative and positive controls agree with the stated
formula. Regression tests also cover repeated compatibility initialization,
error propagation, call counts, default preservation and non-TDC black boxes.

```bash
python -m unittest tests.test_tdc_compat tests.test_pmo_controls tests.test_structure -v
python scripts/diagnostics/check_pmo.py --output <fresh-audit-path.json>
HDBO_TORCH_DEVICE=cpu python scripts/diagnostics/check_pmo_latent.py --output <fresh-latent-path.json>
```

The latent check loads the existing 128-dimensional molecular VAE, decodes three
fixed points, and compares direct, normalized-wrapper and diagnostic-wrapper
scores. It does not fit a GP or evaluate an optimizer trajectory.

Original-model inference is reproducible with
`scripts/diagnostics/check_pmo_models.py`: write fingerprints using the research
environment, install `scikit-learn==0.23.0 numpy==1.19.5 scipy==1.5.4 joblib==0.15.1 threadpoolctl==2.1.0`
into a fresh temporary target using Python 3.8, then invoke the script using that
Python with `-S --packages <target> --fingerprints <probes.json> --model-dir oracle --output <fresh-model-check.json>`.
No pretrained artifact is downloaded by these checks; the model files already
present in `oracle/` are used. Full audit HTTP access is blocked explicitly.

Evidence:

- [Summary and coverage](verification/pmo_summary.json)
- [Original audit, reproducing fallback re-entry failure](verification/pmo_before.json)
- [After package-metadata fix, before error guard](verification/pmo_audit.json)
- [Final audit with error guard](verification/pmo_verified.json)
- [Original scikit-learn classifier checks](verification/pmo_original_sklearn.json)
- [Latent wrapper checks](verification/pmo_latent.json)

The first audit initializes the fallback before exercising the repository loader,
which deliberately exposes the re-entry bug; a fresh process's first original
loader call could succeed. Final results describe this environment and these
controls, not exhaustive mathematical or biological validation. The unrelated
Windows BLAS failure during GP fitting remains outside this objective audit.
