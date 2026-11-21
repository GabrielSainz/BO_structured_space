# COWBOYS fidelity audit

> Historical pre-fix findings. The subsequently authorized corrections and
> execution evidence are documented in [COWBOYS_IMPLEMENTATION.md](COWBOYS_IMPLEMENTATION.md).

Reviewed 2026-11-17 on `refactor/agent-repo`, local HEAD
`bf9475c4d554e0a708cb4b8f649832fbfce135d8`. This is an audit; no solver,
decoder, checkpoint, default, objective, or experiment protocol was changed.

## Conclusion and reference versions

The canonical local `solvers/cowboys.py` is very close to the authors' released
implementation. Its GP fitting, fingerprint construction and VAE setter have
identical Python ASTs. After removing local diagnostics and inlining their
temporary novelty variable, `next_candidate` also has an identical AST. Every
method of the molecular `VAESelfies` class matches upstream. However, matching
the released implementation does **not** establish fidelity to the paper's
Gaussian-prior sampling target or its reported experimental settings.

References inspected:

- The supplied 15-page PDF, *Return of the Latent Space COWBOYS*,
  arXiv:2507.03910v1. Equations and Algorithms 2-4 were inspected visually.
  [Paper](https://arxiv.org/html/2507.03910v1), especially Sections 2.1, 5.1,
  7.2 and Appendix A.
- The authors' repository, pinned to
  [6bac4e48f3f345de6727d8d3b8b8162b443a212e](https://github.com/henrymoss/ROTLSC/tree/6bac4e48f3f345de6727d8d3b8b8162b443a212e).
  Relevant files: root `cowboys.py`, `run.py`, `run.sh`, molecular
  `vae_selfies.py`, `vae_factory.py` and `problem_transformations.py`.
- Current local runner, solver registry, molecular problem transformation,
  VAE/model loaders, iteration metrics and candidate diagnostics. The installed
  `poli_baselines` base solver was also read to resolve inherited bounds.

Flow and diffusion variants were not audited. This review concerns the
molecular COWBOYS baseline and the Section 7.2 benchmark, rather than reproducing
the paper's other VAE architectures and batch experiments.

## Findings, ordered by scientific impact

### F1. High: pCN operates in normalized coordinates with the wrong prior

Local references: `benchmarks/problem_transformations.py:202-223`,
`utils/experiments/load_generative_models.py:25`,
`experiments/run.py:367-368`, `solvers/cowboys.py:124,155,169-170`.

The wrapper stores coordinates `u = (z + 10) / 20`, where `z` is a raw VAE
latent. The current runner sets `solver.bounds = (-10, 10)`, so both the
surrogate's decoder and the objective wrapper decode `z = 20u - 10`.
But the sampler applies

```text
u' = sqrt(1 - beta^2) u + beta epsilon,  epsilon ~ N(0, I).
```

This proposal preserves a standard normal in **u**, which corresponds to
`N(-10 * 1, 400 I)` in raw VAE coordinates. The intended VAE prior in Equation
(5) is `N(0, I)` in **z**. The raw-coordinate form of the implemented proposal is

```text
z' = sqrt(1 - beta^2) z + 10 (sqrt(1 - beta^2) - 1) + 20 beta epsilon.
```

At `z=0, beta=0.1`, its mean is about `-0.05013` per coordinate and its standard
deviation is `2`, versus mean `0` and standard deviation `0.1` for raw-latent
pCN. These are properties of the proposal's invariant Gaussian, not a claim
that the finite, adaptive, PI-weighted samples actually follow that Gaussian.

The coordinate design is inherited from upstream, whose sampler also consumes
normalized history. The local runner's explicit bounds assignment is an
additional behavioral difference: upstream leaves `self.bounds` inherited,
while its setter only records `vae_bounds`. In the installed base class that
inherited default is `(-2, 2)`; the upstream run environment's exact historical
default was not established. Removing the local assignment would introduce a
decoder mismatch with the objective wrapper in this environment, not solve the
prior problem.

Recommended correction: make raw versus normalized coordinates explicit, run
pCN in raw VAE space and normalize only at the solver/objective boundary.
Equivalently, a normalized-coordinate implementation must preserve
`N(0.5 * 1, 0.0025 I)`. Use the explicit VAE decoding bounds rather than relying
on an inherited acquisition-optimization bound. This changes scientific
behavior and must be recorded as a corrected baseline, separate from an exact
upstream reproduction.

### F2. High: initial molecule scores can be paired with reconstructed molecules

Local references: `benchmarks/load_problems.py:54-70`,
`benchmarks/problem_transformations.py:217-223`, `solvers/cowboys.py:131-139`.

The loader evaluates the original initial molecules. The continuous wrapper
then replaces their structures with encoder means while retaining their
original scores. At GP fitting time, the solver decodes those encoder means
and fingerprints the reconstructions. When reconstruction changes a molecule,
the training pair becomes `(fingerprint(reconstruction), score(original))`.
This conflicts with the paper's structure-space GP dataset of actual evaluated
structures and can also distort which structures count as previously observed.
It is inherited from upstream.

A bounded probe with a deliberately lossy toy decoder confirmed the data-flow
issue without calling any real objective. A separate CPU inference check used
the existing default 128-dimensional checkpoint and ten cached seed-1 initial
molecules: only **7/10** reconstructions were the same canonical molecule.
The other three also had different folded GP fingerprints. This confirms
that reconstruction changes matter to the actual surrogate representation.
Per-molecule evidence is in
[`cowboys_reconstruction.json`](verification/cowboys_reconstruction.json).
No scores were recomputed; reconstruction differences alone do not quantify
the resulting objective error or its impact on optimization performance.

Recommended correction: retain original evaluated structures alongside their
latent coordinates and scores; fit the GP on those original structures.
Use encoded latents as warm starts with explicit provenance. Do not silently
replace labels by reevaluating reconstructions, since that changes initial
design and objective cost.

### F3. High for protocol comparison: final diagnostics make extra oracle calls

Local references: `experiments/run.py:180-184`,
`utils/logging/iteration_metrics.py:143-155`,
`utils/logging/candidate_diagnostics.py:63-76,102-111,150-156`.

Within `next_candidate`, local top-k diagnostics use `evaluate_objectives=False`
and preserve RNG state while scoring candidates. However, final result saving
enables objective evaluation whenever status is not `running`. Iteration-result
saving is enabled by default. Finalization can therefore evaluate up to ten
diagnostic candidates per recorded iteration in addition to selected BO points.
For 300 iterations the diagnostic requests can total up to 3,000 candidate
evaluations; actual expensive work depends on available candidates and oracle
caching. This is a local extension, absent from the upstream runner.

The diagnostic helper snapshots and restores Python-visible counters and RNG
state. That does not undo external work or make these calls free. A toy oracle
probe performed one actual evaluation while its visible evaluation counter was
restored to zero. For the molecular wrapper, the resolved diagnostic function
uses a raw evaluator, so ordinary budget/observer accounting can be bypassed.
These post-run values do not feed back into the completed BO choices, but
total oracle cost is not represented by the nominal optimization budget alone.

Recommended correction: make objective-valued diagnostics explicitly opt-in
and report their cost separately. Keep acquisition-only diagnostics available.
This finding confirms the call path questioned in TD-11; production external
oracle side effects were not exercised by this audit.

### F4. Medium: novelty is checked for stopping but not applied to EI choices

Local references: `solvers/cowboys.py:191-208`.

`unique_idx` counts newly decoded strings and controls termination. It is never
used to subset the candidate pool passed to `optimize_acqf_discrete`. That pool
contains all accepted states, including previously observed structures and
duplicates. Mapping chosen fingerprints back with `list.index` also selects
the first matching fingerprint, which need not uniquely identify a molecule.

The toy probe accepted one old molecule and one new molecule, satisfied the
novelty stopping test, and then returned the old molecule because it had the
highest acquisition score. This behavior is inherited from upstream. The
paper does not explicitly require a no-repeats policy; this is a budget and
implementation-consistency issue, rather than a claim that every repeated
evaluation contradicts its pseudocode.

Recommended correction: define novelty at molecule level, filter the eligible
pool before EI selection and retain direct latent/structure indices. Raw SELFIES
string uniqueness does not guarantee canonical molecular uniqueness. Document
this policy as a deliberate departure from the released implementation.

### F5. Medium, conditional: threshold relaxation mixes old and new likelihoods

Local references: `solvers/cowboys.py:156,162-165,174-180`.

If sampling continues beyond a block of `n_steps`, the implementation multiplies
`best_so_far` by `0.95` and constructs a new PI acquisition. It does not
recompute `log_lik_currents` at the new threshold. Subsequent acceptance ratios
can compare a candidate's likelihood at the new threshold with a current
state's likelihood at the old one. The fallback also changes the threshold
used for final EI selection.

A forced three-step toy case recorded likelihood thresholds
`[1, 1, 1, 0.95]` and a spurious log ratio `0.1` where the probe's likelihood
was constant over structures. This path is inherited from upstream; its
comment claiming it is never triggered is not a guarantee. Moreover,
multiplication does not lower a zero threshold and raises a negative one.
The paper's Algorithm 3 keeps its threshold fixed and reruns PCN as needed.

Recommended correction: explicitly choose the fallback policy. A paper-target
implementation should keep the incumbent threshold fixed; if relaxation is
retained, recompute cached likelihoods and document the changed target and EI
threshold. Bounded failure behavior needs a separately specified policy.

## Matches and paper-versus-code differences

| Component | Local versus authors' repository | Comparison with supplied paper |
|---|---|---|
| Exact GP on molecular features | Same fitting method AST | Matches the structural GP design |
| Count fingerprints | Same radius-3 Morgan counts, folded to 2,048 entries | Matches count-feature Tanimoto; implementation specifics come from released code |
| Kernel | Same `ScaleKernel(TanimotoKernel())` | Matches Section 2.1's scaled dot-product Tanimoto, with library numerical stabilization |
| PI likelihood | Same `LogProbabilityOfImprovement` | Matches Equation (5)'s GP improvement factor, subject to F1 |
| Deterministic decoding | All VAE method ASTs match; token-wise argmax | Matches Section 5.1; stochastic decoding would change the specified practical approximation |
| Default checkpoint | Active data copy and packaged copy both match upstream Git blob `ab3522fcb2cb1aae773a7925c1fada9c5eb8bd43`, 92,838,066 bytes | Consistent with Section 7.2's updated 128-dimensional VAE; verified without downloading weights |
| Chain start | Same best recorded latent, repeated across chains | Matches Appendix A's best-so-far warm start, subject to initial reconstruction issue F2 |
| Accepted states | Same accepted-move storage; no burn-in | Matches Appendix A's stated approximation |
| Candidate utility | Same greedy discrete `qLogExpectedImprovement` | Consistent with Appendix A's qEI subset heuristic; log/smoothed Monte Carlo implementation is not a claim of exact numerical identity with analytic EI |
| Chain count | Both default to 10 chains, batch size 1 | Section 7 says one chain for batch size 1; Appendix B also studies 10 chains |
| Minimum steps | Both use 100 | Matches the stated 100 steps, but fallback can extend the run |
| Step-size adaptation | Both update log-beta with diminishing gain `0.5 (n+1)^(-0.6)`, target `0.234`, clip to `[1e-6, 1]`, retain betas across BO iterations | Algorithm 4 instead initializes beta to 0.1 per chain and shows additive `0.1 (alpha - 0.243)` updates |
| Initial design and horizon | Local loader uses ten molecules; sweep uses 300 additional iterations, seeds 1-5 | Matches Section 7.2's design size and horizon; generic Algorithm 2 instead illustrates drawing the initial design from the VAE |
| Invalid SMILES handling | Both replace unparseable molecules with toluene for fingerprints | Undocumented fallback, not evidence that the actual objective evaluated toluene |

The output directory name containing `stoch_sampling` is not evidence of
stochastic VAE decoding: the inspected decoder uses argmax, in agreement with
the paper's practical approximation.

### Do not add an extra Gaussian prior ratio mechanically

Algorithm 4 prints an acceptance ratio containing both likelihood and prior,
whereas released/local code uses the likelihood ratio alone. For standard pCN
in the **correct prior coordinates**, its proposal is reversible with respect
to the Gaussian prior: the prior cancels against the Hastings proposal ratio.
The likelihood-only acceptance rule is therefore appropriate for the target in
Equation (5). Inserting an additional Gaussian-density ratio without the
proposal correction would double-count the prior. This mathematical
interpretation explains a paper/code discrepancy; it does not resolve F1 or
establish exact sampling under ongoing adaptation and finite chains.

## Verification and limits

- Existing Python 3.10.20 `hdbo_benchmark` environment; CPU PyTorch 2.10.0.
- Three dependency-free structural tests passed:
  `python -m unittest tests.test_structure -v`.
- Bounded toy probes called the real local `next_candidate` and transformation
  methods, replacing GP fitting, PI/EI and proposal noise. They demonstrated
  F2, F4 and F5 and checked candidate/RNG preservation with diagnostics on/off
  in that controlled case. Another probe exercised the actual diagnostic
  helper with a toy oracle. These are not end-to-end BO validations.
- One inference-only check encoded/decoded ten cached molecules using existing
  CPU VAE weights. No networks were trained, weights downloaded, GPUs used,
  sweeps launched or real objectives called.
- Source AST comparison and local file hashing confirmed the matches above.
  Machine-readable traces are in
  [`cowboys_fidelity.json`](verification/cowboys_fidelity.json).
- The known Windows BLAS GP-fitting failure was neither rerun nor repaired.
  See the [original verification evidence](verification/phase1.json). The existing
  [PMO oracle limitations](PMO_DIAGNOSTICS.md) also remain.
- No claim is made about paper-score reproduction, corrected optimization
  trajectories, other latent dimensions or all seeds. Historical results and
  checkpoints were preserved.
- The current seed-1 inference latents did not reproduce the saved Phase 1
  latent array (maximum absolute normalized-coordinate difference about 0.301).
  Therefore the 7/10 reconstruction measurement describes the current weights
  and cached seed-1 sample, not a reconstruction audit of that historical run.
  The reason for that historical difference was not established here.

The appropriate next scientific change is to fix F1 and preserve the evaluated
structures in F2, with regression checks, then explicitly choose whether the
baseline targets the released implementation or the paper's reported
single-chain protocol. Do not silently change the beta update or add a prior
factor to force a superficial pseudocode match.
