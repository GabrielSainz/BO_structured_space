# Corrected COWBOYS baseline

This implements the fixes authorized after the
[2026-11-17 audit](COWBOYS_FIDELITY_AUDIT.md). The implementation identifier in
new status files is `cowboys_corrected_pcn_v1`. Historical results and weights
are unchanged. This is a corrected baseline, not a byte-for-byte reproduction
of the authors' released implementation.

## Run the experiments

Use the existing **activated Python 3.10 `hdbo_benchmark` environment**. From
Git Bash or another Bash shell with Conda initialized:

```bash
conda activate hdbo_benchmark
bash scripts/experiments/run.sh
```

On this Windows machine, PowerShell can launch Git Bash inside the activated
environment explicitly:

```powershell
& 'C:/Users/gabri/miniconda3/Scripts/conda.exe' run --no-capture-output -n hdbo_benchmark 'C:/Program Files/Git/bin/bash.exe' scripts/experiments/run.sh
```

Calling `envs/hdbo_benchmark/python.exe` directly from the base environment
still fails in Windows BLAS/threadpool handling. With `conda run`, the bounded
real COWBOYS check completed using the existing GP optimizer and dependencies.
No library installation, downgrade or optimizer substitution was needed.

The script runs focused regression checks, then a separate one-iteration real
albuterol preflight. If either fails, it stops before the sweep. The preflight
uses ten initial evaluations plus one BO evaluation and has its own output
subdirectory. Its evaluations are additional verification work, not part of any
reported full run.

Defaults are CPU execution, disabled W&B, cached/offline generative models,
all 25 Table 1 tasks and seeds 1-5. Each run uses ten initial molecules, a
128-dimensional VAE, batch size one, one chain, at least 100 pCN steps and 300
additional BO iterations. The task list and reported reference values are in
[`configs/cowboys_paper_table1.json`](../configs/cowboys_paper_table1.json),
transcribed from [Table 1](https://arxiv.org/html/2507.03910v1#S7.T1).
The paper's Section 7.2 and released sweep specify 300 BO steps; its table
caption uses the wording 300 evaluations. This script follows the former
protocol, for 310 nominal objective evaluations per full run.

New outputs go to `results/cowboys/<RUN_ID>/`. A fresh run ID is generated
automatically; an existing directory is rejected by the sweep to avoid
overwriting results. Each run has a log, status JSON, history NPZ, iteration
JSON and best-performance NPY. The NPZ now includes `structures`, aligned with
`x` and `y`, so original initial molecules remain recoverable. Status files
record seed, initialization size, sampler settings and checkpoint SHA-256.

For a deliberately small pipeline check:

```bash
FUNCTIONS=albuterol_similarity SEEDS=1 MAX_ITER=1 bash scripts/experiments/run.sh
```

`SKIP_PREFLIGHT=1` skips the unit tests and separate real preflight when they
have already passed. `RUN_ID`, `PYTHON_BIN` and `HDBO_TORCH_DEVICE` can also be
set explicitly.

### Run pending task and seed pairs

On 2026-11-20, the user authorized raising the default `max_sampling_steps`
from 10,000 to 50,000. This only extends the finite pCN sampling allowance
within each BO iteration. The one-chain configuration, minimum 100 steps,
fixed threshold, novelty rule, beta adaptation and objective budget remain
unchanged. New status JSON files record `max_sampling_steps: 50000`; historical
artifacts remain unchanged. Treat these as changed-cap experiments, and retain
the original failed/interrupted attempts when reporting results. A larger cap
does not guarantee successful completion.

`configs/cowboys_pending.tsv` contains only albuterol seeds 1 and 2. The other
23 task/seed pairs completed 300 steps in the consolidated results. These two
albuterol attempts still hit the 50,000-step sampling cap; restarting with the
same settings may repeat the failure. The launcher starts fresh 300-step runs,
not resumptions of partial observation archives. Run from the repository root
in PowerShell:

```powershell
Remove-Item Env:FUNCTIONS, Env:SEEDS, Env:DRY_RUN -ErrorAction SilentlyContinue
$env:RUNS_FILE = 'configs/cowboys_pending.tsv'
$env:RUN_ID = "cowboys_pending_cap50000_$(Get-Date -Format 'yyyyMMdd_HHmmss')"
$env:MAX_ITER = '300'
$env:HDBO_TORCH_DEVICE = 'cpu'
& 'C:/Users/gabri/miniconda3/Scripts/conda.exe' run --no-capture-output -n hdbo_benchmark 'C:/Program Files/Git/bin/bash.exe' scripts/experiments/run.sh
```

Keep `FUNCTIONS` and `SEEDS` unset when using `RUNS_FILE`; combining them is
rejected to avoid an unintended Cartesian product. The TSV is validated for
duplicate pairs before any experiment is launched and copied into the fresh
output directory as `run_plan.tsv`. `DRY_RUN=1` prints only the selected pairs,
without tests, preflight, objectives, models or output-directory creation.
The normal launch retains the existing separate one-step preflight and offline
post-run metrics export. The launcher rejects an existing output directory.

`tests.test_cowboys_sweep` checks pending-pair selection, Cartesian sweep order
and duplicate/ambiguous plan rejection using dry runs without models or objectives.

## What changed scientifically

- pCN now proposes in raw VAE coordinates with prior `N(0, I)`. Only the
  solver/objective boundary uses normalized coordinates. Proposals are not
  clipped to the normalization box; bounds are an affine coordinate map.
  The Gaussian prior still cancels against the pCN Hastings proposal ratio.
- The wrapper retains original supervised molecules. COWBOYS fits its GP on
  those actual evaluated structures, then appends the actual deterministic
  decodings of each evaluated candidate. Initial reconstructions are never
  rescored or substituted for original observations.
  An initial encoder mean remains an approximate chain warm start if the best
  original molecule is not reconstructed exactly.
- EI eligibility uses valid, canonically distinct molecules absent from the
  observed history. Acquisition scores select latent indices directly,
  including when different molecules have identical fingerprints.
- The incumbent threshold stays fixed throughout sampling and EI selection.
  If no novel valid molecule is found in 50,000 pCN steps, the run fails
  explicitly rather than relaxing its target or looping forever. This cap is
  a documented operational addition; direct solver users can configure
  `max_sampling_steps`.
  The original corrected baseline used a 10,000-step cap; the current default
  was raised under explicit user authorization on 2026-11-20, as documented above.
- The default is one chain and 100 minimum steps, matching the paper's reported
  single-point setup. The released implementation's diminishing log-beta
  adaptation, target 0.234 and persistence between BO iterations are retained.
  These differ from Appendix A's illustrative additive update. No extra prior
  factor or undocumented replacement adaptation was introduced.
- Invalid/empty molecules are rejected, not assigned toluene fingerprints.
  Count Morgan radius-3/2,048 fingerprints, scaled Tanimoto GP, PI sampling,
  log-EI selection, NaN penalty, initial sample size and objective directions
  remain as specified. Both intended exact-GP GPyTorch settings now enter their
  context managers correctly.
- Result saving no longer calls diagnostic objectives by default. Extra
  post-run oracle work requires `--evaluate-top-candidate-objectives`; that
  flag is explicitly disabled by the sweep. Acquisition-only top-k records
  remain available. The shared runner's opt-in default applies to its other
  methods too; their sampling algorithms were not modified.

Direct API callers must provide original observations through the molecular
wrapper, or pass `initial_structures=...` to `set_vae_and_bounds`. Silently
recovering original molecules from lossy encodings is impossible. Import/CLI
aliases remain supported; old solver-object resume behavior was not certified.

## Results and their limits

Seventeen focused checks passed, including coordinate scale/centering,
unbounded latent proposals, original GP features, novelty, fingerprint
collisions, fixed zero/negative thresholds, bounded failure, saved observation
alignment, no diagnostic oracle calls by default and exclusion of failed or
partial paper comparisons. The shell script also passed Bash syntax checking.

A real CPU albuterol run with seed 1 completed one BO iteration under proper
Conda activation: 11 observations, selected score 0.1449275362 and best score
0.3294117647 (the best initial molecule). A separate one-task/one-seed/one-step
execution of `run.sh` completed and generated the comparison artifacts. These
checks validate execution, not performance after 300 iterations.

Evidence: [validation summary](verification/cowboys_corrected_validation.json),
[shell pipeline](verification/cowboys_script_smoke.json), and
[classifier preflight](verification/cowboys_classifier_preflight.json).

After the sweep, `paper_comparison.csv` and `paper_comparison.json` show final
best-score means, local sample standard deviations and differences from the
reported paper means. They count only completed 300-step runs with matching
initialization size, checkpoint, chain/step settings and disabled diagnostic
objective calls. Missing, partial, failed and incompatible runs are excluded
explicitly. Fewer than five eligible seeds is marked incomplete; a one-step
smoke run is never reported as a paper result. Local SD uses the sample
definition; the source caption does not specify its variance denominator.

To regenerate the comparison without evaluating any objectives:

```bash
python -m hdbo_benchmark.evaluation.compare_cowboys_paper results/cowboys/<RUN_ID>
```

JNK3/GSK3beta classifier compatibility remains an independent runtime issue;
see [PMO diagnostics](PMO_DIAGNOSTICS.md). They are included in the sweep, which
continues after failures, records their nonzero exit codes and returns nonzero
overall. Failed tasks are never converted to zero-valued successes. Raw
`sa_tdc` remains in the source's maximizing convention; this should not be
interpreted as minimizing synthetic difficulty or as the standard 23-task PMO
suite. No scores or oracle models were altered to match reference values.

Passing these checks does not guarantee the paper's mean scores. That requires
the full five-seed experiment and compatible oracles; intentional corrections
also change trajectories relative to the released code.
