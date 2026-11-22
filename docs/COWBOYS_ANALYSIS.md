# Read-only COWBOYS experiment report

From the repository root, use the Python 3.10 research environment:

```powershell
conda run -n hdbo_benchmark python scripts/analysis/analyze_cowboys.py --input-root results/cowboys/merged_20261121 --output-root reports/figures/analysis_cowboys/merged_20261121
```

This produces `report.md`, methods in `README.md`, inventory, tidy per-iteration
data, seed/task summaries, T00-T05 as CSV/HTML, thirteen core figures as PNG/PDF,
plot denominators and `diagnostics.json` with thresholds, validation and source
hashes. `insights.md` adds a plain-language selection walkthrough, metrics per
100 evaluations, and three figures under `figures/explainer/`: histograms of all
evaluated scores (including initialization), early/middle/late score distributions,
and actual versus predicted record wins per 100 evaluations. Histogram counts,
stage denominators and simple per-seed/task metrics are saved as additional CSVs.
Output must be outside the raw-results tree. No objective, optimizer,
GP fit, generative model or experiment is run. Repeated task/seed identities are
inventoried and excluded; use a narrower input root to select one attempt.

Repeat `--task NAME` or `--seed N` to filter. `--planned-steps` defaults to 300;
`--include-optional` adds exploratory diagnostics separately. The module entry
point is `python -m hdbo_benchmark.evaluation.cowboys_analysis`.

The report uses available independent seeds, sample SD, trailing 25-step windows
with ten valid observations, and no missing-value imputation. AUC is mean logged
best-so-far over a common observed horizon within each task, distinct from the
older exporter's initialization-inclusive trapezoidal AUC. Logged novelty is
retained with its original radius-3 count fingerprint. New structural metrics
use Morgan radius 2, 2048 bits, chirality off. Quality thresholds are retrospective
task-level percentiles, and diverse-hit counts are chronological greedy proxies.
GP diagnostics require aligned pre-evaluation predictions; unselected-candidate
selection scores never become observed objectives. Missing initial history
explicitly limits structural metrics to BO-selected molecules.

The generator-variety logs count distinct molecular identities from accepted
pCN moves, excluding rejected proposal decodings. Their cross-iteration novelty
memory differs from the oracle-evaluated history. The minimum 100 sampler steps
are attempted moves, not 100 distinct candidates. Selection scores the entire
eligible unevaluated pool with qLogEI and chooses one molecule. The saved top-10
ranking is a truncated diagnostic view, not a restriction on selection; the
untruncated eligible pool count is also available in the saved sampling metrics.

The November 21 merged working set contains mixed 10,000/50,000 sampling caps
and partial albuterol runs. The overall report is descriptive; configuration
summaries retain that distinction. Neither selected-point GP accuracy nor
novelty/error associations establish global accuracy or causal effects.

Lightweight analysis checks (no oracle calls):

```powershell
conda run -n hdbo_benchmark python -m unittest tests.test_cowboys_analysis tests.test_structure -v
```
