"""Plain-language views of saved COWBOYS outcomes; no objective/model calls."""

from pathlib import Path
import textwrap

import numpy as np
import pandas as pd

from .metric_io import io_path


STAGES = [("early", 1, 100, "Steps 1-100"),
          ("middle", 101, 200, "Steps 101-200"),
          ("late", 201, np.inf, "Steps 201+")]
ELIGIBLE = "sample_unique_decoded_molecules_not_in_observed_history"


def evaluated_scores(data, runs):
    """Retain evaluation multiplicity; initial observations appear exactly once."""
    rows = []
    for run in runs:
        inv = run["inventory"]
        for index, (key, y, _, _) in enumerate(run["initial"]):
            rows.append({"task": inv["task"], "seed": inv["seed"], "phase": "initial",
                         "iteration": 0, "initial_index": index, "canonical_smiles": key,
                         "y": y if key is not None and np.isfinite(y) else np.nan})
    for row in data.itertuples():
        rows.append({"task": row.task, "seed": row.seed, "phase": "BO",
                     "iteration": row.iteration, "initial_index": np.nan,
                     "canonical_smiles": row.canonical_smiles, "y": row.y})
    return pd.DataFrame(rows)


def score_histograms(evaluations, data, tasks, n_bins=25):
    """All-outcome counts and stage distributions with seed-first normalization.

    Empty observed bins are zero. A seed/stage with no valid scores is missing,
    not an all-zero distribution. Shared task edges include initialization.
    """
    absolute, stage_seed, stage_task, distributions = [], [], [], []
    for task in tasks:
        all_task = evaluations[evaluations.task == task]
        valid = all_task.dropna(subset=["y"])
        if valid.empty:
            continue
        lo, hi = valid.y.min(), valid.y.max()
        if lo == hi:
            width = max(abs(lo) * .05, .01)
            lo, hi = lo - width, hi + width
        edges = np.linspace(lo, hi, n_bins + 1)
        counts_all = np.histogram(valid.y, bins=edges)[0]
        mode = int(np.argmax(counts_all))
        distributions.append({"task": task, "n_valid": len(valid), "n_missing": len(all_task) - len(valid),
                              "n_initial": int((valid.phase == "initial").sum()),
                              "n_bo": int((valid.phase == "BO").sum()),
                              "median": valid.y.median(), "p25": valid.y.quantile(.25),
                              "p75": valid.y.quantile(.75), "mode_bin_low": edges[mode],
                              "mode_bin_high": edges[mode + 1], "mode_bin_count": int(counts_all[mode]),
                              "mode_bin_percent": 100 * counts_all[mode] / len(valid)})
        for seed in sorted(all_task.seed.unique()):
            for phase in ("initial", "BO"):
                g = all_task[(all_task.seed == seed) & (all_task.phase == phase)]
                ys = g.y.dropna()
                counts = np.histogram(ys, bins=edges)[0]
                for i, count in enumerate(counts):
                    absolute.append({"task": task, "seed": seed, "phase": phase, "bin": i,
                                     "low": edges[i], "high": edges[i + 1], "count": int(count),
                                     "n_valid": len(ys), "n_missing": int(g.y.isna().sum())})
        task_data = data[data.task == task]
        for seed in sorted(task_data.seed.unique()):
            for stage, start, end, _ in STAGES:
                g = task_data[(task_data.seed == seed) & (task_data.iteration >= start) & (task_data.iteration <= end)]
                ys = g.y.dropna()
                counts = np.histogram(ys, bins=edges)[0]
                for i, count in enumerate(counts):
                    stage_seed.append({"task": task, "seed": seed, "stage": stage, "bin": i,
                                       "low": edges[i], "high": edges[i + 1], "count": int(count),
                                       "n_valid": len(ys), "n_missing": int(g.y.isna().sum()),
                                       "percent": 100 * count / len(ys) if len(ys) else np.nan})
    seeds = pd.DataFrame(stage_seed, columns=["task", "seed", "stage", "bin", "low", "high", "count", "n_valid", "n_missing", "percent"])
    if len(seeds):
        for keys, group in seeds.groupby(["task", "stage", "bin"], sort=True):
            present = group[group.n_valid > 0]
            stage_task.append({"task": keys[0], "stage": keys[1], "bin": keys[2],
                               "low": group.low.iloc[0], "high": group.high.iloc[0],
                               "mean_percent": present.percent.mean(), "sd_percent": present.percent.std(ddof=1),
                               "n_seeds": len(present), "n_valid": int(present.n_valid.sum()),
                               "count": int(present["count"].sum())})
    return (pd.DataFrame(absolute, columns=["task", "seed", "phase", "bin", "low", "high", "count", "n_valid", "n_missing"]),
            seeds, pd.DataFrame(stage_task, columns=["task", "stage", "bin", "low", "high", "mean_percent", "sd_percent", "n_seeds", "n_valid", "count"]),
            pd.DataFrame(distributions, columns=["task", "n_valid", "n_missing", "n_initial", "n_bo", "median", "p25", "p75",
                                                "mode_bin_low", "mode_bin_high", "mode_bin_count", "mode_bin_percent"]))


def plain_metrics(data):
    rows = []
    for (task, seed), group in data.groupby(["task", "seed"], sort=True):
        g = group.sort_values("iteration").copy()
        initial_centers = g.initial_circles.iloc[0]
        g["new_centers"] = g.circles.diff()
        g.loc[g.index[0], "new_centers"] = g.circles.iloc[0] - initial_centers
        g["high_quality"] = np.where(g.y.notna(), (g.y >= g.tau).astype(float), np.nan)
        for period, block in [("first100", g[g.iteration <= 100]),
                              ("last100", g[g.iteration > g.iteration.max() - 100])]:
            n = int(block.y.notna().sum())
            row = {"task": task, "seed": seed, "period": period,
                   "start": int(block.iteration.min()), "end": int(block.iteration.max()),
                   "n_valid_outcomes": n, "n_decisions": int(block.success.notna().sum()),
                   "record_wins_per100": 100 * block.success.mean(),
                   "high_quality_per100": 100 * block.high_quality.mean(),
                   "new_quality_centers_per100": 100 * block.new_centers.sum() / n if n else np.nan,
                   "predicted_wins_per100": 100 * block.pi.mean(),
                   "absolute_gp_error": block.abs_error.mean(),
                   "n_gp_errors": int(block.abs_error.notna().sum()),
                   "n_pi": int(block.pi.notna().sum()),
                   "eligible_choices": block[ELIGIBLE].mean(),
                   "n_eligible_counts": int(block[ELIGIBLE].notna().sum())}
            rows.append(row)
    seeds = pd.DataFrame(rows)
    aggregates = []
    metrics = ["record_wins_per100", "high_quality_per100", "new_quality_centers_per100",
               "predicted_wins_per100", "absolute_gp_error", "eligible_choices"]
    for (task, period), group in seeds.groupby(["task", "period"], sort=True):
        row = {"task": task, "period": period, "n_seeds": len(group)}
        for metric in metrics:
            row.update({metric + "_mean": group[metric].mean(), metric + "_sd": group[metric].std(ddof=1),
                        metric + "_n_seeds": int(group[metric].notna().sum())})
        aggregates.append(row)
    return seeds, pd.DataFrame(aggregates)


SELECTION_EXPLANATION = """## What happens in one BO iteration?

1. Fit the GP to the molecules whose oracle scores are already known.
2. Start the latent MCMC chain at the best evaluated latent point. Attempt at
   least 100 pCN moves. Decode proposed latent points and use GP probability
   of improvement to guide acceptance. Rejected proposals are not entered in
   the saved sampling-variety counts. Accepted moves can decode to repeated
   molecules, and different latent points can decode to the same molecule.
3. Among accepted moves, collect valid distinct molecular identities that have
   never been oracle-evaluated in this run. If none are available after 100
   moves, continue until one appears or the recorded sampling cap is reached.
   The default is one chain and one selected molecule per BO iteration.
4. Compute qLogEI for the entire eligible pool and choose its highest-scoring
   candidate. This is a GP acquisition score, not an oracle score.
5. Evaluate that one molecule with the oracle and add it to history.

Thus 100 means minimum attempted MCMC steps, not 100 independent candidates,
100 accepted moves, or 100 unique molecules. The current cap is 50,000; these
artifacts also contain historical 10,000-cap runs. The stop condition needs
only one eligible candidate once the minimum steps have been taken.

The saved top-10 "shortlist" is a truncated diagnostic ranking of the eligible
pool, not an extra selection gate. Selection uses the whole eligible pool.
Diagnostic ranking recomputes Monte Carlo acquisition scores, so its rank 1 is
not guaranteed to match the actual selection ranking exactly. A saved count of
10 means at least ten eligible candidates; it does not reveal the full pool.
The logged sample_unique_decoded_molecules_not_in_observed_history gives the
full eligible unique pool size and is used in the simpler summary below.

## What does "new to sampling" mean?

The existing generator-variety fields count accepted MCMC decodings, not every
attempted decode. "New" means absent from the memory of accepted decodings in
earlier BO sampling iterations in this run. This is a different memory from
the oracle-evaluated history. Rejected decodings and the initial design are
not inserted into that sampling memory just because they exist.

Example: suppose A was evaluated initially. Accepted moves in iteration 1
decode A, A, B, C. There are 3 distinct accepted sampled molecules. B and C
are eligible for selection; assume B is chosen and evaluated. In iteration 2,
accepted moves decode B, C, D, D. Again there are 3 distinct sampled molecules,
but only D is new to sampling memory. C and D remain eligible for evaluation:
C was generated before but was never selected. B is excluded because its
oracle score is now known. Sampling can revisit C many times without evaluating it.

These counts describe proposal variety, availability of unevaluated choices,
and eventual oracle evaluations separately. None gives an acceptance fraction
or repeat rate per attempted MCMC move: proposal/acceptance denominators were
not recorded in these artifacts.
"""


def create_explainer(data, runs, output, tasks, titles):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from .cowboys_analysis import markdown_table

    output = Path(output)
    directory = output / "figures" / "explainer"
    io_path(directory).mkdir(parents=True, exist_ok=True)
    evaluations = evaluated_scores(data, runs)
    absolute, stage_seed, stage_task, distributions = score_histograms(evaluations, data, tasks)
    seed_metrics, task_metrics = plain_metrics(data)
    tables = {"evaluated_scores": evaluations, "score_histogram_counts": absolute,
              "score_stage_histograms_per_seed": stage_seed, "score_stage_histograms_per_task": stage_task,
              "score_distribution_summary": distributions, "plain_metrics_per_seed": seed_metrics,
              "plain_metrics_per_task": task_metrics}
    for name, frame in tables.items():
        frame.to_csv(io_path(output / "tables" / (name + ".csv")), index=False, float_format="%.12g")
    colors = ["#2166ac", "#d66022", "#278267"]
    manifest = []
    def grid(title, caption):
        fig, axes = plt.subplots(2, 3, figsize=(16, 9))
        fig.suptitle(title, fontsize=15, y=.985)
        fig.text(.5, .947, textwrap.fill(caption, 145), ha="center", va="top", fontsize=10)
        for ax, task in zip(axes.flat, tasks):
            partial = [r["inventory"] for r in runs if r["inventory"]["task"] == task and r["inventory"]["incomplete"]]
            ax.set_title(titles.get(task, task.replace("_", " ").title()) +
                         ("\nPartial: " + ", ".join(f"s{r['seed']} ({r['completed_steps']} steps)" for r in partial) if partial else ""), fontsize=10)
            ax.grid(alpha=.15)
            ax.spines[["right", "top"]].set_visible(False)
        for ax in list(axes.flat)[len(tasks):]:
            ax.set_visible(False)
        return fig, axes
    def save(fig, name):
        handles, labels = [], []
        for ax in fig.axes:
            h, l = ax.get_legend_handles_labels()
            for handle, label in zip(h, l):
                if label not in labels:
                    handles.append(handle)
                    labels.append(label)
        if handles:
            fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=9)
        fig.tight_layout(rect=(0, .07, 1, .89), h_pad=2)
        for extension in ("png", "pdf"):
            fig.savefig(io_path(directory / f"{name}.{extension}"), dpi=190)
        plt.close(fig)
        manifest.append({"figure": name, "task_order": tasks, "directory": "figures/explainer"})
    fig, axes = grid("Where are all the evaluated scores concentrated?", "Actual saved outcomes, including initialization; pooled counts are descriptive, not independent replicates")
    for ax, task in zip(axes.flat, tasks):
        g = absolute[absolute.task == task]
        if g.empty:
            ax.text(.5, .5, "No valid scores", transform=ax.transAxes, ha="center")
            continue
        agg = g.groupby(["phase", "bin"])["count"].sum()
        bins = g.drop_duplicates("bin").sort_values("bin")
        edges = np.r_[bins.low.to_numpy(), bins.high.iloc[-1]]
        bo = agg.loc["BO"].reindex(bins.bin, fill_value=0).to_numpy()
        initial = agg.loc["initial"].reindex(bins.bin, fill_value=0).to_numpy()
        ax.bar(edges[:-1], bo, width=np.diff(edges), align="edge", color=colors[0], alpha=.8, label="Selected BO evaluations")
        ax.bar(edges[:-1], initial, bottom=bo, width=np.diff(edges), align="edge", color=colors[1], alpha=.9, label="Initial evaluations")
        tau = data.loc[data.task == task, "tau"].iloc[0]
        ax.axvline(tau, color="#444", ls="--", lw=1.1, label="Retrospective high-quality cutoff")
        summary = distributions[distributions.task == task].iloc[0]
        ax.text(.03, .96, f"N={int(summary.n_valid)} ({int(summary.n_initial)} initial + {int(summary.n_bo)} BO)\n"
                f"Median={summary['median']:.4g}; middle 50%: {summary.p25:.4g}-{summary.p75:.4g}",
                transform=ax.transAxes, va="top", fontsize=8, bbox={"facecolor": "white", "alpha": .8, "edgecolor": "none"})
        ax.set(xlabel="Observed objective score (higher is better)", ylabel="Number of recorded evaluations", ylim=(0, None))
    save(fig, "01_all_evaluated_scores_histogram")
    fig, axes = grid("Does the score distribution shift as optimization progresses?", "BO selections only; percentages within each seed/stage, then seed mean +/- sample SD; same score bins as all-score histogram")
    for ax, task in zip(axes.flat, tasks):
        labels = []
        for color, (stage, _, _, label) in zip(colors, STAGES):
            g = stage_task[(stage_task.task == task) & (stage_task.stage == stage)].sort_values("bin")
            if g.empty or not g.n_seeds.max():
                labels.append(f"{label}: no valid data")
                continue
            x = (g.low.to_numpy() + g.high.to_numpy()) / 2
            mean, sd = g.mean_percent.to_numpy(), g.sd_percent.to_numpy()
            ax.plot(x, mean, color=color, lw=2, label=label)
            ax.fill_between(x, np.maximum(0, mean - sd), np.minimum(100, mean + sd), color=color, alpha=.12)
            labels.append(f"{label}: N={int(g.n_valid.iloc[0])}, seeds={int(g.n_seeds.iloc[0])}")
        ax.text(.02, .97, "\n".join(labels), transform=ax.transAxes, va="top", fontsize=8,
                bbox={"facecolor": "white", "alpha": .8, "edgecolor": "none"})
        ax.set(xlabel="Observed objective score (higher is better)", ylabel="Evaluations in score bin (%)", ylim=(0, None))
    save(fig, "02_score_distribution_by_stage")
    fig, axes = grid("How often do evaluations find a new best?", "Per 100 evaluations: first 100 observed, last 100 observed, and GP-predicted late wins; seed mean +/- sample SD")
    for ax, task in zip(axes.flat, tasks):
        g = task_metrics[task_metrics.task == task].set_index("period")
        if g.empty:
            ax.text(.5, .5, "No supported run", transform=ax.transAxes, ha="center")
            continue
        bars = [("first100", "record_wins_per100", "Actual early", colors[0]),
                ("last100", "record_wins_per100", "Actual late", colors[1]),
                ("last100", "predicted_wins_per100", "Predicted late", colors[2])]
        for i, (period, metric, label, color) in enumerate(bars):
            m, sd = g.loc[period, metric + "_mean"], g.loc[period, metric + "_sd"]
            ax.bar(i, m, color=color, alpha=.8, label=label)
            if np.isfinite(sd):
                ax.errorbar(i, m, yerr=np.array([[min(sd, m)], [min(sd, 100 - m)]]), color="#333", capsize=4)
            if np.isfinite(m):
                ax.text(i, m + (sd if np.isfinite(sd) else 0) + .5, f"{m:.1f}", ha="center", fontsize=9)
        ax.set(xticks=[0, 1, 2], xticklabels=["Actual early", "Actual late", "Predicted late"],
               ylabel="New-record evaluations per 100", ylim=(0, None))
    save(fig, "03_record_wins_per_100_evaluations")
    progress, model = [], []
    def value(v):
        return f"{v:.3g}" if np.isfinite(v) else "missing"
    def pair(group, metric):
        a, b = group.loc["first100", metric + "_mean"], group.loc["last100", metric + "_mean"]
        return value(a) + " -> " + value(b)
    for task in tasks:
        g = task_metrics[task_metrics.task == task].set_index("period")
        if g.empty:
            continue
        progress.append({"Task": task, "Record wins /100": pair(g, "record_wins_per100"),
                         "High-quality scores /100": pair(g, "high_quality_per100"),
                         "New distinct high-quality hits /100": pair(g, "new_quality_centers_per100")})
        model.append({"Task": task, "Typical absolute GP error": pair(g, "absolute_gp_error"),
                      "Late predicted wins vs actual /100": value(g.loc['last100', 'predicted_wins_per100_mean']) + " vs " + value(g.loc['last100', 'record_wins_per100_mean']),
                      "Unevaluated choices available": pair(g, "eligible_choices")})
    usable_tasks = sorted(task_metrics.task.unique())
    def changed(metric, decrease=False):
        count = 0
        for task in usable_tasks:
            g = task_metrics[task_metrics.task == task].set_index("period")
            a, b = g.loc["first100", metric + "_mean"], g.loc["last100", metric + "_mean"]
            count += bool(b < a) if decrease else bool(b > a)
        return count
    optimistic = sum(
        row.predicted_wins_per100_mean > row.record_wins_per100_mean
        for row in task_metrics[task_metrics.period == "last100"].itertuples())
    score_overview = distributions[["task", "n_valid", "median", "p25", "p75"]].copy().round(5)
    score_overview["Most populated score bin"] = [
        f"{r.mode_bin_low:.3g}-{r.mode_bin_high:.3g} ({r.mode_bin_percent:.1f}% of outcomes)"
        for r in distributions.itertuples()]
    parts = ["# COWBOYS: the results in plain language", SELECTION_EXPLANATION,
             "## Three figures to start with",
             "1. [All evaluated scores](figures/explainer/01_all_evaluated_scores_histogram.png): bars show where actual oracle scores accumulate. "
             "Both initial and BO evaluations are included; every observation is counted once, without deduplication. Empty score bins are genuine zeros; missing scores are excluded.",
             "2. [Early, middle and late score distributions](figures/explainer/02_score_distribution_by_stage.png): a shift to the right means higher selected scores; "
             "a narrower peak means scores concentrate. Similar scores do not establish similar chemistry. This view gives each available seed equal weight within each stage.",
             "3. [Wins per 100 evaluations](figures/explainer/03_record_wins_per_100_evaluations.png): actual early wins, actual late wins, and model-predicted late wins. "
             "The probability is a diagnostic under the GP observation model, not the acquisition used to select candidates.",
             "## What stands out",
             f"High-quality scores become more frequent in {changed('high_quality_per100')}/{len(usable_tasks)} tasks, while record wins become less frequent in "
             f"{changed('record_wins_per100', decrease=True)}/{len(usable_tasks)}. This distinguishes repeatedly finding good molecules from repeatedly breaking the record. "
             f"New distinct high-quality representatives become more frequent in {changed('new_quality_centers_per100')}/{len(usable_tasks)} tasks under the retrospective cutoff; "
             "this does not support a blanket claim that novel high-quality discovery has stopped.",
             f"The eligible pool contracts in {changed('eligible_choices', decrease=True)}/{len(usable_tasks)} tasks, and typical selected-point GP error falls in "
             f"{changed('absolute_gp_error', decrease=True)}/{len(usable_tasks)}. Predicted late record wins exceed observed wins in {optimistic}/{len(usable_tasks)}. "
             "These coincide descriptively; they do not establish a causal link between concentration, predictability and stagnation.",
             "## Easy-to-read metrics",
             "Arrows compare the first 100 with the last 100 observed steps in each run. Values are means of seed-level metrics, with sample SD and denominators in the CSVs. "
             "Partial runs end earlier, and short runs may have overlapping windows. These are rates per 100, not a claim of observing 100 events in every block.",
             markdown_table(pd.DataFrame(progress)), markdown_table(pd.DataFrame(model)),
             "A record win beats everything evaluated earlier. A high-quality score passes the shared retrospective task cutoff. "
             "A new distinct high-quality hit additionally passes the chronological ECFP4 distance >=0.4 rule. These answer different questions: "
             "many good molecules can be discovered after records become rare. The cutoff is not an external success standard and can leave some seeds with zero qualifying hits.",
             "The full eligible-choice count is easier to interpret than the top-10 diagnostic because it is not truncated. "
             "Typical absolute GP error is mean abs(observed - predicted) on chosen molecules, in the task's score units; lower is better within task.",
             "## Score concentration",
             markdown_table(score_overview.rename(columns={"n_valid": "Evaluations", "p25": "25th percentile", "p75": "75th percentile"})),
             "The middle 50% interval and histogram show score concentration. They do not measure structural concentration. "
             "Histogram binning is 25 equal-width bins per task spanning all valid evaluated scores; all stages use those same edges. "
             "Raw pooled counts are descriptive and give longer runs more observations. The stage view instead normalizes each seed separately. "
             "Partial seeds are not filled in, and absent seed/stage distributions are missing, not all-zero histograms. "
             "A most-populated bin depends on bin width; it is a descriptive score interval, not a chemical cluster. Multiple score peaks can reflect different seeds, "
             "so a pooled histogram alone does not show whether an individual run visits all of those score ranges.",
             "## Interpretation limits",
             "This collection mixes 10,000 and 50,000 sampling caps and includes partial runs. Distinct seed runs are the replicates. "
             "GP errors concern adaptively selected molecules and remain selection-biased. A growing evaluated history can itself reduce nearest-neighbor distances. "
             "No objective scores exist for unselected candidates in these diagnostics; their quality cannot be inferred. "
             "The 13 detailed figures remain in [the full report](report.md). All new views use existing artifacts only."]
    io_path(output / "insights.md").write_text("\n\n".join(parts) + "\n", encoding="utf-8")
    return {"figures": manifest, "n_valid_evaluations": int(evaluations.y.notna().sum()),
            "n_initial_evaluations": int(((evaluations.phase == "initial") & evaluations.y.notna()).sum()),
            "histogram_bins_per_task": 25, "summary": "insights.md",
            "sampling_count_semantics": "Unique molecular identities from accepted pCN moves only; rejected decodings are not counted"}
