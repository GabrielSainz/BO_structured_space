"""Causal, offline diagnostics of selected-candidate EI and observed record gains."""

import csv
import math
from pathlib import Path
from statistics import mean

import numpy as np

from .metric_io import digest_file, display_path, io_path, write_json


def _finite(value):
    return float(value) if value is not None and math.isfinite(float(value)) else None


def _cumulative_available(values):
    """An unknown increment makes all later totals unknown, never zero-filled."""
    total, result = 0., []
    for value in values:
        total = total + value if total is not None and value is not None else None
        result.append(total)
    return result


def analyze_improvements(run):
    if run["task"]["direction"] != "maximize" or run["task"]["score_transform"] != "identity":
        raise ValueError("EI diagnostics currently require the recorded maximizing objective scale")
    if run["design"]["batch_size"] != 1:
        raise ValueError("This EI diagnostic requires one evaluated molecule per BO step")
    initial = [e for e in run["evaluations"] if e["phase"] == "initial"]
    if len(initial) != run["design"]["n_initial_observed"]:
        raise ValueError("The complete initial design is required for a causal incumbent")
    def eligible_score(e):
        return _finite(e["score_for_metrics"]) if e.get("valid_molecule") is not False else None
    initial_scores = [eligible_score(e) for e in initial]
    initial_scores = [score for score in initial_scores if score is not None]
    if not initial_scores:
        raise ValueError("No eligible initial score; cannot reconstruct the incumbent")
    observations = sorted([e for e in run["evaluations"] if e["phase"] == "optimization"
                           and e["bo_step"] is not None], key=lambda e: e["bo_step"])
    if not observations or [e["bo_step"] for e in observations] != list(range(1, len(observations) + 1)):
        raise ValueError("A contiguous positive BO-step grid is required")
    incumbent = initial_best = max(initial_scores)
    states = {p["bo_step"]: p for p in run.get("trajectory", [])}
    rows = []
    for e in observations:
        score = eligible_score(e)
        before = incumbent
        improvement = max(0., score - before) if score is not None else None
        if score is not None:
            incumbent = max(incumbent, score)
        p = e.get("pre_evaluation_prediction") or {}
        mu, sigma = _finite(p.get("mean")), _finite(p.get("latent_std"))
        predictive_std = _finite(p.get("predictive_std"))
        recorded_incumbent = _finite(p.get("incumbent_for_acquisition"))
        threshold_matches = (recorded_incumbent is not None and
                             math.isclose(recorded_incumbent, before, rel_tol=1e-10, abs_tol=1e-12))
        recorded_ei = _finite(p.get("analytic_expected_improvement"))
        ei = recorded_ei if threshold_matches and recorded_ei is not None and recorded_ei >= 0 else None
        mean_advantage = max(0., mu - before) if mu is not None and threshold_matches else None
        uplift = None
        reason = None
        if ei is not None and mean_advantage is not None:
            if ei + 1e-12 < mean_advantage:
                reason = "Recorded EI is smaller than the positive mean advantage; decomposition unavailable"
            else:
                uplift = max(0., ei - mean_advantage)
        if ei is None:
            reason = ("Recorded EI incumbent differs from the observed prefix" if recorded_incumbent is not None
                      and not threshold_matches else "No comparable recorded analytic EI")
        acq = e.get("recorded_acquisition") or {}
        log_ei = (_finite(acq.get("selection_score"))
                  if acq.get("convention") == "qLogExpectedImprovement" else None)
        proxy = None
        if log_ei is not None and threshold_matches:
            try:
                proxy = _finite(math.exp(log_ei))
            except OverflowError:
                pass
        residual = score - mu if score is not None and mu is not None else None
        rows.append({
            "bo_step": e["bo_step"], "observation_index": e["observation_index"],
            "canonical_smiles": e.get("canonical_smiles"), "observed_score": score,
            "incumbent_before": before, "incumbent_after": incumbent,
            "top_10": states.get(e["bo_step"], {}).get("quality", {}).get("top_10"),
            "realized_improvement": improvement, "strict_record_improvement": improvement > 0 if improvement is not None else None,
            "predicted_mean": mu, "latent_std": sigma, "predictive_std": predictive_std,
            "recorded_analytic_ei": recorded_ei, "recorded_ei_incumbent": recorded_incumbent,
            "ei_incumbent_matches_observed_prefix": threshold_matches,
            "analytic_expected_improvement": ei, "recorded_log_ei": log_ei,
            "exp_recorded_log_ei_smoothed_mc_proxy": proxy,
            "positive_mean_advantage": mean_advantage, "uncertainty_uplift": uplift,
            "observed_minus_predicted_mean": residual,
            "within_one_latent_std": abs(residual) <= sigma if residual is not None and sigma is not None else None,
            "within_90pct_predictive_interval": abs(residual) <= 1.6448536269514722 * predictive_std
            if residual is not None and predictive_std is not None else None,
            "missing_reason": reason,
        })
    expected_sum = _cumulative_available([r["analytic_expected_improvement"] for r in rows])
    realized_sum = _cumulative_available([r["realized_improvement"] for r in rows])
    for row, expected, realized in zip(rows, expected_sum, realized_sum):
        row["cumulative_analytic_ei"] = expected
        row["cumulative_realized_improvement"] = realized
    hits = [r for r in rows if r["strict_record_improvement"]]
    residuals = [r["observed_minus_predicted_mean"] for r in rows if r["observed_minus_predicted_mean"] is not None]
    latent_coverage = [r["within_one_latent_std"] for r in rows if r["within_one_latent_std"] is not None]
    predictive_coverage = [r["within_90pct_predictive_interval"] for r in rows if r["within_90pct_predictive_interval"] is not None]
    first_final = next((r["bo_step"] for r in rows if r["incumbent_after"] == incumbent), 0) if incumbent > initial_best else 0
    last_record_step = hits[-1]["bo_step"] if hits else 0
    after_last = [r for r in rows if r["bo_step"] > last_record_step]
    plateau_expected = _cumulative_available([r["analytic_expected_improvement"] for r in after_last])
    means = [r for r in rows if r["predicted_mean"] is not None]
    def state_value(step, key):
        state = states.get(step, {})
        return state.get("quality", {}).get(key) if key == "top_10" else state.get(key)
    scale = max((abs(r) for r in residuals), default=0.)
    rmse = scale * math.sqrt(mean((r / scale) ** 2 for r in residuals)) if scale else (0. if residuals else None)
    summary = {
        "initial_best": initial_best, "final_best": incumbent,
        "realized_best_score_gain": incumbent - initial_best,
        "n_bo_steps": len(rows), "n_record_improvements": len(hits),
        "record_improvement_steps": [r["bo_step"] for r in hits],
        "first_step_of_final_best": first_final,
        "last_record_improvement_step": last_record_step,
        "steps_without_a_new_best_at_end": len(rows) - last_record_step,
        "n_comparable_analytic_ei": sum(r["analytic_expected_improvement"] is not None for r in rows),
        "sum_analytic_ei": expected_sum[-1], "sum_realized_improvements": realized_sum[-1],
        "sum_analytic_ei_after_last_record": plateau_expected[-1] if plateau_expected else 0.,
        "top_10_at_last_record": state_value(last_record_step, "top_10"),
        "final_top_10": state_value(rows[-1]["bo_step"], "top_10"),
        "top_10_internal_diversity_at_last_record": state_value(last_record_step, "top_10_internal_diversity"),
        "final_top_10_internal_diversity": state_value(rows[-1]["bo_step"], "top_10_internal_diversity"),
        "n_predicted_means": len(means),
        "n_means_below_incumbent": sum(r["predicted_mean"] < r["incumbent_before"] for r in means),
        "mean_observed_minus_predicted": mean(residuals) if residuals else None,
        "prequential_rmse": rmse,
        "coverage_one_latent_std": mean(int(x) for x in latent_coverage) if latent_coverage else None,
        "coverage_90pct_predictive_interval": mean(int(x) for x in predictive_coverage) if predictive_coverage else None,
    }
    if realized_sum[-1] is not None and not math.isclose(
        realized_sum[-1], incumbent - initial_best, rel_tol=1e-10, abs_tol=1e-12
    ):
        raise ValueError("Realized improvements do not telescope to the best-score gain")
    return {"summary": summary, "rows": rows, "definitions": {
        "realized_improvement": "max(0, observed score minus best eligible score before evaluation); initialization included in incumbent",
        "analytic_expected_improvement": "Recorded Gaussian latent-posterior EI; only compared when recorded acquisition incumbent matches the observed prefix",
        "exp_recorded_log_ei_smoothed_mc_proxy": "Exponentiated recorded qLogExpectedImprovement diagnostic; smoothed Monte Carlo proxy, not exact analytic EI",
        "uncertainty_uplift": "EI - max(predicted mean minus incumbent, 0); Jensen gap, not the two terms of the usual EI formula",
        "cumulative_analytic_ei": "Sum of one-step conditional forecasts along the realized adaptive path; not a promised or feasible final-score target",
        "missing_values": "Retained as null/gaps; a missing increment makes subsequent cumulative totals unknown",
        "replication": "One adaptively sampled run, not independent molecules or seeds; no cross-seed uncertainty claims",
    }}


def _plot(analysis, run, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    rows, summary = analysis["rows"], analysis["summary"]
    def array(key):
        return np.asarray([r[key] if r[key] is not None else np.nan for r in rows], dtype=float)
    x, ei, gains = array("bo_step"), array("analytic_expected_improvement"), array("realized_improvement")
    positive = np.isfinite(gains) & (gains > 0)
    blue, orange = "#1f77b4", "#df7e25"
    with plt.rc_context({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}):
        fig, axes = plt.subplots(2, 2, figsize=(14, 9.3), sharex=True)
        a, b, c, d = axes.ravel()
        a.plot(x, ei, color=blue, linewidth=1.3, label="Analytic Gaussian EI")
        proxy = array("exp_recorded_log_ei_smoothed_mc_proxy")
        if np.isfinite(proxy).any():
            a.plot(x, proxy, color="#606060", linestyle="--", linewidth=.9, alpha=.8,
                   label="exp(recorded log-EI): smoothed MC proxy")
        a.vlines(x[positive], 0, gains[positive], color=orange, linewidth=1.4)
        a.scatter(x, gains, color=orange, s=9, zorder=4,
                  label=f"Realized gain ({summary['n_record_improvements']} record improvements)")
        a.set_yscale("symlog", linthresh=1e-4, linscale=.5)
        a.set_ylim(bottom=0)
        a.yaxis.set_major_formatter(FuncFormatter(lambda value, pos: f"{value:g}"))
        a.set(title="A. Did expected gains become real gains?", ylabel="Score improvement (symlog; zero retained)")
        a.legend(fontsize=7.5, loc="upper right")
        b.plot(np.r_[0, x], np.r_[0, array("cumulative_analytic_ei")], color=blue, linewidth=1.8,
               label="Sum of one-step EI forecasts (not a target)")
        b.step(np.r_[0, x], np.r_[0, array("cumulative_realized_improvement")], where="post",
               color=orange, linewidth=2, label="Cumulative realized record gain")
        b.set(title="B. Forecast gains vs achieved gains", ylabel="Cumulative score improvement")
        b.legend(fontsize=8, loc="upper left")
        if summary["sum_analytic_ei"] is not None:
            b.annotate(f"{summary['sum_analytic_ei']:.3f}", (x[-1], summary["sum_analytic_ei"]),
                       xytext=(-4, 6), textcoords="offset points", ha="right", color=blue)
        if summary["sum_realized_improvements"] is not None:
            b.annotate(f"{summary['sum_realized_improvements']:.3f}", (x[-1], summary["sum_realized_improvements"]),
                       xytext=(-4, 6), textcoords="offset points", ha="right", color=orange)
        c.step(np.r_[0, x], np.r_[summary["initial_best"], array("incumbent_after")], where="post",
               color=orange, linewidth=2, label="Best observed score")
        c.scatter(x[positive], array("incumbent_after")[positive], color=orange, s=20, zorder=4)
        if np.isfinite(array("top_10")).any():
            initial_top_10 = next((p["quality"]["top_10"] for p in run.get("trajectory", []) if p["bo_step"] == 0), np.nan)
            c.plot(np.r_[0, x], np.r_[initial_top_10, array("top_10")], color="#32866a", linestyle="--",
                   linewidth=1.5, label="Mean of Top-10 unique molecules")
        c.axhline(summary["initial_best"], color="#777777", linestyle=":", linewidth=1, label="Best initial molecule")
        c.set(title="C. Record and Top-10 progress", ylabel="Best score / Top-10 mean")
        c.legend(fontsize=8, loc="lower right")
        c.text(.04, .15, f"Final best {summary['final_best']:.4f}, first reached at step {summary['first_step_of_final_best']}\n"
               f"No new best in the final {summary['steps_without_a_new_best_at_end']} steps", transform=c.transAxes, fontsize=9)
        advantage, uplift = array("positive_mean_advantage"), array("uncertainty_uplift")
        if np.isfinite(advantage + uplift).any():
            d.stackplot(x, advantage, uplift, colors=("#5ba27e", "#78abd0"), alpha=.85,
                        labels=("Positive mean advantage: max(mean - incumbent, 0)", "Uncertainty uplift: EI minus positive mean advantage"))
        d.set(title="D. How does uncertainty contribute to EI?", ylabel="Analytic EI decomposition")
        d.text(.03, .93, f"Mean below incumbent: {summary['n_means_below_incumbent']}/{summary['n_predicted_means']} predictions",
               transform=d.transAxes, fontsize=9, va="top")
        if d.get_legend_handles_labels()[0]:
            d.legend(fontsize=7.5, loc="upper left", bbox_to_anchor=(0, .88))
        last = summary["last_record_improvement_step"]
        for axis in axes.ravel():
            if last < x[-1]:
                axis.axvline(last, color="#888888", linestyle=":", linewidth=1)
            axis.set_xlim(0, x[-1])
            axis.grid(axis="y", alpha=.15)
            axis.set_xlabel("BO step (0 = after initialization)")
            axis.tick_params(labelbottom=True)
        fig.suptitle(f"COWBOYS | {run['task']['task_name']} | seed {run['design']['seed']} | improvement diagnostics", fontsize=14)
        fig.text(.06, .018, "Pre-evaluation forecasts only. Dotted vertical line: last record improvement. "
                 "Analytic EI and recorded smoothed MC log-EI are distinct diagnostics.\n"
                 "One adaptive run: cumulative forecast sums and EI decomposition are descriptive, not proof of calibration or optimal selection.",
                 fontsize=8.5, color="#555555")
        fig.tight_layout(rect=(0, .07, 1, .96))
        for extension in ("png", "pdf"):
            fig.savefig(io_path(output / f"improvement_analysis.{extension}"), dpi=170, facecolor="white")
        plt.close(fig)


def _write_interpretation(path, analysis, run):
    s = analysis["summary"]
    def value(key, decimals=3):
        return f"{s[key]:.{decimals}f}" if s[key] is not None else "unavailable"
    text = f"""# Improvement diagnostics: {run['task']['task_name']}, seed {run['design']['seed']}

The mean/standard-deviation plot checks the surrogate's score forecasts. It does
not show whether the candidate beats the incumbent. COWBOYS samples PI-weighted
proposals and selects using Monte Carlo log expected improvement; the scientific
objective remains the recorded oracle score.

For this run, the GP mean is below the incumbent on {s['n_means_below_incumbent']}
of {s['n_predicted_means']} recorded predictions. The fraction of observed scores
inside one latent standard deviation is {value('coverage_one_latent_std')}; central
90% predictive-interval coverage is {value('coverage_90pct_predictive_interval')}.
These are descriptive, selected-point diagnostics. Aggregate coverage does not
establish that the upper tail relevant to EI is correctly calibrated.

## Reading the new figure

- A compares recorded analytic Gaussian EI with actual gain over the **previous**
  incumbent. The dashed curve is exp(recorded qLogEI), a smoothed Monte Carlo
  proxy, not exact analytic EI. Zero gains remain visible on a symlog axis.
- B compares the sum of conditional one-step EI forecasts ({value('sum_analytic_ei')})
  with achieved record gains ({value('sum_realized_improvements')}). The forecast
  sum is not a promised or feasible final-score target.
- C shows improvement from initial best {value('initial_best', 6)} to final best
  {value('final_best', 6)}. There are {s['n_record_improvements']} strict record
  improvements at steps {s['record_improvement_steps']}; the final best first
  appears at step {s['first_step_of_final_best']}.
- D splits analytic EI into positive mean advantage and the remaining
  uncertainty uplift. This is EI minus max(mean minus incumbent, 0), not an
  explicit exploration/exploitation split inside the solver.

The final {s['steps_without_a_new_best_at_end']} evaluations did not improve the
record, while their conditional EI forecasts sum to
{value('sum_analytic_ei_after_last_record')}. This locates a useful question for
further checks: are optimistic upper-tail forecasts persisting after progress
plateaus? It is not proof of miscalibration or of a poor optimization policy.
Early large wins can also be much larger than their individual expected gains;
an expectation need not match any single outcome.

A record plateau does not imply no further useful discoveries. The Top-10 mean
at the last record is {value('top_10_at_last_record', 6)}, versus
{value('final_top_10', 6)} at the end. Top-10 internal diversity is
{value('top_10_internal_diversity_at_last_record', 6)} at the last record and
{value('final_top_10_internal_diversity', 6)} at the end (the existing Morgan
radius-2 bit-fingerprint metric). Panel C preserves the recorded Top-10 definition.

To assess the optimization policy, compare best-score/AUC and discovery curves
across independent seeds and matched baselines. To examine EI calibration,
inspect gain forecasts by optimization phase and uncertainty/PI reliability
across those runs. This one selected trajectory cannot establish that unselected
candidates would have been better, since their oracle outcomes are unavailable.

No solver or oracle was rerun. The prior mean/std plot and original artifacts
were preserved. Full definitions and source hashes are in improvement_analysis.json;
all plotted points are in improvement_analysis.csv.
"""
    io_path(path).write_text(text, encoding="utf-8")


def improvement_report(status_file, output_dir=None):
    from .export_runs import build_run, validate_run
    status_file = Path(status_file)
    run = build_run(status_file)
    validate_run(run)
    analysis = analyze_improvements(run)
    output = Path(output_dir) if output_dir else status_file.parent / "metrics" / status_file.name.removesuffix("_status.json") / "figures"
    io_path(output).mkdir(parents=True, exist_ok=True)
    _plot(analysis, run, output)
    with io_path(output / "improvement_analysis.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(analysis["rows"][0]))
        writer.writeheader()
        writer.writerows(analysis["rows"])
    _write_interpretation(output / "improvement_analysis.md", analysis, run)
    unchanged = all(digest_file(Path(s["path"])) == s["sha256"] for s in run["provenance"]["sources"])
    if not unchanged:
        raise ValueError("A source checkpoint changed during analysis; rerun on a stable saved run")
    report = {"schema_version": "1.0.0", "run_id": run["run_id"], "task": run["task"]["task_name"],
              "seed": run["design"]["seed"], "source_status": run["source_status"],
              "summary": analysis["summary"], "definitions": analysis["definitions"],
              "original_inputs_unchanged": unchanged, "oracle_or_solver_rerun": False,
              "sources": run["provenance"]["sources"],
              "files": {kind: display_path(output / f"improvement_analysis.{kind}") for kind in ("png", "pdf", "csv", "md")}}
    write_json(output / "improvement_analysis.json", report)
    return report
