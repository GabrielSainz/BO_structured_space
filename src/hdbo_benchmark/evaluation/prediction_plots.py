"""Additional GP diagnostic figures; existing molecular plots stay unchanged."""

import numpy as np


def plot_predictions(groups, output):
    import matplotlib.pyplot as plt
    from .aggregate_runs import _finish_plot
    from .metric_io import io_path

    manifest = []
    figure_dir = io_path(output / "figures")
    for group_id, records in groups.items():
        task = records[0]["run"]["task"]["task_name"]
        fig, axis = plt.subplots(figsize=(7, 4.5))
        all_scores = []
        for record in records:
            paired = [e for e in record["run"]["evaluations"]
                      if (e.get("pre_evaluation_prediction") or {}).get("mean") is not None
                      and e["raw_oracle_score"] is not None]
            if paired:
                means = [e["pre_evaluation_prediction"]["mean"] for e in paired]
                observed = [e["raw_oracle_score"] for e in paired]
                all_scores.extend(means + observed)
                axis.scatter(means, observed, s=12, alpha=.5,
                             label=f"{record['variant']} seed {record['run']['design']['seed']}")
        if all_scores:
            low, high = min(all_scores), max(all_scores)
            axis.plot([low, high], [low, high], color="black", linestyle="--", linewidth=1)
        _finish_plot(fig, axis, manifest, figure_dir, group_id, "gp_predicted_vs_observed", task,
                     bool(all_scores), "Pre-evaluation GP mean (recorded objective scale)", "Observed oracle score")
        # Separate run panels/figures prevent pooling different sequential GPs.
        for record in records:
            run = record["run"]
            observations = [e for e in run["evaluations"] if e["phase"] == "optimization" and e["bo_step"] is not None]
            fig, axis = plt.subplots(figsize=(8, 4.5))
            available = False
            if observations:
                x = [e["bo_step"] for e in observations]
                predictions = [e.get("pre_evaluation_prediction") or {} for e in observations]
                mu = np.asarray([p.get("mean") if p.get("mean") is not None else np.nan for p in predictions])
                sigma = np.asarray([p.get("predictive_std") if p.get("predictive_std") is not None else np.nan for p in predictions])
                observed = [e["raw_oracle_score"] if e["raw_oracle_score"] is not None else np.nan for e in observations]
                available = bool(np.isfinite(mu).any())
                if available:
                    axis.plot(x, mu, label="Pre-evaluation GP mean", color="C0", marker=".", markersize=3)
                    axis.fill_between(x, mu - 1.6448536269514722 * sigma, mu + 1.6448536269514722 * sigma,
                                      alpha=.2, color="C0", label="90% predictive interval (includes noise)")
                    if len(x) == 1:
                        axis.errorbar(x, mu, yerr=1.6448536269514722 * sigma,
                                      fmt="none", color="C0", capsize=4)
                    axis.scatter(x, observed, label="Observed score", color="C1", s=10)
            _finish_plot(fig, axis, manifest, figure_dir, group_id,
                         f"gp_prediction_trace_{run['run_id']}", task, available,
                         "Completed BO step", "Recorded objective score")
    return manifest
