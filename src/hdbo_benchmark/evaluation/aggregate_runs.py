"""Offline run-level aggregation and plots, extending the local evaluation package."""

import csv
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, stdev

import numpy as np

from .export_runs import validate_run
from .metric_io import digest_json, display_path, io_path, read_json, write_json
from .molecular_metrics import HITS, QUALITY


CURVES = (*QUALITY, *HITS, "top_10_internal_diversity")
SCALARS = (*[f"final_{m}" for m in CURVES], *[f"auc_{m}" for m in (*QUALITY, *HITS)])


def curve_value(point, metric):
    if metric in QUALITY:
        return point["quality"][metric]
    if metric in HITS:
        return point["diversity_by_profile"]["primary"][metric]
    return point.get(metric)


def scalar_value(run, metric):
    prefix, key = metric.split("_", 1)
    if prefix == "final" and key in (*QUALITY, "top_10_internal_diversity"):
        return run["summary"]["final"].get(key)
    if prefix == "auc" and key in QUALITY:
        return run["summary"]["auc"].get(key)
    return run["summary"]["diversity_by_profile"]["primary"].get(metric)


def compatibility(run):
    """Algorithm/config varies within an overlay; all metric/protocol scales match."""
    task, design = run["task"], run["design"]
    return {"task": task, "metric_policy": run["metric_policy"],
            "planned_bo_steps": design["bo_steps_planned"],
            "n_initial_observed": design["n_initial_observed"],
            "n_initial_planned": design["n_initial_planned"],
            "initialization_policy": design["initialization_policy"],
            "batch_size": design["batch_size"],
            "diagnostic_objective_evaluations_enabled": run["summary"].get("diagnostic_objective_evaluations_enabled")}


def describe(values, *, bootstrap_replicates=2000):
    """Sample SD and deterministic percentile bootstrap CI across independent runs."""
    values = [float(v) for v in values if v is not None]
    if not values:
        return {"n": 0, "mean": None, "sample_sd": None, "ci95_low": None, "ci95_high": None}
    lo = hi = None
    if len(values) >= 2:
        rng = np.random.default_rng(0)
        array = np.asarray(values)
        means = array[rng.integers(0, len(array), size=(bootstrap_replicates, len(array)))].mean(axis=1)
        lo, hi = np.quantile(means, [.025, .975]).tolist()
    return {"n": len(values), "mean": mean(values),
            "sample_sd": stdev(values) if len(values) >= 2 else None,
            "ci95_low": lo, "ci95_high": hi}


def _write_csv(path, rows, fields):
    path = io_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def aggregate_directories(results_dirs, *, output_dir=None, plots=True):
    roots = [Path(p).resolve() for p in results_dirs]
    output = Path(output_dir).resolve() if output_dir else roots[0] / "metrics/aggregate"
    sources = sorted({path for root in roots for path in io_path(root).rglob("run.json")
                      if "preflight" not in path.relative_to(io_path(root)).parts})
    groups, rows, issues, loaded = defaultdict(list), [], [], []
    seen_ids = set()
    for source in sources:
        try:
            run = read_json(source)
            validate_run(run)
            if run["run_id"] in seen_ids:
                issues.append({"source": display_path(source), "reason": "Duplicate run_id excluded"})
                continue
            seen_ids.add(run["run_id"])
            group_id = digest_json(compatibility(run))[:12]
            variant = f"{run['algorithm']['name']}:{digest_json(run['algorithm']['config'])[:8]}"
            record = {"run": run, "source": source, "group_id": group_id, "variant": variant}
            groups[group_id].append(record)
            loaded.append(record)
        except (ValueError, OSError, KeyError, TypeError) as exc:
            issues.append({"source": display_path(source), "reason": str(exc)})
    if not loaded:
        raise ValueError("No valid run.json files found; export a selected result directory first")
    summaries, trajectory_rows, individual_rows = [], [], []
    for group_id, records in groups.items():
        multiplicities = Counter((r["variant"], r["run"]["design"]["seed"]) for r in records)
        for record in records:
            run = record["run"]
            excluded = []
            if not run["summary"]["full_budget_completed"]:
                excluded.append("partial_or_unverified_budget")
            if run["task"].get("oracle_version") is None:
                excluded.append("oracle_version_not_recorded_or_supplied")
            if run["design"].get("initialization_policy_source") == "not_recorded":
                excluded.append("initialization_sampling_policy_not_recorded_or_supplied")
            if run["design"]["seed"] is None:
                excluded.append("independent_seed_unknown")
            if multiplicities[(record["variant"], run["design"]["seed"])] > 1:
                excluded.append("repeated_seed_for_algorithm_configuration")
            record["eligible"] = not excluded
            base = {"group_id": group_id, "variant": record["variant"], "run_id": run["run_id"],
                    "task": run["task"]["task_name"], "seed": run["design"]["seed"],
                    "status": run["status"], "auc_scope": run["summary"]["auc_scope"],
                    "included_in_full_budget_statistics": not excluded,
                    "exclusion_reason": ";".join(excluded), "source": display_path(record["source"])}
            rows.append({**base, **{m: scalar_value(run, m) for m in SCALARS}})
            for point in run["trajectory"]:
                individual_rows.append({**base, "bo_step": point["bo_step"],
                                        **{m: curve_value(point, m) for m in CURVES}})
        for variant in sorted({r["variant"] for r in records}):
            eligible = [r["run"] for r in records if r["eligible"] and r["variant"] == variant]
            for metric in SCALARS:
                summaries.append({"group_id": group_id, "variant": variant, "metric": metric,
                                  **describe([scalar_value(run, metric) for run in eligible])})
            steps = sorted({p["bo_step"] for run in eligible for p in run["trajectory"]})
            by_step = {run["run_id"]: {p["bo_step"]: p for p in run["trajectory"]} for run in eligible}
            for step in steps:
                for metric in CURVES:
                    values = [curve_value(by_step[run["run_id"]][step], metric) for run in eligible
                              if step in by_step[run["run_id"]]]
                    trajectory_rows.append({"group_id": group_id, "variant": variant, "bo_step": step,
                                            "metric": metric, **describe(values)})
    manifest = _plot(groups, trajectory_rows, output) if plots else [
        {"status": "skipped", "missing_reason": "Plot generation disabled by --no-plots"}]
    payload = {
        "schema_version": "1.0.0", "source_run_jsons": [display_path(r["source"]) for r in loaded],
        "aggregation_policy": {"unit_of_replication": "independent_seed_per_algorithm_configuration",
            "included_statuses": ["completed_full_budget_with_known_oracle_version"],
            "curve_alignment": "exact_saved_BO_steps; no_padding_or_interpolation",
            "uncertainty": "sample_SD_and_percentile_bootstrap_95pct_CI_across_runs",
            "bootstrap_replicates": 2000, "bootstrap_seed": 0,
            "partial_run_handling": "individual_only; no full-budget aggregate",
            "unknown_oracle_version": "individual_only; supply version only with provenance evidence"},
        "compatibility_groups": {key: compatibility(value[0]["run"]) for key, value in groups.items()},
        "run_summary_rows": rows, "metric_summary_rows": summaries,
        "trajectory_summary_rows": trajectory_rows, "plot_manifest": manifest, "issues": issues,
    }
    write_json(output / "experiment_summary.json", payload)
    write_json(output / "plot_manifest.json", manifest)
    base_fields = ["group_id", "variant", "run_id", "task", "seed", "status", "auc_scope",
                   "included_in_full_budget_statistics", "exclusion_reason", "source"]
    _write_csv(output / "per_run.csv", rows, [*base_fields, *SCALARS])
    _write_csv(output / "per_run_trajectories.csv", individual_rows, [*base_fields, "bo_step", *CURVES])
    stats = ["n", "mean", "sample_sd", "ci95_low", "ci95_high"]
    _write_csv(output / "metric_summary.csv", summaries, ["group_id", "variant", "metric", *stats])
    _write_csv(output / "trajectory_summary.csv", trajectory_rows, ["group_id", "variant", "bo_step", "metric", *stats])
    return payload


def _plot(groups, trajectory_rows, output):
    # Match existing plotting dependency/backend; never import optimization code.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    manifest = []
    figure_dir = io_path(output / "figures")
    figure_dir.mkdir(parents=True, exist_ok=True)
    for group_id, records in groups.items():
        task = records[0]["run"]["task"]["task_name"]
        for metric in CURVES:
            fig, axis = plt.subplots(figsize=(7, 4.5))
            any_values = False
            variants = sorted({r["variant"] for r in records})
            for color_index, variant in enumerate(variants):
                color = f"C{color_index % 10}"
                for record in [r for r in records if r["variant"] == variant]:
                    run = record["run"]
                    # Null remains a gap, never bridge unavailable trajectory states.
                    x = [p["bo_step"] for p in run["trajectory"]]
                    y = [curve_value(p, metric) for p in run["trajectory"]]
                    if not any(v is not None for v in y):
                        continue
                    any_values = True
                    scope = "full" if record["eligible"] else "individual; excluded from mean"
                    axis.plot(x, [v if v is not None else np.nan for v in y], color=color,
                              alpha=.45, marker=".", linestyle="-" if record["eligible"] else "--",
                              label=f"{variant} seed {run['design']['seed']} ({run['run_id'][:5]}, {scope})")
                stats = [r for r in trajectory_rows if r["group_id"] == group_id
                         and r["variant"] == variant and r["metric"] == metric]
                if any(r["n"] for r in stats):
                    counts = [r["n"] for r in stats]
                    n_label = str(counts[0]) if min(counts) == max(counts) else f"{min(counts)}..{max(counts)}"
                    axis.plot([r["bo_step"] for r in stats],
                              [r["mean"] if r["mean"] is not None else np.nan for r in stats], color=color,
                              linewidth=2, label=f"{variant} mean (n={n_label})")
                    if any(r["ci95_low"] is not None for r in stats):
                        axis.fill_between([r["bo_step"] for r in stats],
                                          [r["ci95_low"] if r["ci95_low"] is not None else np.nan for r in stats],
                                          [r["ci95_high"] if r["ci95_high"] is not None else np.nan for r in stats],
                                          alpha=.2, color=color)
            _finish_plot(fig, axis, manifest, figure_dir, group_id, metric, task, any_values,
                         "Completed BO step (0 = after initialization)", metric)
        for metric in SCALARS:
            fig, axis = plt.subplots(figsize=(7, 4.5))
            points = [(r, scalar_value(r["run"], metric)) for r in records]
            points = [(r, v) for r, v in points if v is not None]
            for i, (record, value) in enumerate(points):
                axis.scatter(i, value, marker="o" if record["eligible"] else "x")
            axis.set_xticks(range(len(points)), [f"{r['variant']}\ns{r['run']['design']['seed']} {r['run']['run_id'][:5]}"
                                                for r, _ in points], rotation=30, ha="right", fontsize=7)
            _finish_plot(fig, axis, manifest, figure_dir, group_id, metric, task, bool(points),
                         "Independent run (x = excluded from full-budget statistics)", metric)
        fig, axis = plt.subplots(figsize=(7, 4.5))
        any_values = False
        for record in records:
            x, y = scalar_value(record["run"], "auc_top_1"), scalar_value(record["run"], "auc_online_diverse_hits")
            if x is not None and y is not None:
                any_values = True
                axis.scatter(x, y, label=f"{record['variant']} seed {record['run']['design']['seed']}",
                             marker="o" if record["eligible"] else "x")
        _finish_plot(fig, axis, manifest, figure_dir, group_id, "quality_vs_diversity", task, any_values,
                     "AUC Top-1 (observed horizon; see per-run scope)", "AUC online diverse hits (count)")
    return manifest


def _finish_plot(fig, axis, manifest, figure_dir, group_id, metric, task, available, xlabel, ylabel):
    import matplotlib.pyplot as plt
    entry = {"group_id": group_id, "metric": metric}
    try:
        if available:
            axis.set(title=f"{task}: {metric}", xlabel=xlabel, ylabel=ylabel)
            if axis.get_legend_handles_labels()[0]:
                axis.legend(fontsize=6)
            axis.grid(alpha=.2)
            fig.tight_layout()
            path = figure_dir / f"{group_id}_{metric}.png"
            fig.savefig(path, dpi=150)
            entry.update(status="generated", path=display_path(path))
        else:
            entry.update(status="skipped", missing_reason="No available values; inspect per-run missing_reason and threshold configuration")
        manifest.append(entry)
    finally:
        plt.close(fig)
