"""Read-only, seed-aware COWBOYS experiment analysis. No optimizer/model imports.

Run with ``python -m hdbo_benchmark.evaluation.cowboys_analysis --help``.
The existing artifact reader and molecular identity/fingerprint helpers are reused;
the rolling, AUC, coverage and >=delta packing policies are specific to this report.
"""

import argparse
from collections import Counter, defaultdict
from functools import lru_cache
from itertools import combinations, product
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import ndtr
from rdkit.Chem.Scaffolds import MurckoScaffold

from .cowboys_artifacts import load_cowboys
from .metric_io import clean, digest_file, io_path, read_json, write_json
from .molecular_metrics import FINGERPRINT, molecule_info, fingerprint, distance


WINDOW, MIN_VALID = 25, 10
BINS = [0, .1, .2, .4, 1.00001]
BIN_LABELS = ["[0,0.1)", "[0.1,0.2)", "[0.2,0.4)", "[0.4,1]"]
GENERATOR = ["sample_unique_decoded_molecules_in_iteration",
             "sample_new_distinct_decoded_molecules",
             "sample_unique_decoded_molecules_not_in_observed_history",
             "sample_cumulative_distinct_decoded_molecules",
             "available_unique_top_10_candidate_count"]
TITLES = {"drd2_docking": "DRD2 classifier (poli: drd2_docking)",
          "albuterol_similarity": "Albuterol similarity", "scaffold_hop": "Scaffold hopping",
          "osimetrinib_mpo": "Osimertinib MPO (poli: osimetrinib_mpo)",
          "ranolazine_mpo": "Ranolazine MPO"}


def number(value):
    try:
        return float(value) if value is not None and math.isfinite(float(value)) else np.nan
    except (ValueError, TypeError):
        return np.nan


def close(a, b):
    return np.isfinite(a) and np.isfinite(b) and math.isclose(a, b, rel_tol=1e-8, abs_tol=1e-12)


@lru_cache(maxsize=30000)
def chemistry(structure):
    info = molecule_info(structure)
    mol = info["mol"]
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol) if mol is not None else None
    return info["canonical_smiles"], fingerprint(mol) if mol is not None else None, scaffold or None


def companion(path, status, key, suffix):
    expected = path.with_name(path.name.removesuffix("_status.json") + suffix)
    return str(expected)


def load_run(path, planned_steps):
    """Validate one unambiguous batch-one attempt before deriving any metrics."""
    raw = load_cowboys(path)
    status, doc = raw["status"], raw["iterations"] or {}
    task, seed = status["function_name"], status["seed"]
    warnings = list(raw["warnings"])
    errors = []
    config = status.get("solver_configuration", {})
    records = doc.get("iterations", [])
    initial = status.get("initial_history_size")
    if not isinstance(initial, int) or initial < 0:
        errors.append("Invalid/missing initial_history_size")
    if config.get("batch_size", doc.get("batch_size")) != 1 or doc.get("batch_size", 1) != 1:
        errors.append("Unsupported batch size: only one selected observation per step is supported")
    if doc.get("initial_history_size", initial) != initial:
        errors.append("Iteration/status initial-history counts disagree")
    completed = status.get("completed_iterations")
    if not isinstance(completed, int) or len(records) != completed:
        errors.append("Iteration record count disagrees with completed_iterations")
    if [r.get("iteration") for r in records] != list(range(1, len(records) + 1)):
        errors.append("Iteration numbering is not consecutive 1..T")
    for key in ("completed_iterations", "tracked_iterations"):
        if key in doc and doc[key] != len(records):
            errors.append(f"{key} disagrees with record count")
    for key, values in doc.get("metric_series", {}).items():
        if len(values) != len(records):
            errors.append(f"Metric-series length mismatch: {key}")
        elif key == "bo_iteration" and values != list(range(1, len(records) + 1)):
            errors.append("Metric-series iteration numbering disagrees")
        elif key != "bo_iteration":
            for r, v in zip(records, values):
                if key in r and not (close(number(v), number(r[key])) or
                                     (not np.isfinite(number(v)) and not np.isfinite(number(r[key])))):
                    errors.append(f"Metric-series values disagree: {key}")
                    break
    for r in records:
        if any(len(r.get(k, [])) != 1 for k in
               ("selected_smiles", "selected_valid_mask", "selected_objective_values")):
            errors.append(f"Malformed/unsupported selected batch at iteration {r.get('iteration')}")
        if isinstance(initial, int) and r.get("history_size_after_iteration", initial + r["iteration"]) != initial + r["iteration"]:
            errors.append(f"History count mismatch at iteration {r['iteration']}")
    scores, structures = raw["scores"], raw["structures"]
    if scores is not None and isinstance(initial, int) and len(scores) != initial + len(records):
        errors.append("NPZ history length disagrees with initial + selected counts")
    direction = status.get("score_direction", "maximize")
    conventions = (raw["predictions"] or {}).get("conventions", {})
    if direction != "maximize":
        errors.append("Unsupported recorded score direction; this report requires maximize")
    # The audited runner maximizes get_best_performance; no objective transform is introduced.
    inventory = {"task": task, "seed": seed, "solver_name": status["solver_name"],
                 "status": status.get("status"), "completed_steps": completed,
                 "initial_history_size": initial, "planned_steps": planned_steps,
                 "incomplete": status.get("status") != "completed" or completed != planned_steps,
                 "max_sampling_steps": config.get("max_sampling_steps"),
                 "configuration": json.dumps(config, sort_keys=True),
                 "history_missing": structures is None or scores is None,
                 "chemistry_scope": "initial_and_BO" if structures is not None and scores is not None else "BO_selected_only",
                 "status_file": str(path.resolve()),
                 **{k: companion(path, status, k, suffix) for k, suffix in
                    [("history_file", "_history.npz"), ("iteration_results_file", "_iteration_results.json"),
                     ("predictions_file", "_predictions.json")]}}
    if errors:
        return {"inventory": {**inventory, "supported": False}, "rows": [], "initial": [],
                "warnings": warnings, "errors": errors, "sources": raw["sources"]}
    initial_obs = []
    if structures is not None and scores is not None:
        for s, y in zip(structures[:initial], scores[:initial]):
            key, fp, scaf = chemistry(s)
            initial_obs.append((key, number(y), fp, scaf))
    initial_best = max((number(y) for y in (scores or [])[:initial] if np.isfinite(number(y))), default=np.nan)
    preds = (raw["predictions"] or {}).get("predictions", [])
    indexed = defaultdict(list)
    for p in preds:
        indexed[p.get("bo_step")].append(p)
    rows, matched = [], 0
    prior = initial_best
    for r in records:
        t = r["iteration"]
        smiles = r["selected_smiles"][0]
        canonical, fp, scaf = chemistry(smiles)
        valid = r["selected_valid_mask"][0] is True and canonical is not None
        recorded_y = number(r["selected_objective_values"][0])
        y = recorded_y if valid else np.nan
        reasons = []
        if not valid:
            reasons.append("Invalid selected molecular identity/mask")
        if not np.isfinite(recorded_y):
            reasons.append("Missing/nonfinite objective")
        if structures is not None:
            hk, _, _ = chemistry(structures[initial + t - 1])
            if hk != canonical:
                errors.append(f"NPZ/selected identity mismatch at step {t}")
                y = np.nan
                reasons.append("History identity mismatch")
        if scores is not None and not (close(recorded_y, number(scores[initial + t - 1])) or
                                       not np.isfinite(recorded_y) and not np.isfinite(number(scores[initial + t - 1]))):
            errors.append(f"NPZ/selected objective mismatch at step {t}")
            y = np.nan
            reasons.append("History score mismatch")
        candidates = indexed[t]
        p = candidates[0] if len(candidates) == 1 else None
        if p is not None:
            pk = chemistry(p.get("canonical_smiles"))[0]
            boundary = p.get("captured") == "after_candidate_selection_before_oracle_evaluation"
            if "captured" not in p:
                boundary = conventions.get("timing") == "after selection, before oracle evaluation and history update"
            if (pk != canonical or canonical is None or not boundary or
                    p.get("n_training_observations") != initial + t - 1 or
                    p.get("observation_index", initial + t - 1) != initial + t - 1):
                errors.append(f"Prediction identity/pre-evaluation boundary mismatch at step {t}")
                p = None
        if len(candidates) > 1:
            errors.append(f"Ambiguous duplicate prediction step {t}; excluded")
        if p is None:
            reasons.append("No unique validated pre-evaluation prediction")
        else:
            matched += 1
            if p.get("missing_reason"):
                reasons.append(json.dumps(p["missing_reason"], sort_keys=True))
        p = p or {}
        b = number(p.get("incumbent_for_acquisition"))
        comparable = not np.isfinite(prior) or close(b, prior)
        if np.isfinite(b) and not comparable:
            errors.append(f"Acquisition incumbent disagrees with pre-evaluation history at step {t}")
            reasons.append("Incumbent mismatch: decision metrics excluded")
        mu = number(p.get("mean"))
        latent, pred_std = number(p.get("latent_std")), number(p.get("predictive_std"))
        latent_valid, pred_valid = np.isfinite(latent) and latent > 0, np.isfinite(pred_std) and pred_std > 0
        if not latent_valid:
            reasons.append("Missing/nonpositive latent uncertainty")
        if not pred_valid:
            reasons.append("Missing/nonpositive predictive uncertainty; PI/coverage/NLL missing")
        ei = number(p.get("analytic_expected_improvement"))
        if ei < 0:
            ei = np.nan
            reasons.append("Negative analytic EI")
        incumbent = b if comparable else np.nan
        decision = np.isfinite(y) and np.isfinite(incumbent)
        gain = max(y - incumbent, 0) if decision else np.nan
        success = float(y > incumbent) if decision else np.nan
        error = y - mu if np.isfinite(y) and np.isfinite(mu) else np.nan
        z = error / pred_std if pred_valid and np.isfinite(error) else np.nan
        pi = float(ndtr((mu - incumbent) / pred_std)) if pred_valid and np.isfinite(mu) and np.isfinite(incumbent) else np.nan
        current = number(r.get("best_so_far_objective"))
        # Compare to recorded outcomes, including initial scores if available.
        expected = max(v for v in (prior, recorded_y) if np.isfinite(v)) if any(np.isfinite(v) for v in (prior, recorded_y)) else np.nan
        if np.isfinite(current) and np.isfinite(expected) and (current + 1e-12 < expected or
                np.isfinite(prior) and not close(current, expected)):
            errors.append(f"Logged best-so-far inconsistent with history at step {t}")
        d = number(r.get("mean_selected_nearest_previous_tanimoto_distance"))
        if np.isfinite(d) and not 0 <= d <= 1:
            errors.append(f"Invalid logged NN distance at step {t}")
            d = np.nan
        top = r.get("top_candidate_diagnostics")
        count = number(r.get("available_unique_top_10_candidate_count"))
        if count < 0 or count > 10:
            count = np.nan
            reasons.append("Invalid top-10 count")
        if not top:
            reasons.append("Empty/missing saved candidate list")
            count = np.nan
        row = {"task": task, "seed": seed, "iteration": t, "canonical_smiles": canonical,
               "valid_smiles": valid, "raw_y": recorded_y, "y": y, "b": b,
               "historical_incumbent_before": prior, "incumbent_comparable": comparable,
               "incumbent_verified": np.isfinite(prior) and close(b, prior),
               "current_best": current, "initial_best": initial_best,
               "mu": mu, "latent_std": latent if latent_valid else np.nan,
               "predictive_std": pred_std if pred_valid else np.nan,
               "raw_latent_std": latent, "raw_predictive_std": pred_std,
               "analytic_expected_improvement": ei,
               "comparable_ei": ei if comparable and np.isfinite(b) else np.nan,
               "n_training_observations": number(p.get("n_training_observations")),
               "prediction_matched": bool(p), "nn_distance": d,
               "gain": gain, "success": success, "error": error, "squared_error": error ** 2,
               "abs_error": abs(error), "z": z,
               "nll": math.log(pred_std) + .5 * math.log(2 * math.pi) + .5 * z ** 2 if np.isfinite(z) else np.nan,
               "pi": pi, "brier": (pi - success) ** 2 if decision and np.isfinite(pi) else np.nan,
               "near_duplicate": float(d < .05) if np.isfinite(d) else np.nan,
               "exact_fp_duplicate": float(d == 0) if np.isfinite(d) else np.nan,
               "max_sampling_steps": config.get("max_sampling_steps"),
               "chemistry_scope": inventory["chemistry_scope"],
               "missing_objective": not np.isfinite(y), "missing_prediction": not np.isfinite(mu),
               "missing_uncertainty": not pred_valid, "missing_nn": not np.isfinite(d),
               "candidate_diagnostics_count": len(top) if top is not None else np.nan,
               "missing_reason": "; ".join(reasons)}
        for level, cutoff in [(50, .6744897501960817), (80, 1.2815515655446004), (95, 1.96)]:
            row[f"coverage_{level}"] = float(abs(z) <= cutoff) if np.isfinite(z) else np.nan
        row.update({k: number(r.get(k)) for k in GENERATOR})
        row["available_unique_top_10_candidate_count"] = count
        rows.append(row)
        prior = current
    if records and not close(number(status.get("best_value")), rows[-1]["current_best"]):
        errors.append("Status best_value disagrees with final logged best")
    unmatched = [p for p in preds if p.get("bo_step") not in range(1, len(records) + 1)]
    if unmatched:
        warnings.append(f"{len(unmatched)} prediction attempts have no completed evaluation")
    inventory.update(supported=True, valid_outcome_count=sum(np.isfinite(r["y"]) for r in rows),
                     available_gp_predictions=matched, missing_gp_predictions=len(rows) - matched,
                     finite_gp_means=sum(np.isfinite(r["mu"]) for r in rows),
                     available_nn_distances=sum(np.isfinite(r["nn_distance"]) for r in rows),
                     missing_diagnostics=sum(not np.isfinite(r["available_unique_top_10_candidate_count"]) for r in rows),
                     prediction_records=len(preds), unmatched_prediction_attempts=len(unmatched),
                     validation_errors=len(errors))
    return {"inventory": inventory, "rows": rows, "initial": initial_obs, "warnings": warnings,
            "errors": errors, "sources": raw["sources"]}


def thresholds_for(runs):
    selected, evaluated = defaultdict(list), defaultdict(list)
    for run in runs:
        task = run["inventory"]["task"]
        selected[task].extend(r["y"] for r in run["rows"] if np.isfinite(r["y"]))
        evaluated[task].extend(y for key, y, _, _ in run["initial"] if key and np.isfinite(y))
        evaluated[task].extend(r["y"] for r in run["rows"] if np.isfinite(r["y"]))
    return {task: {"tau": float(np.quantile(values, .9)) if values else np.nan,
                   "epsilon": float(.01 * np.ptp(np.quantile(selected[task], [.05, .95]))) if selected[task] else np.nan,
                   "n_evaluated_for_tau": len(values), "n_selected_for_epsilon": len(selected[task]),
                   "source": "retrospective task pooled evaluated 90th percentile; not available during BO",
                   "delta": .4} for task, values in evaluated.items()}


def add_chemistry(run, threshold):
    unique, centers, hit_ids = {}, [], set()
    hit_count = 0
    def observe(key, y, fp, scaffold):
        nonlocal hit_count
        if key is None or fp is None or not np.isfinite(y):
            return
        if key not in unique or y > unique[key][0]:
            unique[key] = (y, fp, scaffold)
        if y >= threshold["tau"]:
            hit_count += 1
            hit_ids.add(key)
            # This report deliberately uses >= .4; the legacy packing helper uses > delta.
            if all(distance(fp, old) >= .4 for old in centers):
                centers.append(fp)
    for obs in run["initial"]:
        observe(*obs)
    for row in run["rows"]:
        key, fp, scaffold = chemistry(row["canonical_smiles"])
        observe(key, row["y"], fp, scaffold)
        top = sorted(unique, key=lambda k: (-unique[k][0], k))[:10]
        row.update(top10_pairwise_distance=np.mean([distance(unique[a][1], unique[b][1])
                   for a, b in combinations(top, 2)]) if len(top) >= 2 else np.nan,
                   top10_k=len(top), top10_scaffolds=len({unique[k][2] for k in top if unique[k][2]}),
                   top10_no_scaffold=sum(unique[k][2] is None for k in top),
                   quality_hits=hit_count, quality_unique_hits=len(hit_ids), circles=len(centers),
                   tau=threshold["tau"], epsilon=threshold["epsilon"])
        ok = np.isfinite(row["gain"]) and np.isfinite(threshold["epsilon"])
        row["epsilon_success"] = float(row["gain"] > threshold["epsilon"]) if ok else np.nan
        row["epsilon_pi"] = float(ndtr((row["mu"] - row["b"] - threshold["epsilon"]) / row["predictive_std"])) if (
            np.isfinite(row["pi"]) and np.isfinite(threshold["epsilon"])) else np.nan


def average(frame, key):
    return frame[key].mean() if len(frame) else np.nan


def rmse(frame):
    return np.sqrt(average(frame, "squared_error"))


def summarize_run(frame, inventory, horizon):
    last = frame[frame.iteration > frame.iteration.max() - 100]
    first = frame[frame.iteration <= 100]
    # A missing decision breaks a known no-improvement streak; it is never a failure.
    longest = streak = 0
    for value in frame.success:
        streak = streak + 1 if value == 0 else 0
        longest = max(streak, longest)
    hits = frame[frame.success == 1]
    common = frame[frame.iteration <= horizon]
    auc = common.current_best.mean() if len(common) == horizon and common.current_best.notna().all() else np.nan
    paired = frame.dropna(subset=["comparable_ei", "gain"])
    total_ei = paired.comparable_ei.sum(min_count=1)
    final = frame.iloc[-1]
    result = {"task": inventory["task"], "seed": inventory["seed"], "row_type": "seed",
              "status": inventory["status"], "incomplete": inventory["incomplete"],
              "chemistry_scope": inventory["chemistry_scope"],
              "max_sampling_steps": inventory["max_sampling_steps"],
              "completed_steps": len(frame), "common_auc_horizon": horizon,
              "final_best": final.current_best, "best_so_far_auc": auc,
              "observed_horizon_auc": frame.current_best.mean() if frame.current_best.notna().all() else np.nan,
              "strict_improvements": int((frame.success == 1).sum()) if frame.success.notna().any() else np.nan,
              "improvement_frequency": average(frame, "success"),
              "epsilon_improvement_frequency": average(frame, "epsilon_success"),
              "last_improving_iteration": hits.iteration.max() if len(hits) else (0 if frame.success.notna().all() else np.nan),
              "longest_no_improvement_streak": longest if frame.success.notna().any() else np.nan,
              "total_realized_gain": frame.gain.sum(min_count=1),
              "mean_nn_distance": average(frame, "nn_distance"), "first100_nn_distance": average(first, "nn_distance"),
              "last100_nn_distance": average(last, "nn_distance"),
              "last100_near_duplicate_fraction": average(last, "near_duplicate"),
              "mean_generator_unique": average(frame, GENERATOR[0]),
              "first100_generator_unique": average(first, GENERATOR[0]),
              "last100_generator_unique": average(last, GENERATOR[0]),
              "last100_generator_new": average(last, GENERATOR[1]),
              "last100_top10_candidates": average(last, GENERATOR[-1]),
              "final_top10_pairwise_distance": final.top10_pairwise_distance,
              "final_top10_scaffolds": final.top10_scaffolds, "final_top10_no_scaffold": final.top10_no_scaffold,
              "quality_hits": final.quality_hits, "quality_unique_hits": final.quality_unique_hits,
              "final_circles": final.circles, "tau": final.tau, "epsilon": final.epsilon,
              "gp_rmse": rmse(frame), "gp_mae": average(frame, "abs_error"), "gp_bias": average(frame, "error"),
              "predictive_sharpness": average(frame, "predictive_std"), "mean_z": average(frame, "z"),
              "gaussian_nll": average(frame, "nll"),
              **{f"coverage_{k}": average(frame, f"coverage_{k}") for k in (50, 80, 95)},
              "first100_gp_rmse": rmse(first), "last100_gp_rmse": rmse(last),
              "first100_coverage_95": average(first, "coverage_95"), "last100_coverage_95": average(last, "coverage_95"),
              "mean_analytic_ei": average(frame, "comparable_ei"), "mean_realized_gain": average(frame, "gain"),
              "paired_realized_to_ei_ratio": paired.gain.sum() / total_ei if total_ei > 0 else np.nan,
              "mean_pi": average(frame, "pi"), "pi_brier": average(frame, "brier"),
              "epsilon_mean_pi": average(frame, "epsilon_pi"),
              "first100_success": average(first, "success"), "last100_success": average(last, "success"),
              "last100_pi": average(last, "pi"), "last100_ei": average(last, "comparable_ei"),
              "last100_gain": average(last, "gain"),
              "first100_circles_added": first.circles.iloc[-1] - (run_initial_circles(frame)),
              "last100_circles_added": final.circles - (frame.loc[frame.iteration <= frame.iteration.max() - 100, "circles"].iloc[-1]
                                                        if len(frame) > 100 else run_initial_circles(frame)),
              "n_paired_ei_gain": len(paired)}
    for key in ["y", "b", "current_best", "mu", "predictive_std", "latent_std", "error", "z", "nll",
                "coverage_95", "nn_distance", "success", "gain", "pi", "brier", "comparable_ei",
                *GENERATOR]:
        result[f"n_{key}"] = int(frame[key].notna().sum())
        result[f"missing_{key}"] = int(frame[key].isna().sum())
    for prefix, subset in [("first100", first), ("last100", last)]:
        for key in ["error", "coverage_95", "nn_distance", "success", *GENERATOR[:2]]:
            result[f"{prefix}_n_{key}"] = int(subset[key].notna().sum())
    return result


def run_initial_circles(frame):
    return frame.initial_circles.iloc[0]


def aggregate_seeds(seed_summary, groups=("task",)):
    numeric = seed_summary.select_dtypes(include=[np.number]).columns.difference(["seed", *groups])
    rows = []
    grouper = groups[0] if len(groups) == 1 else list(groups)
    for keys, g in seed_summary.groupby(grouper, dropna=False, sort=True):
        keys = (keys,) if len(groups) == 1 else keys
        row = {**dict(zip(groups, keys)), "row_type": "task", "n_seeds": len(g),
               "n_complete": int((~g.incomplete).sum())}
        for key in numeric:
            row.update({key + "_mean": g[key].mean(), key + "_sd": g[key].std(ddof=1),
                        key + "_n_seeds": int(g[key].notna().sum())})
        rows.append(row)
    return pd.DataFrame(rows)


def novelty_bins(frame):
    rows = []
    for (task, seed), g in frame.groupby(["task", "seed"], sort=True):
        for i, label in enumerate(BIN_LABELS):
            all_bin = g[(g.nn_distance >= BINS[i]) & (g.nn_distance < BINS[i + 1])]
            b = all_bin.dropna(subset=["error"])
            rows.append({"task": task, "seed": seed, "bin": label, "row_type": "seed",
                         "n_valid": len(b), "n_in_bin": len(all_bin), "contributes": len(b) >= 5,
                         "mean_nn_distance": average(b, "nn_distance"), "rmse": rmse(b),
                         "mae": average(b, "abs_error"), "coverage_95": average(b, "coverage_95"),
                         "predictive_std": average(b, "predictive_std"), "bias": average(b, "error"),
                         "improvement_frequency": average(b, "success"),
                         "n_coverage": int(b.coverage_95.notna().sum()), "n_success": int(b.success.notna().sum())})
    seeds = pd.DataFrame(rows)
    for (task, label), g in seeds.groupby(["task", "bin"], sort=False):
        eligible = g[g.contributes]
        row = {"task": task, "seed": np.nan, "bin": label, "row_type": "task",
               "n_valid": int(g.n_valid.sum()), "n_in_bin": int(g.n_in_bin.sum()),
               "n_contributing_records": int(eligible.n_valid.sum()), "n_seeds": len(eligible)}
        for key in ["mean_nn_distance", "rmse", "mae", "coverage_95", "predictive_std", "bias", "improvement_frequency"]:
            row.update({key + "_mean": eligible[key].mean(), key + "_sd": eligible[key].std(ddof=1),
                        key + "_n_seeds": int(eligible[key].notna().sum())})
        rows.append(row)
    return pd.DataFrame(rows)


def rolling_frame(frame):
    frames = []
    keys = ["success", "nn_distance", *GENERATOR, "squared_error", "coverage_95", "comparable_ei",
            "gain", "pi", "y", "near_duplicate", "exact_fp_duplicate", "abs_error", "predictive_std"]
    for _, group in frame.groupby(["task", "seed"], sort=True):
        g = group.copy().sort_values("iteration")
        for key in keys:
            window = g[key].rolling(WINDOW, min_periods=MIN_VALID)
            g[key + "_rolling_n"] = g[key].rolling(WINDOW, min_periods=1).count()
            g[key + "_rolling"] = window.mean()
        g["rmse_rolling"] = np.sqrt(g.squared_error_rolling)
        frames.append(g)
    return pd.concat(frames, ignore_index=True)


def curve_statistics(frame, key):
    """Pointwise independent-seed mean, sample SD and observed seed count."""
    return frame.groupby("iteration")[key].agg(mean="mean", sd=lambda v: v.std(ddof=1), n="count")


def write_csv(frame, path):
    io_path(path).parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(io_path(path), index=False, float_format="%.12g")


def markdown_table(frame):
    # Avoid an optional tabulate dependency.
    def cell(value):
        return str(value).replace("|", "\\|").replace("\n", " ")
    return "| " + " | ".join(map(cell, frame.columns)) + " |\n| " + " | ".join(["---"] * len(frame.columns)) + " |\n" + "\n".join(
        "| " + " | ".join(map(cell, row)) + " |" for row in frame.itertuples(index=False, name=None))


def save_table(frame, output, name):
    write_csv(frame, output / "tables" / (name + ".csv"))
    io_path(output / "tables" / (name + ".html")).write_text(
        '<meta charset="utf-8"><style>table{border-collapse:collapse;font:13px sans-serif}td,th{padding:6px;border:1px solid #ccc}</style>'
        + frame.to_html(index=False, na_rep="missing", float_format=lambda v: f"{v:.5g}"), encoding="utf-8")


CORE = [
    ("01_best_so_far", "Optimization: best observed objective", [("current_best", "Best so far")], "Objective score (higher is better)", None),
    ("02_improvement_rate", "Optimization: strict record improvement rate", [("success_rolling", "Observed strict improvement")], "Fraction of selected evaluations", (0, 1)),
    ("03_nearest_previous_tanimoto_distance", "Search geography: selected-molecule novelty", [("nn_distance_rolling", "Logged NN distance")], "Logged count-fingerprint Tanimoto distance", (0, 1)),
    ("04_generator_variety", "Search geography: candidate generator variety", [(GENERATOR[0] + "_rolling", "Unique within iteration"), (GENERATOR[1] + "_rolling", "Never decoded in prior iterations")], "Generated unique structures (count)", None),
    ("05_top10_candidate_pool_size", "Search geography: available ranked shortlist", [(GENERATOR[-1] + "_rolling", "Eligible saved top-10 candidates")], "Truncated shortlist size (count)", (0, 10)),
    ("06_top10_high_quality_diversity", "Search geography: diversity of top-10 evaluated molecules", [("top10_pairwise_distance", "Top-10 mean pairwise distance")], "ECFP4 bit Tanimoto distance (1 - similarity)", (0, 1)),
    ("07_high_quality_circles", "Search geography: chronological greedy dissimilar high-quality hits", [("circles", "#Circles proxy (delta = 0.4)")], "Accepted high-quality centers (cumulative count)", None),
    ("09_gp_rolling_rmse", "GP: online one-step-ahead prediction error", [("rmse_rolling", "RMSE on chosen molecules")], "RMSE (objective units)", None),
    ("10_gp_95pct_coverage", "GP: predictive 95% interval coverage", [("coverage_95_rolling", "Empirical predictive coverage")], "Covered fraction (predictive std includes noise)", (0, 1)),
    ("12_expected_vs_realized_improvement", "Decision quality: expected versus realized improvement", [("comparable_ei_rolling", "Analytic latent-posterior EI diagnostic"), ("gain_rolling", "Realized positive record gain")], "Improvement (objective units)", None),
    ("13_predicted_vs_observed_improvement_probability", "Decision quality: predicted versus observed strict success", [("pi_rolling", "Observed-outcome PI diagnostic"), ("success_rolling", "Observed strict improvement")], "Probability / fraction (predictive observation model)", (0, 1)),
]


def make_plots(frame, inventory, bins, summary, output, tasks, planned_steps, include_optional):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.spines.right": False, "axes.spines.top": False,
                         "axes.grid": True, "grid.alpha": .15, "savefig.dpi": 190})
    plot_counts, manifest = [], []
    colors = ["#2166ac", "#d66022", "#278267"]
    def grid(title, subtitle):
        fig, axes = plt.subplots(2, 3, figsize=(17, 9.2))
        for ax, task in zip(axes.flat, tasks):
            inv = inventory[(inventory.task == task) & (inventory.supported == True)]
            partial = inv[inv.incomplete == True]
            ax.set_title(TITLES.get(task, task.replace("_", " ").title()) + f"\n{len(inv)} seeds; " +
                         ("partial: " + ", ".join(f"s{r.seed}:{r.completed_steps}" for r in partial.itertuples()) if len(partial) else "all complete"), fontsize=10)
        for ax in list(axes.flat)[len(tasks):]:
            ax.set_visible(False)
        fig.suptitle(title + "\n" + subtitle, fontsize=13, y=.985)
        return fig, axes
    def save(fig, name, optional=False):
        directory = output / "figures" / ("optional" if optional else "core")
        io_path(directory).mkdir(parents=True, exist_ok=True)
        handles, labels = [], []
        for ax in fig.axes:
            h, l = ax.get_legend_handles_labels()
            for a, b in zip(h, l):
                if b not in labels:
                    handles.append(a)
                    labels.append(b)
        if labels:
            fig.legend(handles, labels, loc="lower center", ncol=min(3, len(labels)), fontsize=9)
        fig.tight_layout(rect=(0, .075, 1, .915), h_pad=2)
        for ext in ["png", "pdf"]:
            fig.savefig(io_path(directory / (name + "." + ext)))
        manifest.append({"figure": name, "optional": optional, "task_order": tasks})
        plt.close(fig)
    def timeseries(spec, optional=False):
        name, title, series, ylabel, limits = spec
        rolling = any(key.endswith("_rolling") for key, _ in series)
        subtitle = ("Trailing 25 steps, >=10 valid values per seed; " if rolling else "Unsmoothed prefix trajectories; ") + "faint seeds, mean +/- 1 sample SD (between seeds)"
        if name.startswith(("06", "07")):
            subtitle += "; initial history included where available"
        fig, axes = grid(title, subtitle)
        for ax, task in zip(axes.flat, tasks):
            g = frame[frame.task == task]
            for idx, (key, label) in enumerate(series):
                color = colors[idx]
                for seed, sg in g.groupby("seed"):
                    partial = bool(inventory[(inventory.task == task) & (inventory.seed == seed)].iloc[0].incomplete)
                    x, y = sg.iteration.to_numpy(), sg[key].to_numpy()
                    if key == "current_best" and np.isfinite(sg.initial_best.iloc[0]):
                        x, y = np.r_[0, x], np.r_[sg.initial_best.iloc[0], y]
                    ax.plot(x, y, color=color, alpha=.20, lw=.9, linestyle=":" if partial else "-")
                    if partial and np.isfinite(y[-1]):
                        ax.scatter(x[-1], y[-1], marker="x", s=20, color=color, alpha=.6)
                stats = curve_statistics(g, key)
                for t, r in stats.iterrows():
                    plot_counts.append({"figure": name, "task": task, "series": key, "iteration": t,
                                        "n_seeds": int(r["n"]), "mean": r["mean"], "sd": r["sd"]})
                x, mean, sd = stats.index.to_numpy(), stats["mean"].to_numpy(), stats.sd.to_numpy()
                ax.plot(x, mean, color=color, lw=2.2, label=label, linestyle="--" if idx else "-")
                nonnegative = key not in {"current_best", "y_rolling"} or (g[key].dropna() >= 0).all()
                lower = np.maximum(0, mean - sd) if nonnegative else mean - sd
                upper = np.minimum(limits[1], mean + sd) if limits else mean + sd
                ax.fill_between(x, lower, upper, color=color, alpha=.16)
            if name.startswith("10_"):
                ax.axhline(.95, color="#555555", lw=1, linestyle="--", label="Nominal 0.95")
            ax.set(xlim=(0 if name.startswith("01_") else 1, max(planned_steps, int(frame.iteration.max()))),
                   xlabel="BO iteration", ylabel=ylabel)
            if limits:
                ax.set_ylim(*limits)
            elif all(key not in {"current_best", "y_rolling"} for key, _ in series):
                ax.set_ylim(bottom=0)
            ns = curve_statistics(g, series[0][0])["n"]
            ns = ns[ns > 0]
            ax.text(.02, .03, f"Seeds where displayed: {int(ns.min()) if len(ns) else 0}-{int(ns.max()) if len(ns) else 0}",
                    transform=ax.transAxes, fontsize=8, bbox={"facecolor": "white", "alpha": .75, "edgecolor": "none", "pad": 2})
        save(fig, name, optional)
    for spec in CORE:
        timeseries(spec)
    fig, axes = grid("GP: online predicted versus observed scores", "Pre-evaluation means; pooled RMSE is descriptive; seed RMSE mean +/- sample SD respects replication")
    for ax, task in zip(axes.flat, tasks):
        g = frame[(frame.task == task)].dropna(subset=["mu", "y"])
        ss = summary[summary.task == task].gp_rmse
        for lo, hi, label, color in [(1, 100, "Steps 1-100", "#80b1d3"), (101, 200, "Steps 101-200", "#36957b"), (201, np.inf, "Steps 201+", "#d66022")]:
            v = g[(g.iteration >= lo) & (g.iteration <= hi)]
            ax.scatter(v.mu, v.y, s=8, alpha=.22, color=color, label=label)
        if len(g):
            lo, hi = min(g.mu.min(), g.y.min()), max(g.mu.max(), g.y.max())
            ax.plot([lo, hi], [lo, hi], color="#555555", lw=1, linestyle="--", label="y = x")
        ax.set(xlabel="Online GP mean (objective units)", ylabel="Observed oracle score")
        ax.text(.03, .97, f"N={len(g)}; pooled RMSE={rmse(g):.4g}\nSeed RMSE={ss.mean():.4g} +/- {ss.std(ddof=1):.3g}; n={ss.notna().sum()}",
                transform=ax.transAxes, va="top", fontsize=8)
        plot_counts.append({"figure": "08_predicted_vs_observed", "task": task, "series": "scatter", "n_records": len(g), "n_seeds": int(ss.notna().sum())})
    save(fig, "08_predicted_vs_observed")
    fig, axes = grid("GP: prediction RMSE by logged chemical novelty", "Per-seed RMSE, then seed mean +/- sample SD; >=5 residuals per seed/bin; N counts all valid residuals")
    for ax, task in zip(axes.flat, tasks):
        g = bins[(bins.task == task) & (bins.row_type == "task")].set_index("bin").reindex(BIN_LABELS)
        means, sd = g.rmse_mean.to_numpy(), g.rmse_sd.to_numpy()
        ax.bar(range(4), means, color=colors[0], alpha=.7)
        finite_sd = np.isfinite(sd) & np.isfinite(means)
        ax.errorbar(np.arange(4)[finite_sd], means[finite_sd], yerr=np.vstack([np.minimum(sd[finite_sd], means[finite_sd]), sd[finite_sd]]), fmt="none", color="#222", capsize=4)
        for i, (_, r) in enumerate(g.iterrows()):
            ax.text(i, .02, f"N={int(r.n_valid)}\ns={int(r.n_seeds)}", transform=ax.get_xaxis_transform(), ha="center", fontsize=8)
        ax.set(xticks=range(4), xticklabels=BIN_LABELS, xlabel="Logged nearest-previous Tanimoto distance bin", ylabel="Online GP RMSE (objective units)")
        ax.set_ylim(bottom=0)
    save(fig, "11_gp_error_by_novelty")
    if include_optional:
        optional_specs = [
            ("rolling_evaluated_score", "Selected score and best observed score", [("y_rolling", "Rolling selected score"), ("current_best", "Best so far (unsmoothed)")], "Objective score", None),
            ("near_duplicate_fraction", "Fingerprint near-duplicate selections", [("near_duplicate_rolling", "Logged NN distance < 0.05"), ("exact_fp_duplicate_rolling", "Logged NN distance = 0")], "Fraction", (0, 1)),
            ("top10_scaffold_diversity", "Distinct raw Bemis-Murcko scaffolds in evaluated top-10", [("top10_scaffolds", "Raw scaffolds (empty excluded)")], "Scaffolds (count)", (0, 10)),
            ("gp_uncertainty_vs_error", "GP predictive uncertainty versus observed absolute error", [("abs_error_rolling", "Mean absolute error"), ("predictive_std_rolling", "Mean predictive std (with noise)")], "Objective units (sharpness diagnostic)", None),
        ]
        for spec in optional_specs:
            try:
                timeseries(spec, optional=True)
            except Exception as exc:
                manifest.append({"optional_skipped": spec[0], "reason": str(exc)})
        try:
            optional_plots(frame, summary, tasks, grid, save, output)
        except Exception as exc:
            manifest.append({"optional_skipped": str(exc)})
    write_csv(pd.DataFrame(plot_counts), output / "tables" / "plot_counts.csv")
    return manifest


def optional_plots(frame, summary, tasks, grid, save, output):
    """Small optional aggregate diagnostics; never prerequisites for core output."""
    fig, axes = grid("PI reliability on selected molecules", "Five equal-frequency bins (ties may reduce count); descriptive pooled points; uncertainty across whole seeds")
    reliability = []
    rng = np.random.default_rng(0)
    for ax, task in zip(axes.flat, tasks):
        g = frame[frame.task == task].dropna(subset=["pi", "success"]).copy()
        ax.plot([0, 1], [0, 1], "--", color="#777", label="Perfect calibration")
        if len(g):
            g["bin"] = pd.qcut(g.pi, 5, duplicates="drop")
            if g["bin"].isna().all():
                g["bin"] = "all tied"
            for label, b in g.groupby("bin", observed=True):
                rates = b.groupby("seed").success.agg(["sum", "count"])
                all_seeds = sorted(g.seed.unique())
                rates = rates.reindex(all_seeds, fill_value=0)
                boot = []
                if len(all_seeds) >= 2:
                    for _ in range(1000):
                        sample = rates.iloc[rng.integers(0, len(all_seeds), len(all_seeds))]
                        if sample["count"].sum():
                            boot.append(sample["sum"].sum() / sample["count"].sum())
                x, y = b.pi.mean(), b.success.mean()
                low, high = np.quantile(boot, [.025, .975]) if boot else (np.nan, np.nan)
                ax.scatter(x, y, color="#2166ac", s=30)
                if boot:
                    ax.vlines(x, low, high, color="#2166ac")
                ax.annotate(f"N={len(b)}", (x, y), xytext=(3, 4), textcoords="offset points", fontsize=8)
                reliability.append({"task": task, "bin": str(label), "n": len(b), "n_seeds": b.seed.nunique(),
                                    "pi": x, "success": y, "seed_bootstrap_low": low, "seed_bootstrap_high": high})
        ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Mean diagnostic PI", ylabel="Observed strict improvement fraction")
    save(fig, "pi_reliability_diagram", True)
    write_csv(pd.DataFrame(reliability), output / "tables" / "optional_pi_reliability.csv")
    fig, axes = grid("Performance and evaluated top-10 diversity", "Five seed points per task; AUC = mean best score over common horizon (no cross-task averaging)")
    for ax, task in zip(axes.flat, tasks):
        g = summary[summary.task == task]
        ax.scatter(g.final_top10_pairwise_distance, g.best_so_far_auc, color="#2166ac")
        for r in g.itertuples():
            ax.annotate(f"s{r.seed}", (r.final_top10_pairwise_distance, r.best_so_far_auc), xytext=(3, 3), textcoords="offset points")
        ax.set(xlim=(0, 1), xlabel="Final evaluated top-10 ECFP4 mean distance", ylabel="Common-horizon mean best score (AUC)")
    save(fig, "performance_diversity_tradeoff", True)
    # PCA is a deterministic exploratory projection; does not assert neighborhood fidelity.
    from sklearn.decomposition import PCA
    from rdkit import DataStructs
    fig, axes = grid("Exploratory chemical space: pooled-seed ECFP4 PCA", "One deterministic projection per task; color = BO iteration; size = within-task score rank; geometry is not chemical distance evidence")
    projection = None
    for ax, task in zip(axes.flat, tasks):
        g = frame[(frame.task == task) & (frame.valid_smiles == True)].dropna(subset=["y"]).copy()
        if len(g) < 2:
            continue
        arrays = []
        for s in g.canonical_smiles:
            a = np.zeros(2048, dtype=np.uint8)
            DataStructs.ConvertToNumpyArray(chemistry(s)[1], a)
            arrays.append(a)
        xy = PCA(n_components=2, svd_solver="randomized", random_state=0).fit_transform(np.asarray(arrays, dtype=float))
        projection = ax.scatter(xy[:, 0], xy[:, 1], c=g.iteration, cmap="viridis", vmin=1, vmax=frame.iteration.max(),
                                s=4 + 14 * g.y.rank(pct=True), alpha=.35)
        ax.set(xlabel="ECFP4 principal component 1", ylabel="ECFP4 principal component 2")
        write_csv(pd.DataFrame({"seed": g.seed.to_numpy(), "iteration": g.iteration.to_numpy(), "pc1": xy[:, 0], "pc2": xy[:, 1]}),
                  output / "tables" / ("optional_pca_" + task + ".csv"))
    if projection is not None:
        # Use the otherwise blank sixth panel for the iteration color key.
        axes.flat[-1].set_visible(True)
        axes.flat[-1].axis("off")
        fig.colorbar(projection, ax=axes.flat[-1], label="BO iteration", fraction=.5)
    save(fig, "chemical_space_2d", True)


METHODS = """# COWBOYS read-only analysis

This pipeline reads status/iteration/prediction JSON and NPZ (allow_pickle=False).
It never imports solvers, VAE, GP models, poli or oracles, refits predictions, or
calls objectives. Source SHA256 hashes are checked again after report generation.
Output must be outside, and must not contain, the raw-results root.

Discovery is recursive, restricted to solver_name=cowboys; preflight directories
are excluded. Task order is alphabetical. Default intended seeds are 1..5 plus
any discovered extra seeds; --seed restricts both discovery and intended grid.
All repeated (solver,task,seed) attempts are inventoried and excluded from seed
aggregation rather than choosing one silently. Missing grid entries are explicit.
Only batch size one and maximizing recorded objectives are supported. Direction
is verified against the audited current runner's maximizing convention; no score
normalization or chemical-desirability transform is introduced. DRD2 is a classifier.

Predictions join by BO step, canonical molecule, training count and (if saved)
observation index and pre-oracle capture marker. If absent, the exact saved timing
convention is required. Missing means do not become zeros. Invalid identities or
null outcomes are excluded from outcome metrics. Nonpositive/nonfinite uncertainty
is missing for PI, intervals and NLL (including sigma=0; no deterministic PI is
invented). RMSE can still use a valid mean with missing uncertainty. The raw stds
remain in CSV. b is the matched prediction's pre-evaluation incumbent; it is
checked against initial scores/previous logged best, never post-evaluation best.
Decision diagnostics exclude incompatible incumbents. At t=1 without history,
incumbent validity is unverified, explicitly flagged. No prediction is refitted.

Rolling curves use trailing 25 calendar BO steps, requiring 10 valid records
within that window, computed separately per seed before aggregation. Cumulative
curves are unsmoothed. Curves have gaps; no forward fill, padding or interpolation.
Faint seed lines, thick mean and +/-1 sample SD use available independent seeds
(ddof=1; no ribbon for one seed). These are variability bands, not confidence
intervals. Partial runs have dotted traces and endpoint crosses. Counts, means
and SD for every plotted time-series point are in tables/plot_counts.csv, and
per-seed window denominators are in per_iteration_metrics.csv. Scatter points
are descriptive adaptive samples, not experimental replicates. Every figure
uses the same task order and a 2x3 grid with the final panel blank for five tasks.

AUC = arithmetic mean of logged best-so-far at steps 1..T, where T is the minimum
available run horizon within task, excluding unsupported/duplicate attempts.
observed_horizon_auc is also reported but unequal-horizon values must not be
compared as full-budget performance. Endpoints and last-100 windows refer to each
run's own observed end; first-100 means steps <=100. Missing observations break
known no-improvement streaks. Counts and totals are available-case, not complete
totals when their denominators are below completed_steps. Last improvement is
unknown when all decisions are missing. Means/SD are always across seed scalars.
The configuration_summary.csv separates sampling caps; the overall task figures
describe a mixed-configuration progress collection, not a controlled comparison.

Logged NN distances are authoritative: the writer uses radius-3 folded count
fingerprints and excludes initial molecules from NN references. The first step
therefore normally has no logged NN distance. New chemical metrics instead use
Morgan radius 2, 2048 BIT fingerprints, chirality off, distance = 1 - Tanimoto.
Identity is canonical isomeric SMILES, with fragments retained; no tautomer or
charge normalization. Repeated identities retain their best observed score so
far. Top-10 evaluated structures use deterministic canonical-SMILES ties and
min(10,n) unique structures, with distance missing below two. Original aligned
initial history is included when available; otherwise chemistry is BO-selected
only while logged best-so-far may include missing initial observations.

tau is the retrospective 90th percentile of all valid evaluated scores within
each task (initial history included where available), fixed across its seeds.
No externally predefined cutoff is supplied by the current task configuration;
no external cutoff is invented. epsilon = .01*(P95-P5) of pooled selected scores.
Both thresholds are analysis-only and were unavailable during optimization.
Chronological greedy high-quality centers require y>=tau and distance >=.4
from every existing center. This #Circles proxy is order/threshold dependent,
not an exact maximum independent set. Quality-hit counts include repeated
evaluations; quality_unique_hits also reports unique identity denominators.
Empty raw Bemis-Murcko scaffolds are excluded; top10_no_scaffold is reported.

GP residual = observed - mean; predictive std includes observation noise.
Intervals use z=.67448975/1.28155157/1.96 for 50/80/95%; NLL uses the Gaussian
predictive density. PI=Phi((mu-b)/predictive_std), strict success=1[y>b], gain=
max(y-b,0). EI is the saved analytic LATENT Gaussian diagnostic, not the actual
MC qLogEI acquisition. Ratios of sums use paired nonmissing EI/gain only.
Unselected candidate scores/outcomes are never inferred from selection_score;
saved top-10 candidates are not extra oracle evaluations or MCMC effective
sample size. Generator counts describe structures decoded within/across sampling
iterations, not evaluations or proposal rates with unknown denominators.

Novelty bins are [0,.1), [.1,.2), [.2,.4), [.4,1.00001). Per-seed RMSE requires
>=5 valid residuals to contribute to bin aggregate means/sample SD. CSV retains
every seed/bin including empty/sparse bins and all metric denominators.
Optional reliability bins pool selected points descriptively; deterministic 95%
bootstrap intervals resample entire seeds (1000 samples, seed=0). Optional PCA
uses pooled-seed fingerprints separately per task (random_state=0); projection
geometry is not evidence of chemically meaningful neighborhoods.
"""


def format_stat(row, key):
    m, sd, n = row.get(key + "_mean"), row.get(key + "_sd"), row.get(key + "_n_seeds", 0)
    if not np.isfinite(number(m)):
        return "missing (n=0)"
    return f"{m:.4g}" + (f" +/- {sd:.3g}" if np.isfinite(number(sd)) else " (SD unavailable)") + f" (n={int(n)})"


def write_report(output, inventory, summary, task_summary, thresholds, diagnostics):
    overview = []
    for _, r in task_summary.iterrows():
        overview.append({"Task": r.task, **{label: format_stat(r, key) for label, key in [
            ("Final best", "final_best"), ("AUC", "best_so_far_auc"), ("Last-100 NN", "last100_nn_distance"),
            ("Top-10 distance", "final_top10_pairwise_distance"), ("Circles", "final_circles"),
            ("GP RMSE", "gp_rmse"), ("95% coverage", "coverage_95"), ("PI Brier", "pi_brier")]}})
    supported = inventory[inventory.supported == True]
    novelty_down = int((task_summary.last100_nn_distance_mean < task_summary.first100_nn_distance_mean).sum())
    error_down = int((task_summary.last100_gp_rmse_mean < task_summary.first100_gp_rmse_mean).sum())
    success_down = int((task_summary.last100_success_mean < task_summary.first100_success_mean).sum())
    pi_over = int((task_summary.last100_pi_mean > task_summary.last100_success_mean).sum())
    text = ["# COWBOYS experiment analysis", "",
            f"Analyzed {len(summary)} unambiguous supported runs across {summary.task.nunique()} tasks. "
            f"{int((~supported.incomplete).sum())} completed the requested {diagnostics['planned_steps']}-step budget; "
            f"{int(supported.incomplete.sum())} are partial. {diagnostics['validation_error_count']} validation errors are recorded. "
            "This is a descriptive progress collection across mixed sampling caps (10,000 versus 50,000), not a homogeneous algorithm comparison. "
            + "; ".join(f"cap {cap:g}: {len(g)} runs, {int((~g.incomplete).sum())} complete" for cap, g in supported.groupby("max_sampling_steps")) + ".", "",
            f"**Main findings:** task mean novelty falls in {novelty_down}/{len(task_summary)} tasks and selected-point GP RMSE falls in "
            f"{error_down}/{len(task_summary)}, comparing first with last 100 observed steps. Strict record-improvement frequency falls in "
            f"{success_down}/{len(task_summary)} tasks. Late diagnostic PI exceeds observed success in {pi_over}/{len(task_summary)}. "
            "High-quality center discovery nevertheless continues; reduced record gains alone do not establish a collapse in novel high-quality discovery.", "",
            "**History scope:** " + ("All analyzed runs have original initial molecules/scores; top-K and chemistry include initialization. "
              if not supported.history_missing.any() else "Some runs lack original initial history: their chemistry/top-K are BO-selected-only, while logged best-so-far can include initialization. ") +
            "Logged novelty always excludes initialization and uses the original radius-3 count fingerprint; new structural metrics use ECFP4 bits.", "",
            "Independent units are seeds. All overview entries are seed mean +/- sample SD with available-seed counts; partial endpoints refer to different horizons. "
            "AUC uses each task's common observed horizon. The GP/PI sample is selected by the optimization policy.", "",
            markdown_table(pd.DataFrame(overview)), "",
            "## 1. Optimization", "",
            "[P01 best-so-far](figures/core/01_best_so_far.png) · [P02 improvement frequency](figures/core/02_improvement_rate.png). "
            "Strict improvement uses the pre-evaluation incumbent. No missing decisions count as failures. "
            "A drop in an available-seed mean at a partial-run endpoint can reflect a change in contributing seeds rather than a fall in an individual incumbent."]
    for _, r in task_summary.iterrows():
        text.append(f"- **{r.task}**: final best {format_stat(r, 'final_best')}; common-horizon AUC {format_stat(r, 'best_so_far_auc')} "
                    f"over T={r.common_auc_horizon_mean:g}. Last record at {format_stat(r, 'last_improving_iteration')}; "
                    f"longest known stall {format_stat(r, 'longest_no_improvement_streak')} steps. "
                    f"Strict hit rate first/last 100: {format_stat(r, 'first100_success')} / {format_stat(r, 'last100_success')}; "
                    f"epsilon hit rate {format_stat(r, 'epsilon_improvement_frequency')}.")
    text += ["", "## 2. Search geography", "",
             "[P03 logged novelty](figures/core/03_nearest_previous_tanimoto_distance.png) · [P04 generator variety](figures/core/04_generator_variety.png) · "
             "[P05 shortlist](figures/core/05_top10_candidate_pool_size.png) · [P06 top-10 diversity](figures/core/06_top10_high_quality_diversity.png) · "
             "[P07 high-quality centers](figures/core/07_high_quality_circles.png). Counts describe candidate generation separately from evaluated chemistry. "
             "Shortlists are truncated to ten; they do not measure whole-chain mixing or effective sample size."]
    for _, r in task_summary.iterrows():
        th = thresholds[r.task]
        text.append(f"- **{r.task}**: mean logged novelty first/last 100 {format_stat(r, 'first100_nn_distance')} / {format_stat(r, 'last100_nn_distance')}; "
                    f"generator within-iteration unique count {format_stat(r, 'first100_generator_unique')} / {format_stat(r, 'last100_generator_unique')}; "
                    f"last-100 new generated {format_stat(r, 'last100_generator_new')}, shortlist {format_stat(r, 'last100_top10_candidates')}. "
                    f"Final top-10 distance {format_stat(r, 'final_top10_pairwise_distance')}. Retrospective tau={th['tau']:.6g}: "
                    f"{format_stat(r, 'final_circles')} centers from {format_stat(r, 'quality_hits')} qualifying evaluated hits "
                    f"({format_stat(r, 'quality_unique_hits')} unique molecules).")
    text += ["", "The chronological greedy #Circles proxy uses distance >=0.4, depends on threshold and arrival order, and is not an exact maximum packing. "
             "Low center counts may reflect few high-quality hits. No unselected-candidate outcomes are known.", "",
             "## 3. GP", "",
             "[P08 predicted/observed](figures/core/08_predicted_vs_observed.png) · [P09 online RMSE](figures/core/09_gp_rolling_rmse.png) · "
             "[P10 predictive coverage](figures/core/10_gp_95pct_coverage.png) · [P11 novelty strata](figures/core/11_gp_error_by_novelty.png). "
             "These are online pre-evaluation diagnostics on chosen molecules, not GP training errors or a global test set."]
    for _, r in task_summary.iterrows():
        text.append(f"- **{r.task}**: seed RMSE {format_stat(r, 'gp_rmse')}; first/last 100 RMSE {format_stat(r, 'first100_gp_rmse')} / "
                    f"{format_stat(r, 'last100_gp_rmse')}; predictive 95% coverage {format_stat(r, 'coverage_95')}; "
                    f"mean predictive std {format_stat(r, 'predictive_sharpness')}; signed bias (y-mu) {format_stat(r, 'gp_bias')}.")
    bs = pd.read_csv(io_path(output / "tables" / "05_gp_novelty_bins.csv"))
    for task, g in bs[bs.row_type == "task"].groupby("task"):
        g = g.set_index("bin")
        sparse = [label for label in BIN_LABELS if g.loc[label, "n_seeds"] < 2 or g.loc[label, "n_valid"] < 25]
        text.append(f"- **{task} novelty strata**: " + "; ".join(
            f"{label}: RMSE " + (f"{g.loc[label, 'rmse_mean']:.4g}" if np.isfinite(g.loc[label, 'rmse_mean']) else "missing") +
            f", N={int(g.loc[label, 'n_valid'])}, contributing seeds={int(g.loc[label, 'n_seeds'])}" for label in BIN_LABELS) +
            (". Sparse bins: " + ", ".join(sparse) + "; interpret comparisons cautiously." if sparse else "."))
    text += ["", "Missing prediction/uncertainty counts and metric-specific denominators are in T03 and the tidy data. "
             "Coverage uses predictive std including GP observation noise. Lower selected-point RMSE can follow a changing selection distribution.", "",
             "## 4. Decision quality", "",
             "[P12 expected/realized gain](figures/core/12_expected_vs_realized_improvement.png) · "
             "[P13 observed-outcome PI](figures/core/13_predicted_vs_observed_improvement_probability.png). "
             "Stored analytic latent-posterior EI is a diagnostic, distinct from the optimizer's Monte Carlo qLogEI. "
             "PI uses the predictive observation model and is not necessarily the solver's acquisition PI."]
    # Place the short GP interpretation before the decision-quality heading.
    bin_tasks = bs[bs.row_type == "task"].pivot(index="task", columns="bin", values="rmse_mean")
    comparable_bins = bin_tasks[[BIN_LABELS[2], BIN_LABELS[3]]].dropna()
    remote_harder = int((comparable_bins[BIN_LABELS[3]] > comparable_bins[BIN_LABELS[2]]).sum())
    coverage = task_summary.dropna(subset=["coverage_95_mean"])
    gp_interpretation = (f"The outer novelty bin has higher seed-mean RMSE than the [0.2,0.4) bin in "
                         f"{remote_harder}/{len(comparable_bins)} comparable tasks. This association is confounded with iteration and selection. ")
    if len(coverage):
        low = coverage.loc[coverage.coverage_95_mean.idxmin()]
        high = coverage.loc[coverage.coverage_95_mean.idxmax()]
        gp_interpretation += (f"Task-mean 95% coverage ranges from {low.coverage_95_mean:.3%} ({low.task}) to "
                              f"{high.coverage_95_mean:.3%} ({high.task}); values below/above nominal describe "
                              "undercoverage/overcoverage under the selected-molecule policy, without a formal significance claim.")
    heading_index = text.index("## 4. Decision quality")
    text.insert(heading_index, gp_interpretation)
    for _, r in task_summary.iterrows():
        text.append(f"- **{r.task}**: last-100 mean EI {format_stat(r, 'last100_ei')} versus realized positive gain {format_stat(r, 'last100_gain')}; "
                    f"last-100 mean PI {format_stat(r, 'last100_pi')} versus observed success {format_stat(r, 'last100_success')}; "
                    f"Brier {format_stat(r, 'pi_brier')}. Retrospective epsilon={thresholds[r.task]['epsilon']:.5g}.")
    over_ei = task_summary[task_summary.last100_ei_mean > task_summary.last100_gain_mean].task.tolist()
    text += ["", "Late mean analytic EI exceeds realized mean gain for " + ", ".join(over_ei) + ". "
             "This is descriptive over-promise along selected paths; EI is neither a guaranteed individual gain nor the acquisition the optimizer used. "
             "Predictive interval coverage and improvement-event calibration answer different questions: broad intervals can cover observations "
             "while PI still overpredicts rare record-breaking events."]
    text += ["", "## 5. Joint interpretation", "",
             "The table below compares each seed's first and last observed 100-step blocks (overlap is possible in short runs). "
             "Counts use only seeds with both block estimates. Negative novelty and RMSE changes alongside negative hit-rate changes "
             "describe coincident concentration, easier local prediction and diminished record discovery; patterns differ by task."]
    joint = []
    for task, g in summary.groupby("task"):
        row = {"Task": task}
        deltas = []
        for label, before, after in [("NN change", "first100_nn_distance", "last100_nn_distance"),
                                     ("RMSE change", "first100_gp_rmse", "last100_gp_rmse"),
                                     ("Hit-rate change", "first100_success", "last100_success")]:
            d = g[after] - g[before]
            row[label] = f"{d.mean():.4g} +/- {d.std(ddof=1):.3g} (n={d.notna().sum()})"
            deltas.append(d)
        coincident = pd.concat(deltas, axis=1).dropna()
        row["Seeds with all three decreasing"] = f"{int((coincident < 0).all(axis=1).sum())}/{len(coincident)}"
        row["First-100 new quality centers"] = f"{g.first100_circles_added.mean():.3g} +/- {g.first100_circles_added.std(ddof=1):.3g}"
        row["Last-100 new quality centers"] = f"{g.last100_circles_added.mean():.3g} +/- {g.last100_circles_added.std(ddof=1):.3g}"
        joint.append(row)
    text += ["", markdown_table(pd.DataFrame(joint)), "",
             "The threshold-based high-quality centers can continue to grow after record improvement becomes rare. "
             "Their first-to-last block changes must be read directly rather than assuming that less novelty implies less high-quality discovery. "
             "These are associations, not evidence of cause. Time, adaptive selection, task score plateaus and mixed sampling caps are confounded. "
             "The GP-error distribution is selection-biased: improved predictability on nearby chosen structures does not establish globally improved accuracy. "
             "Novel high-quality discovery is assessed through newly accepted centers as well as record gains; a plateau alone does not prove chemical-space collapse. "
             "A fixed held-out diagnostic molecular panel spanning novelty ranges would be a useful **future experiment**; it is outside this read-only analysis.", "",
             "## Audit and tables", "",
             "- [Run inventory](tables/00_run_inventory.html), including partial/missing/ambiguous attempts.",
             "- [Optimization](tables/01_optimization_summary.html), [diversity](tables/02_diversity_summary.html), "
             "[GP](tables/03_gp_summary.html), [acquisition](tables/04_acquisition_summary.html), [novelty bins](tables/05_gp_novelty_bins.html).",
             "- [Configuration-separated seed summaries](configuration_summary.csv), [per-seed summary](per_seed_summary.csv), "
             "[per-task summary](per_task_summary.csv), [tidy observations](per_iteration_metrics.csv), [plot counts](tables/plot_counts.csv).",
             "- [Methods](README.md), [validation, thresholds and source hashes](diagnostics.json). No missing value is zero-filled.", ""]
    io_path(output / "report.md").write_text("\n\n".join(part for part in text if part), encoding="utf-8")


def analyze(input_root, output_root, tasks=None, seeds=None, planned_steps=300, include_optional=False):
    source, output = Path(input_root).resolve(), Path(output_root).resolve()
    if output == source or source in output.parents or output in source.parents:
        raise ValueError("Output and raw-results roots must be disjoint (neither may contain the other)")
    if not source.is_dir():
        raise ValueError("Input root does not exist")
    if planned_steps < 1:
        raise ValueError("planned_steps must be positive")
    paths, ignored, discovery_errors = [], [], []
    for path in sorted(io_path(source).rglob("*_status.json")):
        if any(p.lower() == "preflight" for p in path.relative_to(io_path(source)).parts):
            ignored.append(str(path))
            continue
        try:
            s = read_json(path)
            if (s.get("solver_name") == "cowboys" and (not tasks or s.get("function_name") in tasks)
                    and (not seeds or s.get("seed") in seeds)):
                paths.append(path)
        except (OSError, ValueError) as exc:
            discovery_errors.append({"path": str(path), "error": str(exc)})
    identities = defaultdict(list)
    for p in paths:
        s = read_json(p)
        identities[(s.get("solver_name"), s.get("function_name"), s.get("seed"))].append(p)
    runs, inventories, failures, source_hashes = [], [], [], {}
    for identity, attempts in sorted(identities.items()):
        for path in attempts:
            try:
                run = load_run(path, planned_steps)
                for entry in run["sources"]:
                    source_hashes[entry["path"]] = entry["sha256"]
                if len(attempts) > 1:
                    run["inventory"]["supported"] = False
                    run["inventory"]["status"] = "ambiguous_duplicate_attempt"
                    run["rows"] = []
                    run["errors"].append("Repeated run identity: all attempts excluded; select a narrower input root")
                inventories.append(run["inventory"])
                failures.append({"status_file": str(path), "warnings": run["warnings"], "errors": run["errors"]})
                if run["rows"]:
                    runs.append(run)
            except (OSError, ValueError, TypeError, KeyError) as exc:
                inventories.append({"task": identity[1], "seed": identity[2], "status": "unreadable",
                                    "supported": False, "incomplete": True, "status_file": str(path)})
                failures.append({"status_file": str(path), "errors": [str(exc)], "warnings": []})
    task_order = sorted(set(tasks or [key[1] for key in identities]))
    seed_order = sorted(set(seeds or list(range(1, 6)) + [key[2] for key in identities]))
    for task, seed in product(task_order, seed_order):
        if ("cowboys", task, seed) not in identities:
            inventories.append({"task": task, "seed": seed, "status": "missing_run", "supported": False, "incomplete": True})
    if len(task_order) > 5:
        raise ValueError("Select at most five tasks with repeated --task for the standardized 2x3 report")
    if not runs:
        # Still produce an actionable inventory, even if no curves can be plotted.
        io_path(output).mkdir(parents=True, exist_ok=True)
        write_csv(pd.DataFrame(inventories), output / "run_inventory.csv")
        write_json(output / "diagnostics.json", {"runs": failures, "discovery_errors": discovery_errors})
        raise ValueError("No supported, unambiguous iteration records; see output inventory/diagnostics")
    thresholds = thresholds_for(runs)
    for run in runs:
        add_chemistry(run, thresholds[run["inventory"]["task"]])
        # Initial accepted-center count follows the same chronology/threshold policy.
        init_centers = []
        for key, y, fp, _ in run["initial"]:
            if key and np.isfinite(y) and y >= thresholds[run["inventory"]["task"]]["tau"] and all(distance(fp, old) >= .4 for old in init_centers):
                init_centers.append(fp)
        for row in run["rows"]:
            row["initial_circles"] = len(init_centers)
    data = rolling_frame(pd.DataFrame([row for run in runs for row in run["rows"]]))
    inventory = pd.DataFrame(inventories).sort_values(["task", "seed", "status"])
    horizons = data.groupby(["task", "seed"]).iteration.max().groupby("task").min().to_dict()
    summaries = [summarize_run(pd.DataFrame(run["rows"]), run["inventory"], horizons[run["inventory"]["task"]]) for run in runs]
    summary = pd.DataFrame(summaries)
    task_summary = aggregate_seeds(summary)
    bins = novelty_bins(data)
    io_path(output).mkdir(parents=True, exist_ok=True)
    for name, frame in [("run_inventory", inventory), ("per_iteration_metrics", data),
                        ("per_seed_summary", summary), ("per_task_summary", task_summary),
                        ("configuration_summary", aggregate_seeds(summary, ("task", "max_sampling_steps")))]:
        write_csv(frame, output / (name + ".csv"))
    save_table(inventory, output, "00_run_inventory")
    categories = {
        "01_optimization_summary": ["completed_steps", "common_auc_horizon", "final_best", "best_so_far_auc", "observed_horizon_auc",
                                    "strict_improvements", "improvement_frequency", "epsilon_improvement_frequency", "last_improving_iteration",
                                    "longest_no_improvement_streak", "total_realized_gain", "n_success", "n_gain", "missing_success"],
        "02_diversity_summary": [c for c in summary if any(s in c for s in ["nn_distance", "duplicate", "generator", "top10", "quality", "circles", "tau"])],
        "03_gp_summary": [c for c in summary if any(s in c for s in ["gp_", "coverage", "sharpness", "mean_z", "nll", "n_mu", "missing_mu", "predictive_std", "n_error", "missing_error", "n_z", "missing_z"])],
        "04_acquisition_summary": [c for c in summary if any(s in c for s in ["ei", "gain", "success", "frequency", "mean_pi", "last100_pi", "brier", "epsilon"])],
    }
    for name, columns in categories.items():
        columns = list(dict.fromkeys(columns))
        seed_cols = list(dict.fromkeys(["task", "seed", "row_type", "status", "completed_steps", "max_sampling_steps", *columns]))
        task_cols = ["task", "row_type", "n_seeds", "n_complete"] + [key for c in columns for key in [c + "_mean", c + "_sd", c + "_n_seeds"] if key in task_summary]
        save_table(pd.concat([summary[seed_cols], task_summary[task_cols]], ignore_index=True), output, name)
    save_table(bins, output, "05_gp_novelty_bins")
    plots = make_plots(data, inventory, bins, summary, output, task_order, planned_steps, include_optional)
    changed = [p for p, before in source_hashes.items() if digest_file(Path(p)) != before]
    diagnostics = {"input_root": str(source), "output_root": str(output), "planned_steps": planned_steps,
                   "task_order": task_order, "intended_seeds": seed_order, "thresholds": thresholds,
                   "fingerprint": FINGERPRINT, "logged_novelty_fingerprint": "radius 3 folded count, BO-only references; original logged values",
                   "rolling_window": WINDOW, "minimum_valid_per_window": MIN_VALID,
                   "runs": failures, "discovery_errors": discovery_errors, "ignored_preflights": ignored,
                   "validation_error_count": sum(len(r["errors"]) for r in failures) + len(discovery_errors),
                   "source_sha256": source_hashes, "source_files_unchanged": not changed,
                   "changed_sources": changed, "plots": plots, "common_auc_horizons": horizons,
                   "sampling_cap_counts": inventory.max_sampling_steps.value_counts().to_dict(),
                   "versions": {"numpy": np.__version__, "pandas": pd.__version__}}
    write_json(output / "diagnostics.json", diagnostics)
    io_path(output / "README.md").write_text(METHODS + "\n\nInvocation: `python -m hdbo_benchmark.evaluation.cowboys_analysis "
        f'--input-root "{source}" --output-root "{output}"' + (" --include-optional" if include_optional else "") + "`\n", encoding="utf-8")
    write_report(output, inventory, summary, task_summary, thresholds, diagnostics)
    if changed:
        raise RuntimeError("Raw source hashes changed during analysis; see diagnostics.json")
    return diagnostics


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--task", action="append", help="Exact task name; repeat to filter")
    parser.add_argument("--seed", action="append", type=int, help="Seed; repeat to filter")
    parser.add_argument("--planned-steps", type=int, default=300)
    parser.add_argument("--include-optional", action="store_true")
    args = parser.parse_args(argv)
    result = analyze(args.input_root, args.output_root, args.task, args.seed, args.planned_steps, args.include_optional)
    print(f"Wrote {args.output_root / 'report.md'}; validation errors={result['validation_error_count']}; "
          f"raw sources unchanged={result['source_files_unchanged']}")


if __name__ == "__main__":
    main()
