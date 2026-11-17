"""Self-contained offline run.json export over existing COWBOYS results."""

import importlib.metadata
import platform
from pathlib import Path

from .cowboys_artifacts import load_cowboys
from .metric_io import digest_json, display_path, io_path, read_json, write_json
from .molecular_metrics import FINGERPRINT, HITS, QUALITY, compute_metrics, finite, normalized_auc


def _version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _iteration_observations(iteration, saved_config, n_initial, direction):
    """Retain selected observations when only iteration JSON survives.

    Initialization is absent, so no full-prefix molecular trajectory is computed.
    Older JSON contains re-decoded latent structures, not verified observations.
    """
    rows = iteration.get("iterations", [])
    scores, structures, steps = [], [], []
    original_structures = saved_config.get("gp_observations") == "actual_evaluated_structures"
    last_step = 0
    for row in rows:
        step = row.get("iteration")
        if not isinstance(step, int) or step <= last_step:
            raise ValueError("Iteration records must have increasing positive steps")
        last_step = step
        values = row.get("selected_objective_values", [])
        selected = row.get("selected_decoded_structures", [])
        if original_structures and len(selected) != len(values):
            raise ValueError("Iteration selected structures and scores disagree")
        scores.extend(values)
        structures.extend(selected if original_structures else [None] * len(values))
        steps.extend([step] * len(values))
    observations, _, _ = compute_metrics(scores, structures if original_structures else None,
                                         0, {}, direction=direction)
    for index, (record, step) in enumerate(zip(observations, steps)):
        record.update(bo_step=step, phase="optimization", step_assignment="recorded_iteration_JSON",
                      is_duplicate=None, novelty_to_initial=None, realized_improvement=None,
                      cumulative_observed_evaluations=None,
                      observation_index=(n_initial + index if isinstance(n_initial, int) else index))
        record["missing_reason"].update(is_duplicate="Initial identities unavailable; no complete duplicate baseline",
                                       realized_improvement="No complete pre-evaluation observation prefix",
                                       cumulative_observed_evaluations="Initialization observations missing",
                                       observation_index="Offset inferred from saved initialization size" if isinstance(n_initial, int)
                                       else "Index within surviving BO records only; absolute history position unavailable")
    return observations


def build_run(status_path, *, task_config=None, planned_bo_steps=None):
    artifact = load_cowboys(status_path)
    status, warnings = artifact["status"], artifact["warnings"]
    config = {} if task_config is None else task_config
    if not isinstance(config, dict):
        raise ValueError("Each task configuration must be a JSON object")
    unknown = set(config) - {"hit_threshold", "delta", "score_direction", "bo_steps_planned",
                             "oracle_version", "initialization_policy"}
    if unknown:
        raise ValueError(f"Unsupported task metric configuration keys: {sorted(unknown)}")
    direction = config.get("score_direction", "maximize")
    theta, delta = config.get("hit_threshold"), config.get("delta", .7)
    planned = config.get("bo_steps_planned", planned_bo_steps)
    if planned is not None and (not isinstance(planned, int) or isinstance(planned, bool) or planned < 0):
        raise ValueError("bo_steps_planned must be a nonnegative integer")
    saved_config = status.get("solver_configuration", {})
    iteration = artifact["iterations"] or {}
    n_initial = status.get("initial_history_size")
    batch = saved_config.get("batch_size", iteration.get("batch_size"))
    completed = status.get("completed_iterations")
    boundaries_known = all(isinstance(v, int) and not isinstance(v, bool) and v >= minimum
                           for v, minimum in ((n_initial, 0), (batch, 1), (completed, 0)))
    scores, structures = artifact["scores"], artifact["structures"]
    ends, mismatch = {}, False
    if boundaries_known and scores is not None and len(scores) >= n_initial:
        horizon = min(completed, (len(scores) - n_initial) // batch)
        ends = {step: n_initial + step * batch for step in range(horizon + 1)}
        mismatch = len(scores) != n_initial + completed * batch
        if mismatch:
            warnings.append("Status and history length disagree; trajectory uses only whole saved completed steps")
    elif scores is not None:
        warnings.append("Initialization/batch/completed-step metadata unavailable or initialization incomplete; no prefix trajectory")
    evaluations, trajectory, representatives = [], [], []
    if scores is not None:
        # Parse raw observations even if phase/boundaries cannot be recovered.
        evaluations, trajectory, representatives = compute_metrics(
            scores, structures, n_initial if isinstance(n_initial, int) else 0, ends,
            theta=theta, delta=delta, direction=direction)
        if not isinstance(n_initial, int):
            for record in evaluations:
                record.update(bo_step=None, phase=None, step_assignment=None)
                record["missing_reason"]["phase"] = "Initialization boundary not recorded"
    else:
        # Validate metric configuration even for status-only exports.
        compute_metrics([], None, 0, {}, theta=theta, delta=delta, direction=direction)
        if iteration:
            evaluations = _iteration_observations(iteration, saved_config, n_initial, direction)
            warnings.append("Only iteration-selected observations survive; initialization scores/structures unavailable, so no prefix trajectory")

    source_status = status.get("status", "unknown")
    horizon = trajectory[-1]["bo_step"] if trajectory else None
    fully_saved = horizon is not None and horizon == completed and not mismatch
    if source_status == "completed" and fully_saved and (planned is None or horizon == planned):
        exported_status = "completed"
    elif source_status in {"failed", "oom"} and not trajectory:
        exported_status = "failed"
    else:
        exported_status = "partial"
    full = exported_status == "completed" and planned is not None and horizon == planned
    scope = "full_budget" if full else ("unverified_budget" if exported_status == "completed" else "partial")
    if planned is None:
        warnings.append("Planned budget absent from source; supply --planned-bo-steps only with protocol evidence")
    if status["function_name"] == "sa_tdc" and direction == "maximize":
        warnings.append("sa_tdc preserves the recorded maximizing protocol; raw SA is lower-is-easier (TD-16)")
    missing = {
        "actual_oracle_calls": "No physical-call/cache flags; history rows are not measured oracle calls",
        "proposal_rates": "No complete proposal chronology or denominator",
        "timings": "No timestamps or component runtime records",
        "gp_predictions_and_calibration": "No pre-evaluation GP means/stds or intervals",
        "oracle_version": "Not recorded in original artifacts" if not config.get("oracle_version") else None,
        "optimization_environment": "Historical git revision/dependency snapshot not recorded",
        "initial_design_id": "No logged initialization set identifier; content hash supplied separately",
        "initialization_policy": "Original initialization sampling policy not logged" if not config.get("initialization_policy") else None,
        "approx_circles": "Optional approximation not implemented",
    }
    if theta is None:
        missing["hit_metrics"] = "No task-specific hit_threshold configured"
    if structures is None:
        missing["molecular_metrics"] = "Original chronological structures absent/misaligned; Top-K and structural metrics unavailable"
    if planned is None:
        missing["planned_budget"] = "Not recorded; not inferred from completed iterations"
    if not trajectory:
        missing["trajectory"] = "Aligned scores and initialization/completed-step boundaries unavailable"
    missing = {key: value for key, value in missing.items() if value is not None}
    auc, auc_reasons = {}, {}
    for name in QUALITY:
        auc[name], reason = normalized_auc(trajectory, lambda p, key=name: p["quality"][key])
        if reason:
            auc_reasons[name] = reason
    diversity_summary = {}
    for name in HITS:
        value, reason = normalized_auc(trajectory, lambda p, key=name: p["diversity_by_profile"]["primary"][key])
        diversity_summary[f"auc_{name}"] = value
        diversity_summary[f"final_{name}"] = trajectory[-1]["diversity_by_profile"]["primary"][name] if trajectory else None
        if reason:
            auc_reasons[name] = reason
    first = next((p["bo_step"] for p in trajectory if theta is not None
                  and p["quality"]["best_so_far"] is not None and p["quality"]["best_so_far"] >= theta), None)
    policy = {
        "top_k": [1, 10, 100], "top_k_if_fewer_than_k": "mean_available_unique_valid",
        "molecular_identity": "RDKit canonical isomeric SMILES; all fragments; no neutralization/tautomer normalization",
        "duplicate_score_policy": "maximum_score_observed_so_far_per_identity; first-seen tie order",
        "valid_observation_policy": "valid_molecule_and_finite_score" if structures is not None else "finite_scores_only; molecular_validity_unverified",
        "auc_rule": "normalized_trapezoid_including_step_0_divided_by_observed_BO_horizon",
        "auc_axis": "completed_BO_steps_after_initialization", "official_PMO_leaderboard_AUC": False,
        "fingerprint": FINGERPRINT,
        "diversity_profiles": {"primary": {"hit_threshold": theta, "delta": delta,
            "distance": "1-Tanimoto", "pairwise_condition": "strictly_greater_than",
            "archive_policy": "chronological_first_threshold_crossing_including_initialization"}},
        "scaffold_policy": "distinct_nonempty_Bemis_Murcko; acyclic_empty_scaffolds_excluded",
        "unique_hit_rate_denominator": "physical_oracle_calls; unavailable_without_call_flags",
    }
    initial_hash = digest_json([[e["observed_structure"], e["raw_oracle_score"]] for e in evaluations[:n_initial]]) if scores is not None and isinstance(n_initial, int) and len(evaluations) >= n_initial else None
    run_id = digest_json({"source": display_path(status_path), "config": saved_config})[:20]
    final = dict(trajectory[-1]["quality"]) if trajectory else {name: None for name in QUALITY}
    # A status-only artifact supports its recorded final maximum, not an invented curve.
    if not trajectory and direction == "maximize":
        final["best_so_far"] = finite(status.get("best_value"))
    final["top_10_internal_diversity"] = trajectory[-1]["top_10_internal_diversity"] if trajectory else None
    # Preserve diagnostics separately: ranked candidates are never treated as evaluated molecules.
    records = {row.get("iteration"): row for row in iteration.get("iterations", [])}
    for observation in evaluations:
        row = records.get(observation["bo_step"], {})
        matches = [candidate for candidate in row.get("top_candidate_diagnostics", [])
                   if candidate.get("canonical_smiles") == observation["canonical_smiles"]
                   and observation["canonical_smiles"] is not None]
        if len(matches) == 1:
            observation["recorded_acquisition"] = {"selection_score": matches[0].get("selection_score"),
                "convention": "qLogExpectedImprovement" if saved_config.get("implementation") == "cowboys_corrected_pcn_v1" else "unspecified_selection_score",
                "source": "iteration top_candidate_diagnostics matched by canonical identity; no GP recomputation"}
    return {
        "schema_version": "1.0.0", "run_id": run_id,
        "experiment_group_id": display_path(status_path.parent), "status": exported_status,
        "source_status": source_status, "source_result_path": display_path(status_path),
        "provenance": {"adapter": "cowboys_checkpoint_v1", "sources": artifact["sources"],
                       "optimization_environment": None,
                       "export_environment": {"python": platform.python_version(),
                           **{name: _version(name) for name in ("numpy", "rdkit", "selfies")}}},
        "task": {"benchmark": "POLI/PMO" if status["function_name"] in _pmo_tasks() else None,
                 "task_name": status["function_name"], "oracle_version": config.get("oracle_version"),
                 "direction": direction, "score_transform": "identity" if direction == "maximize" else "negate_recorded_score",
                 "score_normalization": "none; recorded objective scale", "score_range": None,
                 "direction_source": "task_config" if "score_direction" in config else "existing_runner_maximizes_recorded_objective"},
        "algorithm": {"name": "COWBOYS", "config": saved_config},
        "design": {"seed": status["seed"], "initial_design_id": None,
                   "initial_content_sha256": initial_hash,
                   "n_initial_planned": None, "n_initial_observed": n_initial,
                   "max_oracle_calls_planned": None,
                   "initialization_policy": config.get("initialization_policy", "source_history_initial_prefix; original sampling policy not logged"),
                   "initialization_policy_source": "task_config" if config.get("initialization_policy") else "not_recorded",
                   "bo_steps_planned": planned, "batch_size": batch, "budget_unit": "BO_steps",
                   "planned_budget_source": "task_config" if "bo_steps_planned" in config else
                       ("export_argument" if planned is not None else None)},
        "metric_policy": policy, "evaluations": evaluations, "trajectory": trajectory,
        "summary": {"available": bool(trajectory), "full_budget_completed": full,
                    "auc_scope": scope, "auc_horizon": horizon,
                    "n_observed_evaluations": len(evaluations),
                    "observation_history_complete": scores is not None and bool(ends) and not mismatch,
                    "n_completed_bo_steps": horizon, "n_oracle_calls_total": None,
                    "diagnostic_objective_evaluations_enabled": status.get("diagnostic_objective_evaluations_enabled"),
                    "final": final, "auc": auc,
                    "final_source": "trajectory" if trajectory else "recorded_status_best_value_only",
                    "final_missing_reason": trajectory[-1]["missing_reason"] if trajectory else
                        {"top_k_and_diversity": "Complete observation prefix unavailable; recorded status maximum alone does not determine Top-K"},
                    "diversity_by_profile": {"primary": {**diversity_summary,
                        "first_step_reaching_threshold": first,
                        "threshold_censored": first is None if theta is not None and trajectory else None,
                        "representatives": representatives}},
                    "auc_missing_reason": auc_reasons,
                    "missing_reason": missing},
        "recorded_iteration_diagnostics": iteration if iteration else None,
        "missing_reason": missing, "warnings": warnings,
    }


def _pmo_tasks():
    # Local audited task catalog, never initialize poli/PMO or an oracle for metadata.
    catalog = Path(__file__).resolve().parents[3] / "configs/cowboys_paper_table1.json"
    return read_json(catalog)["tasks"] if io_path(catalog).exists() else {}


def export_directory(results_dir, *, recursive=False, task_configs=None, planned_bo_steps=None):
    root = Path(results_dir).resolve()
    if not io_path(root).is_dir():
        raise ValueError(f"Results directory does not exist: {root}")
    if task_configs is not None and not isinstance(task_configs, dict):
        raise ValueError("--task-config must be a JSON object keyed by task name")
    paths = sorted(io_path(root).rglob("cowboys_*_status.json") if recursive else io_path(root).glob("cowboys_*_status.json"))
    outputs, issues, seen_families = [], [], set()
    for path in paths:
        relative = path.relative_to(io_path(root))
        if "preflight" in relative.parts or "metrics" in relative.parts:
            continue
        seen_families.add(path.with_name(path.name.removesuffix("_status.json")))
        try:
            metadata = read_json(path)
            if not isinstance(metadata, dict):
                raise ValueError("Status JSON must be an object")
            task = metadata.get("function_name")
            run = build_run(path, task_config=(task_configs or {}).get(task), planned_bo_steps=planned_bo_steps)
            validate_run(run)
            output = path.parent / "metrics" / path.name.removesuffix("_status.json") / "run.json"
            write_json(output, run)
            outputs.append({"run_id": run["run_id"], "path": display_path(output), "status": run["status"]})
        except (ValueError, OSError, KeyError, TypeError) as exc:
            issues.append({"source": display_path(path), "missing_reason": str(exc)})
    # Report artifacts whose provenance/boundaries cannot safely be inferred from names.
    candidates = io_path(root).rglob("cowboys_*.n*") if recursive else io_path(root).glob("cowboys_*.n*")
    for path in sorted(candidates):
        if any(part in {"metrics", "preflight"} for part in path.relative_to(io_path(root)).parts):
            continue
        stem = path.name.removesuffix("_history.npz").removesuffix(".npy")
        if path.with_name(stem) not in seen_families:
            issues.append({"source": display_path(path), "missing_reason": "No companion status metadata; cannot establish task/seed/initialization safely"})
    sweep = io_path(root) / "sweep_status.tsv"
    if sweep.exists():
        import csv
        with sweep.open(encoding="utf-8", newline="") as stream:
            for row in csv.DictReader(stream, delimiter="\t"):
                expected = io_path(root) / f"cowboys_{row['task']}_{row['seed']}_status.json"
                if not expected.exists():
                    issues.append({"task": row["task"], "seed": row["seed"], "exit_code": row["exit_code"],
                                   "missing_reason": "No checkpoint; failure/interruption before first save or missing artifact"})
    manifest = {"schema_version": "1.0.0", "results_dir": display_path(root),
                "recursive": recursive, "runs": outputs, "issues": issues}
    write_json(root / "metrics/export_manifest.json", manifest)
    return manifest


def validate_run(run, *, schema=True):
    if schema:
        try:
            import jsonschema
        except ImportError:
            pass  # Semantic validation remains available without optional jsonschema.
        else:
            try:
                jsonschema.Draft202012Validator(read_json(Path(__file__).with_name("run_schema.json"))).validate(run)
            except jsonschema.ValidationError as exc:
                raise ValueError(f"Run schema violation at {list(exc.absolute_path)}: {exc.message}") from exc
    trajectory = run["trajectory"]
    if trajectory and [p["bo_step"] for p in trajectory] != list(range(len(trajectory))):
        raise ValueError("Trajectory must contain exactly steps 0..B")
    if trajectory:
        if run["summary"]["auc_horizon"] != trajectory[-1]["bo_step"]:
            raise ValueError("Summary horizon disagrees with trajectory")
        if run["summary"]["n_completed_bo_steps"] != trajectory[-1]["bo_step"]:
            raise ValueError("Completed step count disagrees with trajectory")
        for key in QUALITY:
            if run["summary"]["final"][key] != trajectory[-1]["quality"][key]:
                raise ValueError("Final summary disagrees with trajectory")
    for point in trajectory:
        quality = point["quality"]
        if quality["top_1"] is not None and quality["top_1"] != quality["best_so_far"]:
            raise ValueError("Top-1 and best-so-far disagree")
    for getter in (lambda p: p["quality"]["best_so_far"],
                   lambda p: p["diversity_by_profile"]["primary"]["online_diverse_hits"]):
        values = [getter(p) for p in trajectory if getter(p) is not None]
        if any(b < a for a, b in zip(values, values[1:])):
            raise ValueError("Cumulative trajectory must be nondecreasing")
    for name in QUALITY:
        value, _ = normalized_auc(trajectory, lambda p: p["quality"][name])
        actual = run["summary"]["auc"][name]
        if (value is None) != (actual is None) or (value is not None and abs(value - actual) > 1e-12):
            raise ValueError("Summary AUC disagrees with trajectory")
    for name in HITS:
        profile = run["summary"]["diversity_by_profile"]["primary"]
        value, _ = normalized_auc(trajectory, lambda p: p["diversity_by_profile"]["primary"][name])
        actual = profile[f"auc_{name}"]
        if (value is None) != (actual is None) or (value is not None and abs(value - actual) > 1e-12):
            raise ValueError("Summary hit AUC disagrees with trajectory")
    if run["summary"]["full_budget_completed"]:
        if run["status"] != "completed" or not trajectory or trajectory[-1]["bo_step"] != run["design"]["bo_steps_planned"]:
            raise ValueError("Full-budget completion requires an explicitly matched planned horizon")
    if (run["summary"]["auc_scope"] == "full_budget") != run["summary"]["full_budget_completed"]:
        raise ValueError("AUC scope disagrees with budget completion flag")
    # Reject invalid JSON numbers even when jsonschema is unavailable.
    import json
    json.dumps(run, allow_nan=False)
