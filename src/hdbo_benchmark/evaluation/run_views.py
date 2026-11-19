"""Readable views alongside the complete run record; preserve every metric."""

import csv

from .metric_io import digest_file, io_path, write_json


EVALUATION_COLUMNS = (
    "observation_index", "bo_step", "phase", "canonical_smiles", "valid_molecule", "is_duplicate",
    "actual_score", "score_for_metrics", "predicted_mean", "latent_std", "predictive_std",
    "analytic_expected_improvement", "recorded_log_ei", "realized_improvement", "prediction_residual",
    "predictive_gaussian_nll", "covered_by_predictive_90pct_interval", "novelty_to_initial",
)


def evaluation_rows(run):
    for e in run["evaluations"]:
        prediction = e.get("pre_evaluation_prediction") or {}
        acquisition = e.get("recorded_acquisition") or {}
        yield {
            **{key: e.get(key) for key in EVALUATION_COLUMNS},
            "actual_score": e["raw_oracle_score"], "predicted_mean": prediction.get("mean"),
            "latent_std": prediction.get("latent_std"), "predictive_std": prediction.get("predictive_std"),
            "analytic_expected_improvement": prediction.get("analytic_expected_improvement"),
            "recorded_log_ei": acquisition.get("selection_score"),
        }


def write_run_views(folder, run):
    """Large ranked-candidate/latent arrays have their own lossless sidecar."""
    raw_diagnostics = run.pop("recorded_iteration_diagnostics", None)
    artifacts = {"evaluations_table": "evaluations.csv", "readable_summary": "summary.json"}
    if raw_diagnostics is not None:
        diagnostic_path = folder / "diagnostics.json"
        write_json(diagnostic_path, raw_diagnostics)
        artifacts["recorded_iteration_diagnostics"] = {
            "path": "diagnostics.json", "sha256": digest_file(diagnostic_path),
            "description": "Lossless original iteration metrics, ranked candidates and latent arrays",
        }
    run["artifacts"] = artifacts
    # Field names and scientific metric contents remain backward compatible.
    write_json(folder / "run.json", run)
    summary = {
        "schema_version": run["schema_version"], "run_id": run["run_id"], "status": run["status"],
        "task": run["task"]["task_name"], "algorithm": run["algorithm"]["name"],
        "seed": run["design"]["seed"], "planned_bo_steps": run["design"]["bo_steps_planned"],
        "completed_bo_steps": run["summary"]["n_completed_bo_steps"],
        "observed_evaluations": run["summary"]["n_observed_evaluations"],
        "final": run["summary"]["final"], "auc": run["summary"]["auc"],
        "auc_scope": run["summary"]["auc_scope"],
        "hit_metrics": run["summary"]["diversity_by_profile"],
        "gp_diagnostics": run["summary"].get("gp_diagnostics"),
        "missing_reason": run["missing_reason"], "warnings": run["warnings"],
        "files": {"complete_record": "run.json", **artifacts},
    }
    # Representatives remain in the complete run, referenced rather than repeated.
    summary["hit_metrics"] = {name: {key: value for key, value in profile.items() if key != "representatives"}
                              for name, profile in summary["hit_metrics"].items()}
    write_json(folder / "summary.json", summary)
    with io_path(folder / "evaluations.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=EVALUATION_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(evaluation_rows(run))
