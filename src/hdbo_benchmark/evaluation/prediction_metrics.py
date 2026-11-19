"""Offline joining and diagnostics for genuinely pre-evaluation GP records."""

import math
from statistics import mean


PREDICTION_FIELDS = ("mean", "latent_std", "predictive_std", "analytic_expected_improvement",
                     "incumbent_for_acquisition", "n_training_observations", "captured", "missing_reason")
Z90 = 1.6448536269514722


def join_predictions(evaluations, sidecar, warnings):
    records = sidecar.get("predictions", []) if sidecar else []
    matched_count = 0
    for evaluation in evaluations:
        if evaluation["phase"] != "optimization":
            evaluation["missing_reason"]["pre_evaluation_prediction"] = "Initial design has no fitted BO GP prediction"
            continue
        matches = [record for record in records if record.get("observation_index") == evaluation["observation_index"]
                   and record.get("canonical_smiles") == evaluation["canonical_smiles"]
                   and evaluation["canonical_smiles"] is not None]
        if not matches:
            continue
        if len(matches) != 1:
            warnings.append(f"Ambiguous GP records for observation {evaluation['observation_index']}; predictions not joined")
            continue
        record = matches[0]
        if (record.get("captured") != "after_candidate_selection_before_oracle_evaluation"
                or record.get("n_training_observations") != evaluation["observation_index"]
                or record.get("bo_step") != evaluation["bo_step"]):
            warnings.append(f"Invalid pre-evaluation GP boundary at observation {evaluation['observation_index']}")
            continue
        prediction = {key: record.get(key) for key in PREDICTION_FIELDS}
        for key in ("mean", "latent_std", "predictive_std", "analytic_expected_improvement"):
            value = prediction[key]
            if value is not None and (not isinstance(value, (int, float)) or not math.isfinite(value)
                                       or (key != "mean" and value < 0)):
                prediction[key] = None
                prediction["missing_reason"] = {**(prediction.get("missing_reason") or {}), key: "Invalid/nonfinite recorded prediction"}
        evaluation["pre_evaluation_prediction"] = prediction
        evaluation["missing_reason"].pop("pre_evaluation_prediction", None)
        score, mu, sigma = evaluation["raw_oracle_score"], prediction["mean"], prediction["predictive_std"]
        residual = score - mu if score is not None and mu is not None else None
        evaluation["prediction_residual"] = residual if residual is None or math.isfinite(residual) else None
        if residual is not None and not math.isfinite(residual):
            evaluation["missing_reason"]["prediction_residual"] = "Nonfinite difference between recorded mean and observed score"
        if evaluation["prediction_residual"] is not None and sigma is not None and sigma > 0:
            standardized = residual / sigma
            nll = math.log(sigma) + .5 * math.log(2 * math.pi) + .5 * standardized * standardized
            evaluation["predictive_gaussian_nll"] = nll if math.isfinite(nll) else None
            evaluation["covered_by_predictive_90pct_interval"] = abs(residual) <= Z90 * sigma
            if not math.isfinite(nll):
                evaluation["missing_reason"]["predictive_gaussian_nll"] = "Nonfinite Gaussian NLL from recorded variance and residual"
        elif sigma is not None and sigma <= 0:
            evaluation["missing_reason"]["predictive_gaussian_nll"] = "Gaussian NLL requires positive predictive standard deviation"
        matched_count += 1
    return matched_count


def prediction_summary(evaluations):
    scored = [e for e in evaluations if e.get("prediction_residual") is not None]
    likelihood = [e for e in scored if e.get("predictive_gaussian_nll") is not None]
    intervals = [e for e in scored if e.get("covered_by_predictive_90pct_interval") is not None]
    scale = max((abs(e["prediction_residual"]) for e in scored), default=0.)
    rmse = scale * math.sqrt(mean((e["prediction_residual"] / scale) ** 2 for e in scored)) if scale else (0. if scored else None)
    return {
        "n_recorded_predictions": sum(e.get("pre_evaluation_prediction") is not None for e in evaluations),
        "n_scored_predictions": len(scored), "n_predictive_intervals": len(intervals),
        "prequential_rmse": rmse,
        "prequential_gaussian_nll": mean(e["predictive_gaussian_nll"] for e in likelihood) if likelihood else None,
        "coverage_90pct": mean(int(e["covered_by_predictive_90pct_interval"]) for e in intervals) if intervals else None,
        "interpretation": "Pre-evaluation diagnostics on adaptively selected molecules; not an independent test set",
        "missing_reason": (None if scored else "No matched, finite pre-evaluation GP means and observed scores"),
    }
