"""Runner-side, pre-oracle COWBOYS diagnostics; never refit or evaluate objectives.

The original next_candidate method runs first and its return value is passed
through unchanged. Diagnostic failures are recorded rather than failing a run.
No solver implementation or acquisition is modified by this recorder.
"""

import math
from functools import wraps

import numpy as np
import torch

from .candidate_diagnostics import preserve_rng_state


def _number(value):
    value = float(value)
    return value if math.isfinite(value) else None


def analytic_expected_improvement(mean, std, incumbent):
    """Gaussian latent-posterior EI; distinct from the solver's MC log-EI."""
    if any(value is None or not math.isfinite(value) for value in (mean, std, incumbent)):
        return None
    if std < 0:
        return None
    gain = _number(mean - incumbent)
    if gain is None:
        return None
    if std == 0:
        return max(0., gain)
    z = gain / std
    cdf = .5 * math.erfc(-z / math.sqrt(2.))
    density = math.exp(-.5 * z * z) / math.sqrt(2. * math.pi)
    return _number(max(0., gain * cdf + std * density))


def _capture(solver, candidate, initial_history_size):
    x, y = solver.get_history_as_arrays()
    history_size = len(x)
    record = {
        "observation_index": history_size,
        "bo_step": history_size - initial_history_size + 1,
        "captured": "after_candidate_selection_before_oracle_evaluation",
        "n_training_observations": history_size,
        "canonical_smiles": None,
        "mean": None, "latent_std": None, "predictive_std": None,
        "analytic_expected_improvement": None, "incumbent_for_acquisition": None,
        "missing_reason": {},
    }
    try:
        with preserve_rng_state(), torch.no_grad():
            if np.asarray(candidate).shape[0] != 1:
                raise ValueError("The COWBOYS recorder supports batch_size=1 only")
            # Reuse the original selected structure when it is in the ranked cache.
            # Match latent coordinates, not rank or fingerprint equality.
            diagnostics = getattr(solver, "iteration_candidate_diagnostics_history", [])
            candidates = diagnostics[-1].get("top_candidates", []) if diagnostics else []
            matches = [item for item in candidates if np.array_equal(
                np.asarray(item.get("unit_latent")), np.asarray(candidate)[0])]
            if len(matches) == 1:
                structure = matches[0]["structure"]
            else:
                decoded = solver.vae.decode_to_string_array(solver._unit_to_latent(np.asarray(candidate)))
                structure = solver._structure_strings(decoded)[0]
            record["canonical_smiles"] = solver._canonical_smiles(structure)
            features = solver.selfies_list_to_fingerprint_tensors([structure])
            model = solver.gp_model_of_objective
            if len(model.train_inputs[0]) != history_size:
                raise ValueError("GP training size disagrees with pre-evaluation history")
            # BoTorch posterior untransforms outcomes to the recorded objective scale.
            posterior = model.posterior(features, observation_noise=False)
            mean = _number(posterior.mean.reshape(-1)[0])
            variance = _number(posterior.variance.reshape(-1)[0])
            record["mean"] = mean
            record["latent_std"] = math.sqrt(variance) if variance is not None and variance >= 0 else None
            processed_y = np.asarray(y, dtype=float).copy()
            processed_y[np.isnan(processed_y)] = solver.penalize_nans_with
            incumbent = _number(processed_y.max())
            record["incumbent_for_acquisition"] = incumbent
            record["analytic_expected_improvement"] = analytic_expected_improvement(
                mean, record["latent_std"], incumbent)
            try:
                noisy = model.posterior(features, observation_noise=True)
                noisy_variance = _number(noisy.variance.reshape(-1)[0])
                record["predictive_std"] = (math.sqrt(noisy_variance)
                    if noisy_variance is not None and noisy_variance >= 0 else None)
            except Exception as exc:
                record["missing_reason"]["predictive_std"] = f"{type(exc).__name__}: {exc}"
            for key in ("mean", "latent_std", "predictive_std", "analytic_expected_improvement"):
                if record[key] is None and key not in record["missing_reason"]:
                    record["missing_reason"][key] = "Posterior value unavailable or nonfinite"
    except Exception as exc:
        record["missing_reason"]["prediction"] = f"{type(exc).__name__}: {exc}"
    return record


def attach_gp_prediction_recorder(solver, initial_history_size):
    """Attach once, at the runner boundary, before step() invokes the oracle."""
    if getattr(solver, "pre_evaluation_prediction_history", None) is not None:
        return
    records = []
    solver.pre_evaluation_prediction_history = records
    original = solver.next_candidate

    @wraps(original)
    def with_prediction(*args, **kwargs):
        candidate = original(*args, **kwargs)
        try:
            records.append(_capture(solver, candidate, initial_history_size))
        except Exception as exc:
            # Even malformed metadata/diagnostic allocation must not alter selection.
            records.append({"captured": "after_candidate_selection_before_oracle_evaluation",
                            "missing_reason": {"prediction": f"{type(exc).__name__}: {exc}"}})
        return candidate

    solver.next_candidate = with_prediction


PREDICTION_CONVENTIONS = {
    "timing": "after selection, before oracle evaluation and history update",
    "score_scale": "recorded oracle objective; BoTorch outcome transform undone",
    "latent_std": "posterior std of latent objective, excluding observation noise",
    "predictive_std": "posterior predictive std including GP likelihood observation noise",
    "analytic_expected_improvement": "Gaussian latent-posterior EI relative to incumbent_for_acquisition; not the MC qLogEI acquisition",
    "rng_policy": "Python, NumPy, Torch and existing CUDA RNG states restored after diagnostic calls",
}
