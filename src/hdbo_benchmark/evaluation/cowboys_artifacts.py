"""Read the current COWBOYS checkpoint family without importing optimization code."""

from pathlib import Path

import numpy as np

from .metric_io import clean, digest_file, display_path, io_path, read_json


def _companion(status_path, name, suffix):
    expected = status_path.with_name(status_path.name.removesuffix("_status.json") + suffix)
    if name is not None:
        # Writer records basenames; never follow external paths from a status record.
        if Path(name).name != name or name != expected.name:
            raise ValueError(f"Unexpected checkpoint companion name: {name}")
    return expected


def load_cowboys(status_path: Path):
    status = read_json(status_path)
    if not isinstance(status, dict) or not isinstance(status.get("solver_configuration", {}), dict):
        raise ValueError("Status and solver_configuration must be JSON objects")
    if status.get("solver_name") != "cowboys":
        raise ValueError("This adapter supports solver_name=cowboys only")
    if not isinstance(status.get("function_name"), str) or not isinstance(status.get("seed"), int):
        raise ValueError("Task name and integer seed must be recorded in status JSON")
    sources = [status_path]
    warnings, scores, structures, iterations = [], None, None, None
    history_path = _companion(status_path, status.get("history_file"), "_history.npz")
    if io_path(history_path).exists():
        try:
            # No pickle: do not unpickle historical models or arbitrary object arrays.
            with np.load(io_path(history_path), allow_pickle=False) as history:
                if "y" in history:
                    y = history["y"]
                    if y.ndim > 2 or (y.ndim == 2 and y.shape[1] != 1):
                        raise ValueError("Only scalar objective histories are supported")
                    scores = y.reshape(-1).astype(float).tolist()
                if "structures" in history:
                    try:
                        observed = history["structures"]
                        if observed.ndim != 1 or observed.dtype.kind not in "US":
                            raise ValueError("Expected a one-dimensional string structure array")
                        structures = observed.astype(str).tolist()
                    except ValueError as exc:
                        warnings.append(f"Original structures unavailable: {exc}")
            sources.append(history_path)
        except (OSError, ValueError) as exc:
            warnings.append(f"History unreadable: {exc}")
    elif status.get("history_file"):
        warnings.append("Status references a missing history checkpoint")
    if structures is not None and (scores is None or len(structures) != len(scores)):
        warnings.append("Structure/score length mismatch; molecular metrics disabled")
        structures = None
    iteration_path = _companion(status_path, status.get("iteration_results_file"), "_iteration_results.json")
    if io_path(iteration_path).exists():
        try:
            iterations = read_json(iteration_path)
            if not isinstance(iterations, dict):
                raise ValueError("Iteration JSON must be an object")
            if any(iterations.get(key) != status.get(key) for key in ("solver_name", "function_name", "seed")):
                raise ValueError("Iteration metadata disagrees with status")
            sources.append(iteration_path)
        except (OSError, ValueError) as exc:
            iterations = None
            warnings.append(f"Iteration JSON unreadable: {exc}")
    elif status.get("iteration_results_file"):
        warnings.append("Status references a missing iteration JSON")
    best_path = status_path.with_name(status_path.name.removesuffix("_status.json") + ".npy")
    if io_path(best_path).exists():
        sources.append(best_path)
    predictions = None
    predictions_path = _companion(status_path, status.get("predictions_file"), "_predictions.json")
    if io_path(predictions_path).exists():
        try:
            predictions = read_json(predictions_path)
            if not isinstance(predictions, dict) or any(
                predictions.get(key) != status.get(key) for key in ("solver_name", "function_name", "seed")
            ) or not isinstance(predictions.get("predictions"), list) or not all(
                isinstance(record, dict) for record in predictions.get("predictions", [])
            ):
                raise ValueError("Prediction sidecar metadata disagrees with status")
            sources.append(predictions_path)
        except (ValueError, OSError) as exc:
            predictions = None
            warnings.append(f"GP predictions unreadable: {exc}")
    elif status.get("predictions_file"):
        warnings.append("Status references a missing GP prediction sidecar")
    return {"status": status, "scores": scores, "structures": structures,
            "iterations": clean(iterations), "predictions": clean(predictions), "warnings": warnings,
            "sources": [{"path": display_path(p), "sha256": digest_file(p)} for p in sources]}
