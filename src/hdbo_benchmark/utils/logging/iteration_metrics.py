from __future__ import annotations

from typing import Any

import numpy as np
import selfies as sf
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

from hdbo_benchmark.utils.logging.candidate_diagnostics import (
    TOP_K_COUNT_METRIC,
    TOP_K_MEAN_OBJECTIVE_METRIC,
    build_top_candidate_diagnostics,
)


def build_iteration_metric_artifacts(
    solver: Any,
    history_x: np.ndarray,
    history_y: np.ndarray,
    initial_history_size: int,
    completed_iterations: int,
    evaluate_top_candidate_objectives: bool = True,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    history_x = np.asarray(history_x, dtype=float)
    history_y = np.asarray(history_y, dtype=float).reshape(-1)
    batch_size = max(1, int(getattr(solver, "batch_size", 1)))

    n_available_new_points = max(history_x.shape[0] - initial_history_size, 0)
    n_tracked_new_points = min(
        n_available_new_points,
        max(completed_iterations, 0) * batch_size,
    )
    n_tracked_iterations = min(
        max(completed_iterations, 0),
        int(np.ceil(n_tracked_new_points / batch_size)) if batch_size > 0 else 0,
    )
    tracked_history_size = initial_history_size + n_tracked_new_points

    tracked_x = history_x[:tracked_history_size]
    tracked_y = history_y[:tracked_history_size]
    decoded_structures = _decode_unit_latents_to_structures(solver, tracked_x)
    molecule_infos = [_structure_to_molecule_info(structure) for structure in decoded_structures]

    series = {
        "bo_iteration": [],
        "best_so_far_objective": [],
        "mean_selected_nearest_previous_tanimoto_distance": [],
        "cumulative_distinct_decoded_molecules": [],
        "new_distinct_decoded_molecules": [],
    }
    sampling_metric_names = [
        "sample_unique_decoded_molecules_in_iteration",
        "sample_new_distinct_decoded_molecules",
        "sample_cumulative_distinct_decoded_molecules",
        "sample_unique_decoded_molecules_not_in_observed_history",
    ]
    sampling_metric_history = getattr(solver, "iteration_sampling_metrics_history", [])
    top_candidate_metric_names = [
        TOP_K_MEAN_OBJECTIVE_METRIC,
        TOP_K_COUNT_METRIC,
    ]
    top_candidate_metric_history = getattr(
        solver,
        "iteration_candidate_diagnostics_history",
        [],
    )
    for metric_name in sampling_metric_names:
        if sampling_metric_history:
            series[metric_name] = []
    for metric_name in top_candidate_metric_names:
        if top_candidate_metric_history:
            series[metric_name] = []
    iteration_records: list[dict[str, Any]] = []

    for iteration_offset in range(n_tracked_iterations):
        iteration_start = initial_history_size + iteration_offset * batch_size
        iteration_stop = min(iteration_start + batch_size, tracked_history_size)
        selected_infos = molecule_infos[iteration_start:iteration_stop]
        selected_structures = decoded_structures[iteration_start:iteration_stop]
        selected_objective_values = tracked_y[iteration_start:iteration_stop]

        previous_valid_fingerprints = [
            info["fingerprint"]
            for info in molecule_infos[initial_history_size:iteration_start]
            if info["fingerprint"] is not None
        ]
        selected_nearest_distances = [
            _nearest_previous_tanimoto_distance(info["fingerprint"], previous_valid_fingerprints)
            for info in selected_infos
        ]
        mean_nearest_distance = _nanmean(selected_nearest_distances)

        finite_history = tracked_y[:iteration_stop][np.isfinite(tracked_y[:iteration_stop])]
        best_so_far = float(finite_history.max()) if finite_history.size > 0 else np.nan

        previous_smiles = {
            info["canonical_smiles"]
            for info in molecule_infos[initial_history_size:iteration_start]
            if info["canonical_smiles"] is not None
        }
        cumulative_smiles = {
            info["canonical_smiles"]
            for info in molecule_infos[initial_history_size:iteration_stop]
            if info["canonical_smiles"] is not None
        }
        seen_in_iteration: set[str] = set()
        n_new_distinct = 0
        for info in selected_infos:
            canonical_smiles = info["canonical_smiles"]
            if canonical_smiles is None:
                continue
            if canonical_smiles in previous_smiles or canonical_smiles in seen_in_iteration:
                continue
            seen_in_iteration.add(canonical_smiles)
            n_new_distinct += 1

        series["bo_iteration"].append(iteration_offset + 1)
        series["best_so_far_objective"].append(best_so_far)
        series["mean_selected_nearest_previous_tanimoto_distance"].append(
            mean_nearest_distance
        )
        series["cumulative_distinct_decoded_molecules"].append(len(cumulative_smiles))
        series["new_distinct_decoded_molecules"].append(n_new_distinct)

        sampling_metrics_for_iteration = (
            sampling_metric_history[iteration_offset]
            if iteration_offset < len(sampling_metric_history)
            else {}
        )
        for metric_name in sampling_metric_names:
            if metric_name not in series:
                continue
            series[metric_name].append(
                sampling_metrics_for_iteration.get(metric_name, np.nan)
            )
        top_candidate_metrics_for_iteration = (
            top_candidate_metric_history[iteration_offset]
            if iteration_offset < len(top_candidate_metric_history)
            else {}
        )
        if (
            evaluate_top_candidate_objectives
            and top_candidate_metrics_for_iteration.get("top_candidates")
        ):
            top_candidate_metrics_for_iteration = build_top_candidate_diagnostics(
                solver.black_box,
                top_candidate_metrics_for_iteration["top_candidates"],
                top_k=int(
                    top_candidate_metrics_for_iteration.get(
                        "top_k",
                        len(top_candidate_metrics_for_iteration["top_candidates"]),
                    )
                ),
                evaluate_objectives=True,
            )
        for metric_name in top_candidate_metric_names:
            if metric_name not in series:
                continue
            series[metric_name].append(
                top_candidate_metrics_for_iteration.get(metric_name, np.nan)
            )

        iteration_record = {
            "iteration": iteration_offset + 1,
            "num_selected": iteration_stop - iteration_start,
            "history_size_after_iteration": iteration_stop,
            "selected_decoded_structures": selected_structures,
            "selected_smiles": [
                info["canonical_smiles"] for info in selected_infos
            ],
            "selected_valid_mask": [
                info["canonical_smiles"] is not None for info in selected_infos
            ],
            "selected_objective_values": _serialize_vector(selected_objective_values),
            "selected_nearest_previous_tanimoto_distances": _serialize_vector(
                selected_nearest_distances
            ),
            "mean_selected_nearest_previous_tanimoto_distance": _serialize_number(
                mean_nearest_distance
            ),
            "best_so_far_objective": _serialize_number(best_so_far),
            "cumulative_distinct_decoded_molecules": len(cumulative_smiles),
            "new_distinct_decoded_molecules": n_new_distinct,
        }
        for metric_name in sampling_metric_names:
            if metric_name not in series:
                continue
            iteration_record[metric_name] = _serialize_number(
                sampling_metrics_for_iteration.get(metric_name, np.nan)
            )
        for metric_name in top_candidate_metric_names:
            if metric_name not in series:
                continue
            iteration_record[metric_name] = _serialize_number(
                top_candidate_metrics_for_iteration.get(metric_name, np.nan)
            )
        if "top_k" in top_candidate_metrics_for_iteration:
            iteration_record["top_k_candidate_diagnostics_limit"] = int(
                top_candidate_metrics_for_iteration["top_k"]
            )
        if "top_candidates" in top_candidate_metrics_for_iteration:
            iteration_record["top_candidate_diagnostics"] = [
                _serialize_candidate_diagnostic(candidate)
                for candidate in top_candidate_metrics_for_iteration["top_candidates"]
            ]
        iteration_records.append(iteration_record)

    series_arrays = {
        name: np.asarray(values, dtype=float if name != "bo_iteration" else int)
        for name, values in series.items()
    }
    json_payload = {
        "initial_history_size": int(initial_history_size),
        "batch_size": batch_size,
        "completed_iterations": int(completed_iterations),
        "tracked_iterations": int(n_tracked_iterations),
        "tracked_history_size": int(tracked_history_size),
        "metric_series": {
            name: _serialize_vector(values)
            for name, values in series_arrays.items()
        },
        "iterations": iteration_records,
    }
    return json_payload, series_arrays


def _decode_unit_latents_to_structures(solver: Any, unit_latents: np.ndarray) -> list[str]:
    if unit_latents.size == 0:
        return []
    if not hasattr(solver, "vae") or not hasattr(solver.vae, "decode_to_string_array"):
        raise ValueError("Solver does not expose a VAE decoder for iteration metrics.")

    if hasattr(solver, "_unit_to_latent"):
        latent_values = solver._unit_to_latent(unit_latents)
    else:
        bounds = getattr(solver, "bounds", None) or getattr(solver, "vae_bounds", None)
        if bounds is None:
            latent_values = unit_latents
        else:
            lower, upper = bounds
            latent_values = unit_latents * (upper - lower) + lower

    decoded = solver.vae.decode_to_string_array(latent_values)
    return _normalize_structure_array(decoded)


def _normalize_structure_array(structures: np.ndarray) -> list[str]:
    normalized: list[str] = []
    for structure in np.asarray(structures, dtype=object):
        if isinstance(structure, str):
            normalized.append(structure)
            continue

        tokens = [str(token) for token in np.asarray(structure, dtype=object).tolist()]
        cleaned_tokens = [
            token for token in tokens if token not in {"<pad>", "<cls>", "<eos>"}
        ]
        normalized.append("".join(cleaned_tokens))
    return normalized


def _structure_to_molecule_info(structure: str) -> dict[str, Any]:
    mol = _structure_to_mol(structure)
    if mol is None:
        return {
            "canonical_smiles": None,
            "fingerprint": None,
        }

    return {
        "canonical_smiles": Chem.MolToSmiles(mol, canonical=True),
        "fingerprint": _mol_to_count_fingerprint(mol),
    }


def _structure_to_mol(structure: str) -> Any:
    candidate_smiles: str | None = None
    try:
        candidate_smiles = sf.decoder(structure)
    except Exception:
        candidate_smiles = structure

    if not candidate_smiles:
        return None
    return Chem.MolFromSmiles(candidate_smiles)


def _mol_to_count_fingerprint(mol: Any) -> np.ndarray:
    fingerprint = rdMolDescriptors.GetMorganFingerprint(
        mol,
        radius=3,
        useCounts=True,
    )
    folded = np.zeros(2048, dtype=float)
    for key, value in fingerprint.GetNonzeroElements().items():
        folded[key % 2048] += value
    return folded


def _nearest_previous_tanimoto_distance(
    fingerprint: np.ndarray | None,
    previous_fingerprints: list[np.ndarray],
) -> float:
    if fingerprint is None or not previous_fingerprints:
        return np.nan

    reference = np.vstack(previous_fingerprints)
    intersection = reference @ fingerprint
    fingerprint_norm = float(np.dot(fingerprint, fingerprint))
    reference_norm = np.sum(reference * reference, axis=1)
    denominator = reference_norm + fingerprint_norm - intersection
    similarities = np.divide(
        intersection,
        denominator,
        out=np.zeros_like(intersection, dtype=float),
        where=denominator > 0,
    )
    return float(1.0 - similarities.max())


def _nanmean(values: list[float]) -> float:
    if not values:
        return np.nan
    array = np.asarray(values, dtype=float)
    finite_values = array[np.isfinite(array)]
    if finite_values.size == 0:
        return np.nan
    return float(finite_values.mean())


def _serialize_vector(values: np.ndarray | list[Any]) -> list[Any]:
    return [_serialize_number(value) for value in np.asarray(values).tolist()]


def _serialize_number(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value):
            return None
        return float(value)
    return value


def _serialize_candidate_diagnostic(candidate: dict[str, Any]) -> dict[str, Any]:
    serialized_candidate: dict[str, Any] = {}
    for key, value in candidate.items():
        if isinstance(value, dict):
            serialized_candidate[key] = {
                nested_key: _serialize_number(nested_value)
                for nested_key, nested_value in value.items()
            }
            continue
        if isinstance(value, (list, tuple, np.ndarray)):
            serialized_candidate[key] = _serialize_vector(value)
            continue
        serialized_candidate[key] = _serialize_number(value)
    return serialized_candidate
