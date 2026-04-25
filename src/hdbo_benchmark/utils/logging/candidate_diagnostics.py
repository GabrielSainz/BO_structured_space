from __future__ import annotations

from typing import Any

import numpy as np


TOP_K_CANDIDATES = 10
TOP_K_MEAN_OBJECTIVE_METRIC = "mean_available_unique_top_10_candidate_objective"
TOP_K_COUNT_METRIC = "available_unique_top_10_candidate_count"


def resolve_diagnostic_evaluator(black_box: Any) -> Any:
    for attribute_name in ("raw_function", "diagnostic_function"):
        evaluator = getattr(black_box, attribute_name, None)
        if callable(evaluator):
            return evaluator
    return black_box


def evaluate_candidate_objectives(
    black_box: Any,
    unit_latents: np.ndarray,
) -> np.ndarray:
    unit_latents = np.asarray(unit_latents, dtype=float)
    if unit_latents.size == 0:
        return np.zeros((0,), dtype=float)

    evaluator = resolve_diagnostic_evaluator(black_box)
    return np.asarray(evaluator(unit_latents), dtype=float).reshape(-1)


def build_top_candidate_diagnostics(
    black_box: Any,
    ranked_candidates: list[dict[str, Any]],
    *,
    top_k: int = TOP_K_CANDIDATES,
) -> dict[str, Any]:
    top_candidates = [candidate.copy() for candidate in ranked_candidates[:top_k]]

    if not top_candidates:
        return {
            "top_k": int(top_k),
            TOP_K_COUNT_METRIC: 0,
            TOP_K_MEAN_OBJECTIVE_METRIC: np.nan,
            "top_candidates": [],
        }

    unit_latents = np.asarray(
        [candidate["unit_latent"] for candidate in top_candidates],
        dtype=float,
    )
    objective_values = evaluate_candidate_objectives(black_box, unit_latents)
    finite_values = objective_values[np.isfinite(objective_values)]
    mean_objective = (
        float(finite_values.mean()) if finite_values.size > 0 else np.nan
    )

    enriched_candidates: list[dict[str, Any]] = []
    for rank, (candidate, objective_value) in enumerate(
        zip(top_candidates, objective_values),
        start=1,
    ):
        enriched_candidates.append(
            {
                **candidate,
                "rank": int(rank),
                "objective_value": float(objective_value),
            }
        )

    return {
        "top_k": int(top_k),
        TOP_K_COUNT_METRIC: int(len(enriched_candidates)),
        TOP_K_MEAN_OBJECTIVE_METRIC: mean_objective,
        "top_candidates": enriched_candidates,
    }
