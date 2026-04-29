from __future__ import annotations

import copy
import random
from contextlib import contextmanager
from typing import Any

import numpy as np
import torch


TOP_K_CANDIDATES = 10
TOP_K_MEAN_OBJECTIVE_METRIC = "mean_available_unique_top_10_candidate_objective"
TOP_K_COUNT_METRIC = "available_unique_top_10_candidate_count"
_STATE_ATTRIBUTE_NAME_MARKERS = (
    "budget",
    "cache",
    "call",
    "count",
    "counter",
    "evaluat",
    "history",
)
_COMMON_STATE_ATTRIBUTE_NAMES = (
    "evaluation_budget",
    "_evaluation_budget",
    "num_evaluations",
    "_num_evaluations",
    "n_evaluations",
    "_n_evaluations",
    "num_calls",
    "_num_calls",
    "n_calls",
    "_n_calls",
    "call_count",
    "_call_count",
    "counter",
    "_counter",
    "cache",
    "_cache",
    "history",
    "_history",
)
_UNCOPYABLE = object()


def resolve_diagnostic_evaluator(black_box: Any) -> Any:
    for attribute_name in ("diagnostic_function", "raw_function"):
        evaluator = getattr(black_box, attribute_name, None)
        if callable(evaluator):
            return evaluator

    for attribute_name in ("_black_box", "function"):
        evaluator = getattr(black_box, attribute_name, None)
        if callable(evaluator):
            return evaluator

    return None


def evaluate_candidate_objectives(
    black_box: Any,
    unit_latents: np.ndarray,
) -> np.ndarray:
    unit_latents = np.asarray(unit_latents, dtype=float)
    if unit_latents.size == 0:
        return np.zeros((0,), dtype=float)

    evaluator = resolve_diagnostic_evaluator(black_box)
    if evaluator is None:
        return np.full((unit_latents.shape[0],), np.nan, dtype=float)

    return np.asarray(
        _call_preserving_diagnostic_state(black_box, evaluator, unit_latents),
        dtype=float,
    ).reshape(-1)


@contextmanager
def preserve_rng_state():
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    cuda_states = (
        torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    )

    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.random.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


def _call_preserving_diagnostic_state(
    black_box: Any,
    evaluator: Any,
    unit_latents: np.ndarray,
) -> Any:
    state_snapshots = _snapshot_diagnostic_state(black_box, evaluator)
    try:
        with preserve_rng_state():
            return evaluator(unit_latents)
    finally:
        _restore_diagnostic_state(state_snapshots)


def build_top_candidate_diagnostics(
    black_box: Any,
    ranked_candidates: list[dict[str, Any]],
    *,
    top_k: int = TOP_K_CANDIDATES,
    evaluate_objectives: bool = True,
) -> dict[str, Any]:
    top_candidates = [
        {
            **candidate,
            "rank": int(candidate.get("rank", rank)),
        }
        for rank, candidate in enumerate(ranked_candidates[:top_k], start=1)
    ]

    if not top_candidates:
        return {
            "top_k": int(top_k),
            TOP_K_COUNT_METRIC: 0,
            TOP_K_MEAN_OBJECTIVE_METRIC: np.nan,
            "top_candidates": [],
        }

    if not evaluate_objectives:
        return {
            "top_k": int(top_k),
            TOP_K_COUNT_METRIC: int(len(top_candidates)),
            TOP_K_MEAN_OBJECTIVE_METRIC: np.nan,
            "top_candidates": top_candidates,
        }

    unit_latents = np.asarray(
        [candidate["unit_latent"] for candidate in top_candidates],
        dtype=float,
    )
    objective_values = evaluate_candidate_objectives(black_box, unit_latents)
    if objective_values.size < len(top_candidates):
        objective_values = np.pad(
            objective_values,
            (0, len(top_candidates) - objective_values.size),
            constant_values=np.nan,
        )
    objective_values = objective_values[: len(top_candidates)]
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
        TOP_K_COUNT_METRIC: int(len(top_candidates)),
        TOP_K_MEAN_OBJECTIVE_METRIC: mean_objective,
        "top_candidates": enriched_candidates,
    }


def _snapshot_diagnostic_state(
    black_box: Any,
    evaluator: Any,
) -> list[tuple[Any, str, Any]]:
    snapshots: list[tuple[Any, str, Any]] = []
    seen_targets: set[int] = set()

    for target in _iter_state_targets(black_box, evaluator):
        target_id = id(target)
        if target_id in seen_targets:
            continue
        seen_targets.add(target_id)

        for attribute_name in _iter_state_attribute_names(target):
            try:
                value = getattr(target, attribute_name)
            except Exception:
                continue
            if callable(value):
                continue

            copied_value = _copy_state_value(value)
            if copied_value is _UNCOPYABLE:
                continue
            snapshots.append((target, attribute_name, copied_value))

    return snapshots


def _restore_diagnostic_state(
    snapshots: list[tuple[Any, str, Any]],
) -> None:
    for target, attribute_name, copied_value in reversed(snapshots):
        try:
            setattr(target, attribute_name, copied_value)
        except Exception:
            continue


def _iter_state_targets(black_box: Any, evaluator: Any) -> list[Any]:
    targets = [
        black_box,
        evaluator,
        getattr(evaluator, "__self__", None),
        getattr(black_box, "_black_box", None),
        getattr(black_box, "function", None),
        getattr(black_box, "raw_function", None),
        getattr(black_box, "diagnostic_function", None),
    ]
    return [target for target in targets if target is not None]


def _iter_state_attribute_names(target: Any) -> set[str]:
    attribute_names = {
        name for name in _COMMON_STATE_ATTRIBUTE_NAMES if hasattr(target, name)
    }
    try:
        attribute_names.update(
            name
            for name in vars(target)
            if any(marker in name.lower() for marker in _STATE_ATTRIBUTE_NAME_MARKERS)
        )
    except TypeError:
        pass
    return attribute_names


def _copy_state_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.copy()
    if torch.is_tensor(value):
        return value.detach().clone()
    try:
        return copy.deepcopy(value)
    except Exception:
        return _UNCOPYABLE
