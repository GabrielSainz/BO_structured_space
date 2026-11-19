"""Prefix-only metrics on recorded molecular observations. No oracle/model calls."""

import math
from itertools import combinations
from statistics import mean

import selfies as sf
from rdkit import Chem, DataStructs, rdBase
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold


QUALITY = ("best_so_far", "top_1", "top_10", "top_100")
HITS = ("unique_hits", "online_diverse_hits", "distinct_hit_scaffolds")
FINGERPRINT = {"type": "Morgan_bit", "radius": 2, "n_bits": 2048,
               "use_chirality": False, "rdkit_version": rdBase.rdkitVersion}


def finite(value):
    return (float(value) if value is not None and math.isfinite(float(value)) else None)


def molecule_info(structure):
    """Same SELFIES-first parsing convention as the existing iteration writer."""
    if structure is None:
        return {"smiles": None, "canonical_smiles": None, "mol": None}
    try:
        smiles = sf.decoder(structure)
    except Exception:
        smiles = structure
    if not smiles:
        # SELFIES returns an empty string (rather than raising) for plain SMILES.
        smiles = structure
    with rdBase.BlockLogs():
        mol = Chem.MolFromSmiles(smiles) if smiles else None
    return {"smiles": smiles, "mol": mol,
            "canonical_smiles": Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
            if mol is not None else None}


def fingerprint(mol):
    return rdFingerprintGenerator.GetMorganGenerator(
        radius=2, fpSize=2048, includeChirality=False).GetFingerprint(mol)


def distance(left, right):
    return 1.0 - DataStructs.TanimotoSimilarity(left, right)


def accepts_representative(candidate, representatives, delta):
    return all(distance(candidate, old) > delta for old in representatives)


def normalized_auc(trajectory, getter):
    """Return (value, missing_reason); no interpolation across absent states."""
    if len(trajectory) < 2 or trajectory[-1]["bo_step"] == 0:
        return None, "AUC requires a positive completed BO horizon"
    steps = [point["bo_step"] for point in trajectory]
    if steps != list(range(steps[-1] + 1)):
        return None, "AUC requires every state from step 0 through the completed horizon"
    values = [getter(point) for point in trajectory]
    if any(v is None for v in values):
        return None, "One or more prefix metric values are unavailable"
    # Mean of equal-width trapezoids avoids overflow in a large finite sum.
    return mean(a / 2 + b / 2 for a, b in zip(values, values[1:])), None


def compute_metrics(scores, structures, n_initial, step_ends, *, theta=None, delta=.7,
                    direction="maximize"):
    """step_ends maps completed BO steps (including 0) to exclusive history ends.

    Physical oracle calls are unknown. Observation indexing/step labels are derived
    from saved history boundaries, explicitly distinct from logged event records.
    """
    if direction not in {"maximize", "minimize"}:
        raise ValueError("score_direction must be maximize or minimize")
    if not isinstance(delta, (int, float)) or isinstance(delta, bool) or not math.isfinite(delta) or not 0 <= delta <= 1:
        raise ValueError("delta must be finite and in [0, 1]")
    if theta is not None and (not isinstance(theta, (int, float)) or isinstance(theta, bool) or not math.isfinite(theta)):
        raise ValueError("hit_threshold must be finite")
    if structures is not None and len(scores) != len(structures):
        raise ValueError("Original structures and scores must be aligned")
    factor = 1 if direction == "maximize" else -1
    infos = [molecule_info(s) for s in structures] if structures is not None else [
        molecule_info(None) for _ in scores]
    fps = {info["canonical_smiles"]: fingerprint(info["mol"]) for info in infos
           if info["mol"] is not None}
    initial_ids = {info["canonical_smiles"] for info in infos[:n_initial]
                   if info["canonical_smiles"] is not None}
    seen, unique, hit_ids, scaffolds = set(), {}, set(), set()
    representatives, rep_fps, evaluations, trajectory = [], [], [], []
    duplicate_count, valid_count, finite_best = 0, 0, None
    reasons = {}
    if structures is None:
        reasons["molecular_metrics"] = "No aligned original structures in source history; latent decoding is not observation recovery"
    if theta is None:
        reasons["hit_metrics"] = "No task-specific hit_threshold configured"
    ends = {end: step for step, end in step_ends.items()}
    # Empty initialization still has a well-defined empty state at step 0.
    def snapshot(step, count):
        ranked = sorted(unique, key=lambda key: -unique[key]["score"])
        top = ranked[:10]
        quality = {"best_so_far": finite_best if structures is None else
                   (unique[ranked[0]]["score"] if ranked else None)}
        for k in (1, 10, 100):
            quality[f"top_{k}"] = mean(unique[key]["score"] for key in ranked[:k]) if ranked else None
        hit_available = theta is not None and structures is not None
        profile = {key: None for key in HITS}
        if hit_available:
            profile.update(unique_hits=len(hit_ids), online_diverse_hits=len(representatives),
                           distinct_hit_scaffolds=len(scaffolds))
        profile.update(unique_hit_rate=None, approx_circles=None,
                       unique_hits_per_observation=len(hit_ids) / count if hit_available and count else None)
        missing = {"cumulative_oracle_calls": "Physical oracle-call/cache flags were not recorded",
                   "unique_hit_rate": "Physical oracle calls were not recorded",
                   "approx_circles": "Optional approximation not computed", **reasons}
        if not ranked:
            missing["top_k"] = "No valid finite-scored molecular identities in this prefix" if structures is not None else reasons["molecular_metrics"]
        if quality["best_so_far"] is None:
            missing["best_so_far"] = "No eligible finite-scored observations in this prefix"
        if len(top) < 2:
            missing["top_10_internal_diversity"] = "Fewer than two eligible unique molecules"
        return {"bo_step": step, "cumulative_oracle_calls": None,
                "cumulative_observed_evaluations": count,
                "n_unique_valid_molecules": len(seen) if structures is not None else None,
                "quality": quality, "diversity_by_profile": {"primary": profile},
                "top_10_internal_diversity": mean(distance(fps[a], fps[b]) for a, b in combinations(top, 2))
                if len(top) >= 2 else None,
                "duplicate_observation_rate": duplicate_count / valid_count if valid_count else None,
                "missing_reason": missing}

    if ends.get(0) == 0:
        trajectory.append(snapshot(0, 0))
    for index, (raw, info) in enumerate(zip(scores, infos)):
        raw = finite(raw)
        score = factor * raw if raw is not None else None
        key = info["canonical_smiles"]
        duplicate = key in seen if key is not None else None
        step = 0 if index < n_initial else next(
            (s for s, end in step_ends.items() if s > 0 and index < end), None)
        old_best = (max(v["score"] for v in unique.values()) if unique else None)
        if score is not None:
            finite_best = score if finite_best is None else max(finite_best, score)
        if key is not None:
            valid_count += 1
            duplicate_count += int(bool(duplicate))
            seen.add(key)
            if score is not None:
                if key not in unique or score > unique[key]["score"]:
                    unique[key] = {"score": score, "observation_index": index}
                if theta is not None and score >= theta and key not in hit_ids:
                    hit_ids.add(key)
                    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=info["mol"], includeChirality=False)
                    if scaffold:
                        scaffolds.add(scaffold)
                    if accepts_representative(fps[key], rep_fps, delta):
                        representatives.append({"canonical_smiles": key, "observation_index": index,
                                                "bo_step": step, "score_at_acceptance": score})
                        rep_fps.append(fps[key])
        missing = {}
        if raw is None:
            missing["raw_oracle_score"] = "Nonfinite/missing recorded score"
        if structures is None:
            missing["canonical_smiles"] = reasons["molecular_metrics"]
        elif key is None:
            missing["canonical_smiles"] = "Recorded structure could not be parsed as a nonempty molecule"
        evaluations.append({
            "observation_index": index, "bo_step": step,
            "phase": "initial" if index < n_initial else "optimization",
            "step_assignment": "inferred_from_saved_initialization_and_batch_boundaries",
            "observed_structure": structures[index] if structures is not None else None,
            "smiles": info["smiles"], "canonical_smiles": key,
            "raw_oracle_score": raw, "score_for_metrics": score,
            "valid_molecule": (key is not None) if structures is not None else None,
            "is_duplicate": duplicate if key is not None else (False if structures is not None else None),
            "actual_oracle_call": None, "cumulative_oracle_calls": None,
            "cumulative_observed_evaluations": index + 1,
            "pre_evaluation_prediction": None, "timings_seconds": None,
            "novelty_to_initial": min(distance(fps[key], fps[initial]) for initial in initial_ids)
            if key is not None and initial_ids and index >= n_initial else None,
            "realized_improvement": finite(max(0., score - old_best))
            if key is not None and score is not None and old_best is not None else None,
            "missing_reason": {**missing, "actual_oracle_call": "Physical-call/cache flags not recorded",
                               "pre_evaluation_prediction": "No recorded GP predictive moments",
                               "timings_seconds": "No recorded timing events"},
        })
        if index + 1 in ends:
            trajectory.append(snapshot(ends[index + 1], index + 1))
    # Representatives from uncompleted trailing observations must not enter final metrics.
    horizon_end = max(step_ends.values(), default=0)
    representatives = [r for r in representatives if r["observation_index"] < horizon_end]
    return evaluations, trajectory, representatives
