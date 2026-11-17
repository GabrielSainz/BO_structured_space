"""Small, offline objective audit; no optimization, training or model downloads.

Run from the repository root with the research environment activated:
python scripts/diagnostics/check_pmo.py --output docs/verification/pmo_before.json

This compares scalar poli calls with uncaught TDC evaluators. A finite zero from
TDC's public API alone is not proof that the underlying evaluator worked.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import random
import time
import traceback
from pathlib import Path
from unittest.mock import patch

import numpy as np
import selfies as sf
from rdkit import Chem
from rdkit.Chem import Descriptors, QED

from hdbo_benchmark.benchmarks.load_problems import load_problem, tokenize_selfies
from hdbo_benchmark.utils.experiments.load_metadata_for_vaes import (
    load_alphabet_for_pmo,
    load_sequence_length_for_pmo,
)
from hdbo_benchmark.utils.tdc_compat import (
    expose_tdc_evaluation_errors,
    patch_broken_tdc_chem_utils,
)
from poli.benchmarks import PMOBenchmark


PROBES = {
    "ethanol": "CCO",
    "aspirin": "CC(=O)Oc1ccccc1C(=O)O",
    "poli_example": "CC(C)(C)[C@H]1CCc2c(sc(NC(=O)COc3ccc(Cl)cc3)c2C(N)=O)C1",
    "valsartan_motif": "CN(C=O)Cc1ccc(c2ccccc2)cc1",
    "valsartan": "CCCCC(=O)N(Cc1ccc(-c2ccccc2-c2nnn[nH]2)cc1)[C@@H](C(C)C)C(=O)O",
}


def error_record(exc):
    return {"type": type(exc).__name__, "message": str(exc)}


def json_ready(value):
    """Keep evidence valid JSON; nonfinite values have a separate finite flag."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_ready(item) for item in value]
    return value


def evaluate(function, x):
    try:
        y = np.asarray(function(x), dtype=float)
        return {"shape": list(y.shape), "values": y.tolist(), "finite": bool(np.isfinite(y).all())}
    except Exception as exc:
        return {"error": error_record(exc)}


def audit_problem(name):
    import poli

    result = {"name": name}
    np.random.seed(1)
    random.seed(1)
    try:
        problem = load_problem(name, max_iter=100, set_observer=False)
        x, y = problem.data_package.supervised_data
        result["initialization"] = {
            "x_shape": list(x.shape), "y_shape": list(y.shape),
            "y": np.asarray(y).tolist(), "finite": bool(np.isfinite(y).all()),
            "valid_molecules": sum(Chem.MolFromSmiles(sf.decoder("".join(row))) is not None for row in x),
            "evaluations": problem.black_box.num_evaluations,
            "evaluation_budget": problem.black_box.evaluation_budget,
        }
    except Exception as exc:
        result["initialization"] = {"error": error_record(exc), "traceback": traceback.format_exc()}

    try:
        smiles_problem = poli.create(name=name, string_representation="SMILES")
        selfies_problem = poli.create(
            name=name, string_representation="SELFIES",
            alphabet=load_alphabet_for_pmo(),
            max_sequence_length=load_sequence_length_for_pmo(),
        )
    except Exception as exc:
        result["creation_error"] = error_record(exc)
        return result

    f = smiles_problem.black_box
    expose_tdc_evaluation_errors(f)
    expose_tdc_evaluation_errors(selfies_problem.black_box)
    oracle = getattr(getattr(f, "inner_function", None), "oracle", None)
    result["oracle_name"] = getattr(oracle, "name", None)
    result["probes"] = {}
    for label, smiles in PROBES.items():
        encoded = sf.encoder(smiles)
        tokens = tokenize_selfies(encoded, load_sequence_length_for_pmo())
        entry = {
            "smiles": evaluate(f, np.array([[smiles]])),
            "selfies_string": evaluate(selfies_problem.black_box, np.array([[encoded]])),
            "selfies_tokens": evaluate(selfies_problem.black_box, np.array([tokens])),
        }
        if oracle is not None:
            entry["raw_evaluator"] = evaluate(oracle.evaluator_func, smiles)
            entry["tdc_list"] = evaluate(oracle, [smiles])
        elif name == "rdkit_logp":
            entry["raw_evaluator"] = evaluate(lambda s: Descriptors.MolLogP(Chem.MolFromSmiles(s)), smiles)
        elif name == "rdkit_qed":
            entry["raw_evaluator"] = evaluate(lambda s: QED.qed(Chem.MolFromSmiles(s)), smiles)
        result["probes"][label] = entry

    result["batch"] = evaluate(f, np.array(list(PROBES.values())).reshape(-1, 1))
    result["repeat"] = evaluate(f, np.array([[PROBES["poli_example"]]]))
    # Invalid representation behavior is recorded separately, not asserted to
    # be universal: RDKit tasks use NaN while TDC tasks commonly return zero.
    result["invalid_smiles"] = evaluate(f, np.array([["not_a_smiles"]]))
    result["invalid_selfies"] = evaluate(selfies_problem.black_box, np.array([["[INVALID]"]]))
    result["unknown_selfies_tokens"] = sorted({
        t for smiles in PROBES.values() for t in sf.split_selfies(sf.encoder(smiles))
        if t not in load_alphabet_for_pmo()
    })
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--problems", nargs="+")
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"Refusing to overwrite existing evidence: {args.output}")
    patch_broken_tdc_chem_utils()
    names = PMOBenchmark(string_representation="SELFIES").problem_names
    if args.problems:
        unknown = set(args.problems) - set(names)
        if unknown:
            parser.error(f"Unknown PMO tasks: {sorted(unknown)}")
        names = args.problems
    versions = {"python": platform.python_version()}
    for package in ["poli-core", "poli-baselines", "PyTDC", "scikit-learn", "numpy", "scipy", "rdkit", "selfies"]:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "unknown"
    report = {
        "versions": versions, "seed": 1, "probes": PROBES,
        "scope": "Installed PMOBenchmark; direct CPU objective probes, no BO or training",
        "repository_strict_errors": True,
        "cached_model_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in Path("oracle").glob("*.pkl")},
        "problems": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Reject HTTP requests rather than accidentally fetching missing models.
    with patch("requests.sessions.Session.request", side_effect=RuntimeError("Network disabled during PMO audit")):
        for name in names:
            start = time.monotonic()
            item = audit_problem(name)
            item["elapsed_seconds"] = round(time.monotonic() - start, 3)
            report["problems"].append(item)
            args.output.write_text(json.dumps(json_ready(report), indent=2, allow_nan=False) + "\n", encoding="utf-8")
            errors = sum("error" in p.get("raw_evaluator", {}) for p in item.get("probes", {}).values())
            print(f"{name}: init={'error' if 'error' in item['initialization'] else 'ok'}, raw_errors={errors}, creation_error={bool(item.get('creation_error'))}", flush=True)


if __name__ == "__main__":
    main()
