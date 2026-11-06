"""Check cached classifier inference in a specified diagnostic Python runtime.

First write RDKit fingerprints in the research environment, then evaluate them
with the pickle's original scikit-learn version in a separate process:

python scripts/diagnostics/check_pmo_models.py --write-fingerprints probes.json
python -S scripts/diagnostics/check_pmo_models.py --packages <temporary-target> \
    --fingerprints probes.json --model-dir oracle --output model_checks.json

This script does not train, convert or replace models, or provide a new runtime
backend for optimization. Only existing, trusted local pickle files are loaded.
"""

import argparse
import hashlib
import json
import pickle
import platform
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-fingerprints", type=Path)
    parser.add_argument("--fingerprints", type=Path)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--packages", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.write_fingerprints or args.output
    if output is None or output.exists():
        parser.error("Specify a fresh output path; existing evidence is never overwritten.")
    if args.write_fingerprints:
        from rdkit import Chem
        from rdkit.Chem import AllChem
        from check_pmo import PROBES

        fingerprints = {}
        for label, smiles in PROBES.items():
            mol = Chem.MolFromSmiles(smiles)
            bits = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
            counts = AllChem.GetMorganFingerprint(mol, 3, useCounts=True, useFeatures=True)
            folded = [0] * 2048
            for index, count in counts.GetNonzeroElements().items():
                folded[index % 2048] += int(count)
            fingerprints[label] = {"smiles": smiles, "bits": list(bits), "counts": folded}
        output.write_text(json.dumps(fingerprints, indent=2) + "\n", encoding="utf-8")
        return
    if args.fingerprints is None or args.model_dir is None:
        parser.error("Inference requires --fingerprints and --model-dir.")
    if args.packages:
        sys.path.insert(0, str(args.packages.resolve()))
    import numpy as np
    import scipy
    import sklearn

    fingerprints = json.loads(args.fingerprints.read_text(encoding="utf-8"))
    report = {
        "versions": {"python": platform.python_version(), "scikit-learn": sklearn.__version__,
                     "numpy": np.__version__, "scipy": scipy.__version__},
        "models": {},
    }
    for name, feature_kind in [("jnk3", "bits"), ("gsk3b", "bits"), ("drd2", "counts")]:
        path = args.model_dir / (name + "_current.pkl")
        item = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        try:
            with path.open("rb") as handle:
                model = pickle.load(handle)
            item["predictions"] = {}
            for label, features in fingerprints.items():
                values = np.asarray(features[feature_kind], dtype=np.int32).reshape(1, -1)
                item["predictions"][label] = float(model.predict_proba(values)[0, 1])
        except Exception as exc:
            item["error"] = {"type": type(exc).__name__, "message": str(exc)}
        report["models"][name] = item
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if any("error" in item for item in report["models"].values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
