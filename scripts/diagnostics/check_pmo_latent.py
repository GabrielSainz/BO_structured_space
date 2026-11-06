"""Compare existing VAE/normalization wrappers with direct PMO evaluations.

Uses three fixed CPU latent points and the shipped 128-dimensional checkpoint.
No optimization, training or model download is performed.
"""

import argparse
import json
import os
from pathlib import Path
from unittest.mock import patch

import numpy as np
import poli
import selfies as sf
import torch
from poli.benchmarks import PMOBenchmark
from rdkit import Chem

from hdbo_benchmark.benchmarks.problem_transformations import (
    _in_the_latent_space_of_molecules,
)
from hdbo_benchmark.generative_models.vae_factory import VAEFactory
from hdbo_benchmark.utils.experiments.normalization import from_range_to_unit_cube
from hdbo_benchmark.utils.tdc_compat import (
    expose_tdc_evaluation_errors,
    patch_broken_tdc_chem_utils,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Refusing to overwrite existing evidence.")
    if torch.cuda.is_available() and os.environ.get("HDBO_TORCH_DEVICE") != "cpu":
        parser.error("Set HDBO_TORCH_DEVICE=cpu for this diagnostic.")
    patch_broken_tdc_chem_utils()
    with patch("requests.sessions.Session.request", side_effect=RuntimeError("Network disabled during audit")):
        vae = VAEFactory().create(experiment_name="benchmark_on_pmo", latent_dim=128)
        vae.eval()
        raw_z = np.random.RandomState(1).normal(size=(3, 128))
        unit_z = from_range_to_unit_cube(raw_z, (-10.0, 10.0))
        with torch.inference_mode():
            decoded = vae.decode_to_string_array(raw_z)
            decoded_smiles = [sf.decoder(s) for s in decoded]
            assert all(Chem.MolFromSmiles(s) is not None for s in decoded_smiles)
            report = {"seed": 1, "latent_shape": [3, 128], "bounds": [-10.0, 10.0],
                      "decoded_smiles": decoded_smiles, "problems": []}
            for name in PMOBenchmark(string_representation="SELFIES").problem_names:
                result = {"name": name}
                try:
                    problem = poli.create(name=name, string_representation="SELFIES")
                    expose_tdc_evaluation_errors(problem.black_box)
                    latent = _in_the_latent_space_of_molecules(problem, vae, (-10.0, 10.0))
                    direct = problem.black_box(decoded)
                    assert direct.shape == (3, 1) and np.isfinite(direct).all()
                    wrapped = latent(unit_z)
                    diagnostic = latent.diagnostic_function(unit_z)
                    np.testing.assert_allclose(wrapped, direct, rtol=1e-12, atol=0)
                    np.testing.assert_allclose(diagnostic, direct, rtol=1e-12, atol=0)
                    result.update(passed=True, y=direct.tolist(), shape=list(wrapped.shape))
                except Exception as exc:
                    result.update(passed=False, error={"type": type(exc).__name__, "message": str(exc)})
                report["problems"].append(result)
                print(name, result["passed"], flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
