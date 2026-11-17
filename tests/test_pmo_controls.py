"""Small offline molecular controls; requires the existing research environment."""

import math
import unittest
from unittest.mock import patch

import numpy as np
import poli
import selfies as sf
from rdkit import Chem
from rdkit.Chem import Descriptors

from hdbo_benchmark.benchmarks.load_problems import tokenize_selfies
from hdbo_benchmark.utils.experiments.load_metadata_for_vaes import load_alphabet_for_pmo
from hdbo_benchmark.utils.tdc_compat import (
    expose_tdc_evaluation_errors,
    patch_broken_tdc_chem_utils,
)


REFERENCE_TARGETS = {
    "albuterol_similarity": "CC(C)(C)NCC(O)c1ccc(O)c(CO)c1",
    "mestranol_similarity": "COc1ccc2[C@H]3CC[C@@]4(C)[C@@H](CC[C@@]4(O)C#C)[C@@H]3CCc2c1",
    "celecoxib_rediscovery": "CC1=CC=C(C=C1)C1=CC(=NN1C1=CC=C(C=C1)S(N)(=O)=O)C(F)(F)F",
    "thiothixene_rediscovery": "CN(C)S(=O)(=O)c1ccc2Sc3ccccc3C(=CCCN4CCN(C)CC4)c2c1",
    "troglitazone_rediscovery": "Cc1c(C)c2OC(C)(COc3ccc(CC4SC(=O)NC4=O)cc3)CCc2c(C)c1O",
}
VALSARTAN_POSITIVE = "CN(C=O)Cc1ccc(-c2ccc(C(N)=O)cc2)cc1O"
SITAGLIPTIN = "NC(CC(=O)N1CCn2c(nnc2C(F)(F)F)C1)Cc1cc(F)c(F)cc1F"


class PMOControlTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        patch_broken_tdc_chem_utils()

    def setUp(self):
        self.network = patch(
            "requests.sessions.Session.request",
            side_effect=AssertionError("Controls must not download models"),
        )
        self.network.start()
        self.addCleanup(self.network.stop)

    def evaluate(self, name, smiles, representation):
        problem = poli.create(name=name, string_representation=representation)
        expose_tdc_evaluation_errors(problem.black_box)
        x = np.array([[smiles]]) if representation == "SMILES" else np.array([
            tokenize_selfies(sf.encoder(smiles), 70)
        ])
        y = problem.black_box(x)
        self.assertEqual(y.shape, (1, 1))
        self.assertEqual(problem.black_box.num_evaluations, 1)
        return float(y[0, 0])

    def test_reference_targets_score_one(self):
        for name, smiles in REFERENCE_TARGETS.items():
            for representation in ["SMILES", "SELFIES"]:
                with self.subTest(problem=name, representation=representation):
                    self.assertAlmostEqual(self.evaluate(name, smiles, representation), 1.0, places=12)

    def test_valsartan_zero_without_motif_and_positive_with_motif(self):
        self.assertFalse(
            set(sf.split_selfies(sf.encoder(VALSARTAN_POSITIVE))) - set(load_alphabet_for_pmo())
        )
        target = Chem.MolFromSmiles(SITAGLIPTIN)
        molecule = Chem.MolFromSmiles(VALSARTAN_POSITIVE)
        # Independently compute the specified three Gaussian descriptor terms
        # and geometric mean, given a matching SMARTS term of one.
        penalties = [
            ((descriptor(molecule) - descriptor(target)) / sigma) ** 2
            for descriptor, sigma in [
                (Descriptors.MolLogP, 0.2), (Descriptors.TPSA, 5.0),
                (Descriptors.BertzCT, 30.0),
            ]
        ]
        expected = math.exp(-sum(penalties) / 8.0)
        self.assertGreater(expected, 1e-5)
        for representation in ["SMILES", "SELFIES"]:
            with self.subTest(representation=representation):
                self.assertEqual(self.evaluate("valsartan_smarts", "CCO", representation), 0.0)
                observed = self.evaluate("valsartan_smarts", VALSARTAN_POSITIVE, representation)
                self.assertAlmostEqual(observed / expected, 1.0, places=12)


if __name__ == "__main__":
    unittest.main()
