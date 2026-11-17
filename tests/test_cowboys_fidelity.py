"""Fast CPU regression checks; no fitted networks, downloads or real objectives."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from poli.core.black_box_information import BlackBoxInformation
from poli.core.data_package import DataPackage
from poli.core.lambda_black_box import LambdaBlackBox
from poli.core.problem import Problem

from hdbo_benchmark.solvers import cowboys
from hdbo_benchmark.generative_models.vae import VAE
from hdbo_benchmark.benchmarks.problem_transformations import transform_problem_from_discrete_to_continuous
from hdbo_benchmark.utils.logging.iteration_metrics import build_iteration_metric_artifacts
from hdbo_benchmark.experiments.run import _save_solver_progress


class ToyVAE(VAE):
    """A deliberately lossy encoder, with a deterministic decoder."""
    def __init__(self):
        torch.nn.Module.__init__(self)
        self.latent_dim = 1
        self.decoded_latents = []

    def encode_from_string_array(self, x):
        return np.zeros((len(x), 1))

    def decode_to_string_array(self, z):
        self.decoded_latents.append(np.asarray(z).copy())
        return np.array(["[N]" if row[0] < -0.05 else "[O]" for row in z])


class ConstantLogPI:
    thresholds = []

    def __init__(self, model, best_f):
        self.thresholds.append(float(best_f))

    def __call__(self, x):
        return torch.zeros(len(x), dtype=torch.float64)


class FeatureEI:
    def __init__(self, model, best_f):
        pass

    def __call__(self, x):
        return x[..., 0].reshape(-1)


class COWBOYSFidelityTests(unittest.TestCase):
    def setUp(self):
        ConstantLogPI.thresholds = []
        self.calls = []
        self.fit_features = None

    def make_solver(self, original="[C]", y=1.0, **kwargs):
        def objective(x):
            self.calls.append(np.asarray(x).copy())
            return np.full((len(x), 1), 2.0)
        bb = SimpleNamespace(__call__=objective, initial_observed_structures=np.array([original]))
        # The callable itself carries the original-structure metadata.
        objective.initial_observed_structures = bb.initial_observed_structures
        solver = cowboys.COWBOYS(objective, np.array([[0.5]]), np.array([[y]]),
                                 n_steps=1, max_sampling_steps=3, **kwargs)
        solver.set_vae_and_bounds(ToyVAE(), (-10., 10.))
        def fit(model, x, y):
            self.fit_features = x.copy()
            return object()
        solver._fit_model = fit
        solver.selfies_list_to_fingerprint_tensors = lambda xs: torch.tensor(
            [[{"[C]": 3., "[O]": 10., "[N]": 1.}[s]] for s in xs], dtype=torch.float64)
        return solver

    def sample(self, solver, noises):
        with patch.object(cowboys, "LogProbabilityOfImprovement", ConstantLogPI), \
             patch.object(cowboys, "qLogExpectedImprovement", FeatureEI), \
             patch.object(torch, "randn", side_effect=[torch.tensor([[n]], dtype=torch.float64) for n in noises]):
            return solver.next_candidate()

    def test_raw_pcn_noise_scale_and_gp_uses_original_structure(self):
        solver = self.make_solver()
        chosen = self.sample(solver, [2.])
        np.testing.assert_allclose(solver._unit_to_latent(chosen), [[.2]], atol=1e-12)
        np.testing.assert_array_equal(self.fit_features, [[3.]])
        self.assertEqual(self.calls, [])

    def test_pcn_center_and_round_trip(self):
        solver = self.make_solver()
        solver.history["x"] = [np.array([[.6]])]
        chosen = self.sample(solver, [0.])
        np.testing.assert_allclose(solver._unit_to_latent(chosen), [[2 * np.sqrt(.99)]])
        raw = np.array([[-30.], [0.], [25.]])
        np.testing.assert_allclose(solver._unit_to_latent(solver._latent_to_unit(raw)), raw)

    def test_latents_are_not_clipped_to_a_box(self):
        solver = self.make_solver()
        chosen = self.sample(solver, [300.])
        self.assertGreater(chosen[0, 0], 1.)
        np.testing.assert_allclose(solver._unit_to_latent(chosen), [[30.]])

    def test_ei_cannot_reselect_an_observed_molecule(self):
        solver = self.make_solver(original="[O]")
        solver.n_steps = 2
        chosen = self.sample(solver, [0., -10.])
        self.assertEqual(solver.vae.decode_to_string_array(solver._unit_to_latent(chosen))[0], "[N]")
        self.assertEqual(solver.iteration_sampling_metrics_history[0]["sample_unique_decoded_molecules_not_in_observed_history"], 1)

    def test_fixed_zero_and_negative_thresholds_and_bounded_failure(self):
        for best in (0., -1.):
            with self.subTest(best=best):
                ConstantLogPI.thresholds = []
                solver = self.make_solver(original="[O]", y=best)
                with self.assertRaisesRegex(RuntimeError, "after 3 pCN steps"):
                    self.sample(solver, [0., 0., 0.])
                self.assertEqual(ConstantLogPI.thresholds, [best])
                self.assertEqual(self.calls, [])

    def test_direct_api_requires_original_observations(self):
        solver = cowboys.COWBOYS(lambda x: x, np.array([[.5]]), np.array([[1.]]))
        with self.assertRaisesRegex(ValueError, "original evaluated"):
            solver.set_vae_and_bounds(ToyVAE(), (-10., 10.))
        solver.set_vae_and_bounds(ToyVAE(), (-10., 10.), initial_structures=np.array(["[C]"]))
        solver.bounds = (-2., 2.)
        np.testing.assert_array_equal(solver._unit_to_latent(np.array([[.5]])), [[0.]])

    def test_canonical_novelty_ignores_selfies_padding(self):
        solver = self.make_solver()
        self.assertEqual(solver._canonical_smiles("[C]"), solver._canonical_smiles("[C][nop]"))
        solver._record_sampling_metrics(["[C]", "[C][nop]"], [])
        self.assertEqual(solver.iteration_sampling_metrics_history[-1]["sample_unique_decoded_molecules_in_iteration"], 1)

    def test_fingerprint_collisions_keep_a_direct_latent_index(self):
        solver = self.make_solver()
        solver.n_steps = 2
        solver.selfies_list_to_fingerprint_tensors = lambda xs: torch.ones((len(xs), 1), dtype=torch.float64)
        chosen = self.sample(solver, [0., -10.])
        np.testing.assert_allclose(chosen, [[.5]])
        self.assertEqual(solver.vae.decode_to_string_array(solver._unit_to_latent(chosen))[0], "[O]")

    def test_wrapper_preserves_original_observations_without_scoring_reconstruction(self):
        info = BlackBoxInformation(name="toy", max_sequence_length=1, aligned=True,
                                   fixed_length=True, deterministic=True, alphabet=None, discrete=True)
        def forbidden(x):
            self.fail("Transformation must not evaluate a reconstruction")
        original = np.array([["[C]"]])
        problem = Problem(black_box=LambdaBlackBox(function=forbidden, info=info), x0=original,
                          data_package=DataPackage(unsupervised_data=original,
                                                   supervised_data=(original, np.array([[1.]]))))
        transformed = transform_problem_from_discrete_to_continuous(problem, ToyVAE(), (-10., 10.))
        np.testing.assert_array_equal(transformed.black_box.initial_observed_structures, original)
        np.testing.assert_array_equal(transformed.data_package.supervised_data[1], [[1.]])

    def test_one_step_and_final_saving_do_not_make_diagnostic_oracle_calls(self):
        solver = self.make_solver()
        with patch.object(cowboys, "LogProbabilityOfImprovement", ConstantLogPI), \
             patch.object(cowboys, "qLogExpectedImprovement", FeatureEI):
            solver.step()
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(solver.observed_structures[0], "[C]")
        self.assertEqual(len(solver.observed_structures), 2)
        solver.black_box.diagnostic_function = lambda x: self.fail("Unexpected diagnostic oracle call")
        x, y = solver.get_history_as_arrays()
        metrics, _ = build_iteration_metric_artifacts(solver, x, y, 1, 1)
        self.assertEqual(metrics["iterations"][0]["selected_decoded_structures"], solver.observed_structures[1:])
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            _save_solver_progress(solver, out, "cowboys", "toy", 1, 1, "completed", 1, True)
            status = json.loads((out / "cowboys_toy_1_status.json").read_text())
            self.assertFalse(status["diagnostic_objective_evaluations_enabled"])
            with np.load(out / "cowboys_toy_1_history.npz") as saved:
                self.assertEqual(saved["structures"].tolist(), solver.observed_structures)
        self.assertEqual(len(self.calls), 1)

    def test_invalid_molecules_are_not_replaced_with_toluene(self):
        solver = self.make_solver()
        self.assertIsNone(solver._canonical_smiles(""))
        with self.assertRaises(ValueError):
            cowboys.COWBOYS.selfies_list_to_fingerprint_tensors(solver, [""])


if __name__ == "__main__":
    unittest.main()
