"""Pre-evaluation GP logging, clean exports and preservation of existing plots."""

import copy
import csv
import json
import math
import random
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from botorch.models import SingleTaskGP

from hdbo_benchmark.evaluation.aggregate_runs import aggregate_directories
from hdbo_benchmark.evaluation.export_runs import build_run, export_directory, validate_run
from hdbo_benchmark.evaluation.metric_io import digest_file, io_path, read_json, write_json
from hdbo_benchmark.evaluation.prediction_metrics import join_predictions, prediction_summary
from hdbo_benchmark.evaluation.run_views import write_run_views
from hdbo_benchmark.experiments.run import _save_solver_progress
from hdbo_benchmark.solvers import cowboys
from hdbo_benchmark.utils.logging.gp_predictions import (
    PREDICTION_CONVENTIONS, analytic_expected_improvement, attach_gp_prediction_recorder,
)
from tests.test_cowboys_fidelity import ConstantLogPI, FeatureEI, ToyVAE


FIXTURE = Path(__file__).parent / "fixtures/cowboys_checkpoint"
STEM = "cowboys_albuterol_similarity_1"


class GPRecorderTests(unittest.TestCase):
    def run_toy(self, recording, failing=False):
        random.seed(123)
        np.random.seed(123)
        torch.manual_seed(123)
        calls, boundaries = [], []
        def objective(x):
            if recording:
                boundaries.append((len(solver.get_history_as_arrays()[0]),
                                   len(solver.pre_evaluation_prediction_history)))
            calls.append(np.asarray(x).copy())
            return np.array([[13.]])
        objective.initial_observed_structures = np.array(["[C]", "[C]"])
        solver = cowboys.COWBOYS(objective, np.array([[.5], [.5]]), np.array([[10.], [12.]]),
                                 n_steps=1, max_sampling_steps=3)
        solver.set_vae_and_bounds(ToyVAE(), (-10., 10.))
        solver.selfies_list_to_fingerprint_tensors = lambda structures: torch.tensor(
            [[{"[C]": 3., "[N]": 1., "[O]": 10.}[s]] for s in structures], dtype=torch.float64)
        class PosteriorGP:
            train_inputs = (torch.ones((2, 1), dtype=torch.float64),)
            def posterior(self, features, observation_noise=False):
                # Deliberately consume every RNG: diagnostics must restore them.
                random.random()
                np.random.random()
                torch.rand(1)
                if failing:
                    raise RuntimeError("diagnostic-only failure")
                self.before_oracle = len(calls) == 0
                return SimpleNamespace(mean=torch.tensor([[11.5]], dtype=torch.float64),
                                       variance=torch.tensor([[1. if observation_noise else .25]], dtype=torch.float64))
        model = PosteriorGP()
        def fit(*args):
            solver.gp_model_of_objective = model
            return model
        solver._fit_model = fit
        if recording:
            attach_gp_prediction_recorder(solver, 2)
            attach_gp_prediction_recorder(solver, 2)
        with patch.object(cowboys, "LogProbabilityOfImprovement", ConstantLogPI), \
             patch.object(cowboys, "qLogExpectedImprovement", FeatureEI):
            selected = solver.step()
        return solver, selected, calls, boundaries, (random.getstate(), np.random.get_state(), torch.random.get_rng_state())

    def test_before_oracle_and_no_changes_to_candidates_calls_rng_or_chain(self):
        plain, plain_selected, plain_calls, _, plain_rng = self.run_toy(False)
        logged, logged_selected, logged_calls, boundaries, logged_rng = self.run_toy(True)
        np.testing.assert_array_equal(plain_selected[0], logged_selected[0])
        np.testing.assert_array_equal(plain_selected[1], logged_selected[1])
        np.testing.assert_array_equal(plain.betas, logged.betas)
        self.assertEqual(plain.observed_structures, logged.observed_structures)
        self.assertEqual(len(plain_calls), 1)
        self.assertEqual(len(logged_calls), 1)
        self.assertEqual(boundaries, [(2, 1)])
        self.assertEqual(plain_rng[0], logged_rng[0])
        np.testing.assert_array_equal(plain_rng[1][1], logged_rng[1][1])
        self.assertEqual(plain_rng[1][2:], logged_rng[1][2:])
        self.assertTrue(torch.equal(plain_rng[2], logged_rng[2]))
        record = logged.pre_evaluation_prediction_history[0]
        self.assertEqual(record["observation_index"], 2)
        self.assertEqual(record["bo_step"], 1)
        self.assertEqual(record["mean"], 11.5)
        self.assertEqual(record["latent_std"], .5)
        self.assertEqual(record["predictive_std"], 1.)
        self.assertTrue(logged.gp_model_of_objective.before_oracle)

    def test_diagnostic_failure_does_not_abort_or_rescore(self):
        solver, _, calls, _, _ = self.run_toy(True, failing=True)
        self.assertEqual(len(calls), 1)
        record = solver.pre_evaluation_prediction_history[0]
        self.assertIsNone(record["mean"])
        self.assertIn("diagnostic-only failure", record["missing_reason"]["prediction"])

    def test_real_gp_posterior_is_on_original_score_scale(self):
        model = SingleTaskGP(torch.tensor([[.1], [.9]], dtype=torch.float64),
                             torch.tensor([[10.], [20.]], dtype=torch.float64))
        model.eval()
        candidate = np.array([[.5]])
        solver = SimpleNamespace(
            next_candidate=lambda: candidate,
            get_history_as_arrays=lambda: (np.array([[.1], [.9]]), np.array([[10.], [20.]])),
            iteration_candidate_diagnostics_history=[{"top_candidates": [
                {"unit_latent": [.5], "structure": "[C]"}]}],
            _canonical_smiles=lambda _: "C",
            selfies_list_to_fingerprint_tensors=lambda _: torch.tensor([[.5]], dtype=torch.float64),
            gp_model_of_objective=model, penalize_nans_with=-10.,
        )
        with torch.no_grad():
            expected = model.posterior(torch.tensor([[.5]], dtype=torch.float64))
            noisy = model.posterior(torch.tensor([[.5]], dtype=torch.float64), observation_noise=True)
        parameters = [p.detach().clone() for p in model.parameters()]
        attach_gp_prediction_recorder(solver, 2)
        self.assertIs(solver.next_candidate(), candidate)
        record = solver.pre_evaluation_prediction_history[0]
        self.assertAlmostEqual(record["mean"], expected.mean.item())
        self.assertAlmostEqual(record["latent_std"], math.sqrt(expected.variance.item()))
        self.assertAlmostEqual(record["predictive_std"], math.sqrt(noisy.variance.item()))
        self.assertGreater(record["mean"], 1.)  # Outcome standardization has been undone.
        for before, after in zip(parameters, model.parameters()):
            self.assertTrue(torch.equal(before, after))
        self.assertFalse(model.training)

    def test_analytic_ei_convention(self):
        self.assertAlmostEqual(analytic_expected_improvement(1., 1., 1.), 1 / math.sqrt(2 * math.pi))
        self.assertEqual(analytic_expected_improvement(2., 0., 1.), 1.)
        self.assertEqual(analytic_expected_improvement(0., 0., 1.), 0.)
        self.assertIsNone(analytic_expected_improvement(None, 1., 1.))
        self.assertIsNone(analytic_expected_improvement(float("inf"), 1., 1.))

    def test_runner_saves_predictions_even_when_legacy_iteration_json_disabled(self):
        solver, _, calls, _, _ = self.run_toy(True)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _save_solver_progress(solver, root, "cowboys", "toy", 1, 1, "completed", 2, False)
            status = read_json(root / "cowboys_toy_1_status.json")
            saved = read_json(root / status["predictions_file"])
            self.assertEqual(len(saved["predictions"]), 1)
            self.assertEqual(saved["predictions"][0]["mean"], 11.5)
            self.assertTrue(status["pre_evaluation_predictions_enabled"])
            self.assertFalse(status["iteration_results_enabled"])
        self.assertEqual(len(calls), 1)


class PredictionExportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "group"
        shutil.copytree(FIXTURE, self.root)
        self.status_path = self.root / f"{STEM}_status.json"

    def add_prediction(self):
        data = read_json(self.root / f"{STEM}_iteration_results.json")
        record = {"observation_index": 10, "bo_step": 1, "n_training_observations": 10,
                  "captured": "after_candidate_selection_before_oracle_evaluation",
                  "canonical_smiles": data["iterations"][0]["selected_smiles"][0],
                  "mean": .2, "latent_std": .1, "predictive_std": .2,
                  "analytic_expected_improvement": .01, "incumbent_for_acquisition": .32941176470588235,
                  "missing_reason": {}}
        path = self.root / f"{STEM}_predictions.json"
        write_json(path, {"solver_name": "cowboys", "function_name": "albuterol_similarity", "seed": 1,
                          "conventions": PREDICTION_CONVENTIONS, "predictions": [record]})
        return record

    def test_prediction_sidecar_join_and_calibration(self):
        self.add_prediction()
        run = build_run(self.status_path, planned_bo_steps=1)
        validate_run(run)
        e = run["evaluations"][-1]
        self.assertEqual(e["pre_evaluation_prediction"]["mean"], .2)
        self.assertAlmostEqual(e["prediction_residual"], .14492753623188404 - .2)
        self.assertAlmostEqual(e["predictive_gaussian_nll"], math.log(.2) + .5 * math.log(2 * math.pi)
                               + .5 * ((.14492753623188404 - .2) / .2) ** 2)
        self.assertAlmostEqual(run["summary"]["gp_diagnostics"]["prequential_rmse"], abs(e["prediction_residual"]))
        self.assertEqual(run["summary"]["gp_diagnostics"]["coverage_90pct"], 1.)
        self.assertIsNone(run["evaluations"][0]["pre_evaluation_prediction"])
        self.assertNotIn("gp_predictions_and_calibration", run["missing_reason"])
        self.assertIn("pre_evaluation_prediction", run["evaluations"][0]["missing_reason"])

    def test_future_model_or_wrong_molecule_records_are_rejected(self):
        record = self.add_prediction()
        baseline = build_run(self.status_path)
        for replacement in ({"n_training_observations": 11}, {"bo_step": 2}, {"canonical_smiles": "C"},
                            {"captured": "after_oracle_evaluation"}):
            evaluations = copy.deepcopy(baseline["evaluations"])
            for e in evaluations:
                e["pre_evaluation_prediction"] = None
            count = join_predictions(evaluations, {"predictions": [{**record, **replacement}]}, [])
            self.assertEqual(count, 0)

    def test_clean_views_preserve_all_metrics_raw_diagnostics_and_repeatability(self):
        originals = {p.name: p.read_bytes() for p in self.root.iterdir() if p.is_file()}
        before = build_run(self.status_path, planned_bo_steps=1)
        manifest = export_directory(self.root, planned_bo_steps=1)
        path = Path(manifest["runs"][0]["path"])
        after = read_json(path)
        self.assertNotIn("recorded_iteration_diagnostics", after)
        self.assertEqual(before["trajectory"], after["trajectory"])
        self.assertEqual(before["summary"], after["summary"])
        self.assertEqual(before["evaluations"], after["evaluations"])
        self.assertEqual(before["recorded_iteration_diagnostics"], read_json(path.parent / "diagnostics.json"))
        self.assertTrue(io_path(path.parent / "summary.json").exists())
        with io_path(path.parent / "evaluations.csv").open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 11)
        self.assertEqual(rows[-1]["predicted_mean"], "")
        self.assertEqual(originals, {p.name: p.read_bytes() for p in self.root.iterdir() if p.is_file()})
        saved = {p.name: p.read_bytes() for p in io_path(path.parent).iterdir()}
        export_directory(self.root, planned_bo_steps=1)
        self.assertEqual(saved, {p.name: p.read_bytes() for p in io_path(path.parent).iterdir()})

    def test_nonfinite_diagnostics_remain_valid_json_and_do_not_discard_scores(self):
        record = self.add_prediction()
        run = build_run(self.status_path)
        evaluations = copy.deepcopy(run["evaluations"])
        evaluations[-1]["raw_oracle_score"] = 1e308
        record.update(mean=0., predictive_std=1e-308)
        join_predictions(evaluations, {"predictions": [record]}, [])
        self.assertEqual(evaluations[-1]["prediction_residual"], 1e308)
        self.assertIsNone(evaluations[-1]["predictive_gaussian_nll"])
        self.assertEqual(prediction_summary(evaluations)["prequential_rmse"], 1e308)
        json.dumps(evaluations, allow_nan=False)

    def test_old_plot_images_unchanged_and_new_gp_plots_are_additive(self):
        legacy = build_run(self.status_path, planned_bo_steps=1)
        folder = self.root / f"metrics/{STEM}"
        write_json(folder / "run.json", legacy)
        baseline = aggregate_directories([self.root], output_dir=self.root / "baseline_plots")
        baseline_images = {Path(p["path"]).name: digest_file(Path(p["path"]))
                           for p in baseline["plot_manifest"] if p["status"] == "generated"}
        self.add_prediction()
        updated = build_run(self.status_path, planned_bo_steps=1)
        write_run_views(folder, updated)
        new = aggregate_directories([self.root], output_dir=self.root / "new_plots")
        new_images = {Path(p["path"]).name: digest_file(Path(p["path"]))
                      for p in new["plot_manifest"] if p["status"] == "generated"}
        for filename, sha in baseline_images.items():
            self.assertEqual(new_images[filename], sha)
        self.assertTrue(any("gp_predicted_vs_observed" in name for name in new_images))
        self.assertTrue(any("gp_prediction_trace" in name for name in new_images))


if __name__ == "__main__":
    unittest.main()
