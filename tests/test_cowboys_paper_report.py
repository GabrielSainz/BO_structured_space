"""Paper comparison must never promote failed/partial runs to full results."""

import copy
import json
import tempfile
import unittest
from pathlib import Path

from hdbo_benchmark.evaluation.compare_cowboys_paper import compare_results


class PaperReportTests(unittest.TestCase):
    def setUp(self):
        self.reference = {
            "tasks": {"toy": {"mean": .5, "sd": .1}}, "repetitions": 5,
            "bo_iterations": 300, "initial_molecules": 10, "num_chains": 1,
            "n_steps": 100, "batch_size": 1, "latent_dim": 128,
            "checkpoint_sha256": "verified-checkpoint",
        }
        self.status = {
            "status": "completed", "completed_iterations": 300,
            "initial_history_size": 10, "best_value": .5,
            "diagnostic_objective_evaluations_enabled": False,
            "solver_configuration": {
                "implementation": "cowboys_corrected_pcn_v1", "num_chains": 1,
                "n_steps": 100, "batch_size": 1, "latent_dim": 128,
                "vae_checkpoint_sha256": "verified-checkpoint",
            },
        }

    def test_five_matching_seeds_produce_mean_sd_and_difference(self):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            for seed in range(1, 6):
                status = {**self.status, "best_value": seed / 10}
                (out / f"cowboys_toy_{seed}_status.json").write_text(json.dumps(status))
            row = compare_results(out, self.reference)[0]
        self.assertTrue(row["comparison_complete"])
        self.assertAlmostEqual(row["local_mean"], .3)
        self.assertAlmostEqual(row["local_sample_sd"], .15811388300841897)
        self.assertAlmostEqual(row["local_minus_paper_mean"], -.2)

    def test_failed_partial_or_wrong_protocol_is_excluded(self):
        cases = [
            {"status": "failed"}, {"completed_iterations": 1},
            {"initial_history_size": 5}, {"diagnostic_objective_evaluations_enabled": True},
            {"best_value": float("nan")},
            {"solver_configuration": {**self.status["solver_configuration"], "num_chains": 10}},
            {"solver_configuration": {**self.status["solver_configuration"], "vae_checkpoint_sha256": "other"}},
        ]
        for override in cases:
            with self.subTest(override=override), tempfile.TemporaryDirectory() as folder:
                out = Path(folder)
                status = {**copy.deepcopy(self.status), **override}
                (out / "cowboys_toy_1_status.json").write_text(json.dumps(status))
                row = compare_results(out, self.reference)[0]
                self.assertEqual(row["completed_matching_seeds"], 0)
                self.assertIsNone(row["local_mean"])
                self.assertFalse(row["comparison_complete"])

    def test_one_completed_seed_is_explicitly_incomplete(self):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            (out / "cowboys_toy_1_status.json").write_text(json.dumps(self.status))
            row = compare_results(out, self.reference)[0]
        self.assertFalse(row["comparison_complete"])
        self.assertEqual(row["completed_matching_seeds"], 1)
        self.assertIsNone(row["local_sample_sd"])


if __name__ == "__main__":
    unittest.main()
