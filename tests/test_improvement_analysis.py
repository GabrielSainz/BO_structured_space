"""Offline EI diagnostics: hand calculations, causal incumbents and missing data."""

import copy
import math
import shutil
import tempfile
import unittest
from pathlib import Path

from hdbo_benchmark.evaluation.improvement_analysis import analyze_improvements, improvement_report
from hdbo_benchmark.evaluation.metric_io import io_path, read_json


def toy_run():
    initial = [{"phase": "initial", "bo_step": 0, "score_for_metrics": value,
                "valid_molecule": True} for value in (.2, .4)]
    observations = []
    for index, (score, mu, incumbent, ei, proxy) in enumerate(
        ((.3, .35, .4, .05, .04), (.7, .5, .4, .15, .14), (.6, .6, .7, .02, .01)), 1
    ):
        observations.append({
            "phase": "optimization", "bo_step": index, "observation_index": index + 1,
            "score_for_metrics": score, "valid_molecule": True, "canonical_smiles": "C" * index,
            "pre_evaluation_prediction": {"mean": mu, "latent_std": .1, "predictive_std": .2,
                                          "incumbent_for_acquisition": incumbent, "analytic_expected_improvement": ei},
            "recorded_acquisition": {"convention": "qLogExpectedImprovement", "selection_score": math.log(proxy)},
        })
    return {"task": {"direction": "maximize", "score_transform": "identity"},
            "design": {"batch_size": 1, "n_initial_observed": 2}, "evaluations": initial + observations}


class ImprovementTests(unittest.TestCase):
    def test_hand_computed_causal_gains_and_telescoping(self):
        result = analyze_improvements(toy_run())
        rows, summary = result["rows"], result["summary"]
        self.assertEqual([r["incumbent_before"] for r in rows], [.4, .4, .7])
        for actual, expected in zip([r["realized_improvement"] for r in rows], [0, .3, 0]):
            self.assertAlmostEqual(actual, expected)
        self.assertEqual(summary["record_improvement_steps"], [2])
        self.assertEqual(summary["first_step_of_final_best"], 2)
        self.assertEqual(summary["steps_without_a_new_best_at_end"], 1)
        self.assertAlmostEqual(summary["sum_realized_improvements"], .3)
        self.assertAlmostEqual(summary["realized_best_score_gain"], .3)
        self.assertAlmostEqual(summary["sum_analytic_ei"], .22)
        self.assertAlmostEqual(summary["sum_analytic_ei_after_last_record"], .02)

    def test_uncertainty_uplift_and_log_ei_remain_distinct(self):
        result = analyze_improvements(toy_run())
        second = result["rows"][1]
        self.assertAlmostEqual(second["positive_mean_advantage"], .1)
        self.assertAlmostEqual(second["uncertainty_uplift"], .05)
        self.assertAlmostEqual(second["analytic_expected_improvement"], .15)
        self.assertAlmostEqual(second["exp_recorded_log_ei_smoothed_mc_proxy"], .14)
        self.assertEqual(result["summary"]["n_means_below_incumbent"], 2)

    def test_future_observations_do_not_rewrite_earlier_metrics(self):
        full = toy_run()
        truncated = copy.deepcopy(full)
        truncated["evaluations"] = truncated["evaluations"][:-1]
        self.assertEqual(analyze_improvements(full)["rows"][:2], analyze_improvements(truncated)["rows"])

    def test_no_missing_ei_forward_fill_or_zero_imputation(self):
        run = toy_run()
        run["evaluations"][3]["pre_evaluation_prediction"] = None
        result = analyze_improvements(run)
        self.assertEqual(result["summary"]["n_comparable_analytic_ei"], 2)
        self.assertEqual([r["cumulative_analytic_ei"] for r in result["rows"]], [.05, None, None])
        self.assertIsNone(result["summary"]["sum_analytic_ei"])
        self.assertAlmostEqual(result["summary"]["sum_realized_improvements"], .3)

    def test_different_ei_threshold_is_not_compared(self):
        run = toy_run()
        run["evaluations"][2]["pre_evaluation_prediction"]["incumbent_for_acquisition"] = .7
        result = analyze_improvements(run)
        row = result["rows"][0]
        self.assertFalse(row["ei_incumbent_matches_observed_prefix"])
        self.assertIsNone(row["analytic_expected_improvement"])
        self.assertIsNone(row["exp_recorded_log_ei_smoothed_mc_proxy"])
        self.assertEqual(row["recorded_analytic_ei"], .05)
        self.assertEqual(row["realized_improvement"], 0.)

    def test_repeated_record_scores_have_zero_new_gain(self):
        run = toy_run()
        run["evaluations"][-1]["score_for_metrics"] = .7
        result = analyze_improvements(run)
        self.assertEqual(result["rows"][-1]["realized_improvement"], 0.)
        self.assertEqual(result["summary"]["n_record_improvements"], 1)

    def test_incompatible_scale_and_absent_initialization_rejected(self):
        for change in ("minimize", "missing_initialization"):
            run = toy_run()
            if change == "minimize":
                run["task"]["direction"] = "minimize"
            else:
                run["evaluations"] = run["evaluations"][1:]
            with self.subTest(change=change), self.assertRaises(ValueError):
                analyze_improvements(run)

    def test_real_fixture_offline_report_preserves_inputs_and_old_plot(self):
        fixture = Path(__file__).parent / "fixtures/cowboys_checkpoint"
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "source"
            shutil.copytree(fixture, root)
            before = {p.name: p.read_bytes() for p in root.iterdir() if p.is_file()}
            output = Path(temp) / "plots"
            output.mkdir()
            previous = output / "gp_mean_vs_observed_over_time.png"
            previous.write_bytes(b"previous plot retained")
            report = improvement_report(root / "cowboys_albuterol_similarity_1_status.json", output)
            self.assertTrue(report["original_inputs_unchanged"])
            self.assertEqual(report["summary"]["n_bo_steps"], 1)
            self.assertEqual(report["summary"]["n_comparable_analytic_ei"], 0)
            self.assertIsNone(report["summary"]["sum_analytic_ei"])
            self.assertEqual(previous.read_bytes(), b"previous plot retained")
            self.assertEqual(before, {p.name: p.read_bytes() for p in root.iterdir() if p.is_file()})
            for kind in ("png", "pdf", "csv", "md"):
                self.assertTrue(io_path(Path(report["files"][kind])).is_file())
            saved = read_json(output / "improvement_analysis.json")
            self.assertFalse(saved["oracle_or_solver_rerun"])


if __name__ == "__main__":
    unittest.main()
