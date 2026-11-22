"""Analysis-only checks: no objectives, optimization or model loading."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from hdbo_benchmark.evaluation.cowboys_analysis import (
    add_chemistry, aggregate_seeds, analyze, chemistry, close, curve_statistics,
    load_run, novelty_bins, rolling_frame, summarize_run, thresholds_for,
)
from hdbo_benchmark.evaluation.metric_io import digest_file


class AnalysisTests(unittest.TestCase):
    def fixture(self, root, scores=(.5, .5, .8), history=True, seed=1):
        stem = f"cowboys_test_task_{seed}"
        status = {"solver_name": "cowboys", "function_name": "test_task", "seed": seed,
                  "initial_history_size": 1, "completed_iterations": len(scores), "status": "completed",
                  "best_value": max(.3, *[v for v in scores if v is not None]),
                  "solver_configuration": {"batch_size": 1, "max_sampling_steps": 10000}}
        molecules = ["CCO", "CCN", "c1ccccc1"]
        records, predictions = [], []
        best = .3
        for i, y in enumerate(scores, 1):
            previous = best
            if y is not None:
                best = max(best, y)
            records.append({"iteration": i, "num_selected": 1, "history_size_after_iteration": 1 + i,
                            "selected_smiles": [molecules[i - 1]], "selected_valid_mask": [True],
                            "selected_objective_values": [y], "best_so_far_objective": best,
                            "mean_selected_nearest_previous_tanimoto_distance": None if i == 1 else .2,
                            "sample_unique_decoded_molecules_in_iteration": 20,
                            "sample_new_distinct_decoded_molecules": 10,
                            "available_unique_top_10_candidate_count": 1,
                            "top_candidate_diagnostics": [{"selection_score": -12345}]})
            predictions.append({"bo_step": i, "observation_index": i, "n_training_observations": i,
                                "canonical_smiles": molecules[i - 1],
                                "captured": "after_candidate_selection_before_oracle_evaluation",
                                "mean": .6, "latent_std": .05, "predictive_std": .2,
                                "analytic_expected_improvement": .123, "incumbent_for_acquisition": previous,
                                "missing_reason": {}})
        path = root / (stem + "_status.json")
        path.write_text(json.dumps(status), encoding="utf-8")
        (root / (stem + "_iteration_results.json")).write_text(json.dumps({
            "solver_name": "cowboys", "function_name": "test_task", "seed": seed, "batch_size": 1,
            "initial_history_size": 1, "iterations": records}), encoding="utf-8")
        pred_path = root / (stem + "_predictions.json")
        pred_path.write_text(json.dumps({"solver_name": "cowboys", "function_name": "test_task", "seed": seed,
                                         "predictions": predictions}), encoding="utf-8")
        if history:
            np.savez(root / (stem + "_history.npz"), y=np.array([.3, *scores], dtype=float).reshape(-1, 1),
                     structures=np.array(["C", *molecules[:len(scores)]]))
        return path, pred_path

    def test_pre_evaluation_join_metrics_and_input_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, _ = self.fixture(root)
            before = {p: digest_file(p) for p in root.iterdir()}
            run = load_run(path, 3)
            self.assertEqual(run["errors"], [])
            self.assertEqual(run["inventory"]["available_gp_predictions"], 3)
            a, tie, final = run["rows"]
            self.assertAlmostEqual(a["gain"], .2)
            self.assertEqual(tie["success"], 0)
            self.assertAlmostEqual(final["gain"], .3)
            self.assertAlmostEqual(final["b"], .5)
            self.assertAlmostEqual(a["analytic_expected_improvement"], .123)
            self.assertAlmostEqual(a["predictive_std"], .2)
            self.assertAlmostEqual(a["latent_std"], .05)
            self.assertAlmostEqual(a["error"], -.1)
            self.assertEqual(a["coverage_95"], 1)
            self.assertGreater(a["pi"], .9)
            self.assertEqual(before, {p: digest_file(p) for p in root.iterdir()})

    def test_missing_zero_uncertainty_and_no_initial_history(self):
        with tempfile.TemporaryDirectory() as directory:
            path, predictions = self.fixture(Path(directory), scores=(.5, None, .8), history=False)
            doc = json.loads(predictions.read_text())
            doc["predictions"][0]["predictive_std"] = 0
            doc["predictions"][2]["mean"] = None
            predictions.write_text(json.dumps(doc))
            run = load_run(path, 3)
            a, b, c = run["rows"]
            self.assertTrue(np.isnan(a["pi"]))
            self.assertTrue(np.isnan(a["coverage_95"]))
            self.assertTrue(np.isfinite(a["error"]))
            self.assertTrue(np.isnan(b["success"]))
            self.assertTrue(np.isnan(b["error"]))
            self.assertTrue(np.isnan(c["error"]))
            self.assertEqual(run["inventory"]["chemistry_scope"], "BO_selected_only")
            self.assertFalse(a["incumbent_verified"])

    def test_wrong_identity_boundary_duplicates_and_incumbent(self):
        for field, value in [("canonical_smiles", "CCCC"), ("n_training_observations", 99),
                             ("captured", "after_evaluation"), ("incumbent_for_acquisition", .01)]:
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                path, predictions = self.fixture(Path(directory))
                doc = json.loads(predictions.read_text())
                doc["predictions"][1][field] = value
                predictions.write_text(json.dumps(doc))
                run = load_run(path, 3)
                self.assertTrue(run["errors"])
                self.assertTrue(np.isnan(run["rows"][1]["gain"]))
        with tempfile.TemporaryDirectory() as directory:
            path, predictions = self.fixture(Path(directory))
            doc = json.loads(predictions.read_text())
            doc["predictions"].append(doc["predictions"][0])
            predictions.write_text(json.dumps(doc))
            run = load_run(path, 3)
            self.assertFalse(run["rows"][0]["prediction_matched"])

    def test_chemistry_thresholds_ties_and_common_horizon_auc(self):
        with tempfile.TemporaryDirectory() as directory:
            path, _ = self.fixture(Path(directory))
            run = load_run(path, 3)
            th = thresholds_for([run])["test_task"]
            self.assertAlmostEqual(th["tau"], np.quantile([.3, .5, .5, .8], .9))
            add_chemistry(run, th)
            frame = pd.DataFrame(run["rows"])
            frame["initial_circles"] = 0
            result = summarize_run(frame, run["inventory"], 2)
            self.assertAlmostEqual(result["best_so_far_auc"], .5)
            self.assertAlmostEqual(result["observed_horizon_auc"], .6)
            self.assertEqual(result["strict_improvements"], 2)
            self.assertEqual(result["longest_no_improvement_streak"], 1)
            self.assertEqual(frame.circles.tolist(), [0, 0, 1])
            self.assertEqual(frame.top10_k.tolist(), [2, 3, 4])
            self.assertEqual(frame.top10_scaffolds.iloc[0], 0)
            self.assertEqual(chemistry("[C][C][O]")[0], chemistry("CCO")[0])

    def test_rolling_calendar_windows_sample_sd_and_bin_counts(self):
        # Unequal horizons and one missing record must not be padded or zero-filled.
        with tempfile.TemporaryDirectory() as directory:
            path, _ = self.fixture(Path(directory))
            run = load_run(path, 3)
            th = thresholds_for([run])["test_task"]
            add_chemistry(run, th)
            base = run["rows"][0]
            rows = []
            for seed, end, error in [(1, 30, 1.), (2, 20, 3.)]:
                for step in range(1, end + 1):
                    rows.append({**base, "seed": seed, "iteration": step,
                                 "squared_error": error ** 2, "error": error, "nn_distance": .2,
                                 "success": np.nan if step == 1 else float(seed == 1)})
            frame = rolling_frame(pd.DataFrame(rows))
            self.assertTrue(np.isnan(frame.query("seed == 1 and iteration == 10").success_rolling.iloc[0]))
            self.assertEqual(frame.query("seed == 1 and iteration == 11").success_rolling.iloc[0], 1)
            stats = curve_statistics(frame, "rmse_rolling")
            self.assertAlmostEqual(stats.loc[10, "mean"], 2)
            self.assertAlmostEqual(stats.loc[10, "sd"], np.sqrt(2))
            self.assertEqual(stats.loc[21, "n"], 1)
            self.assertTrue(np.isnan(stats.loc[21, "sd"]))
            bins = novelty_bins(frame)
            b = bins.query("row_type == 'task' and bin == '[0.2,0.4)'").iloc[0]
            self.assertEqual(b.n_valid, 50)
            self.assertEqual(b.n_seeds, 2)
            self.assertEqual(b.rmse_mean, 2)
            self.assertAlmostEqual(b.rmse_sd, np.sqrt(2))

    def test_output_guard_and_unsupported_batches(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, _ = self.fixture(root)
            with self.assertRaises(ValueError):
                analyze(root, root / "analysis")
            status = json.loads(path.read_text())
            status["solver_configuration"]["batch_size"] = 2
            path.write_text(json.dumps(status))
            run = load_run(path, 3)
            self.assertFalse(run["inventory"]["supported"])
            self.assertFalse(run["rows"])

    def test_seed_summary_aggregation(self):
        result = aggregate_seeds(pd.DataFrame([
            {"task": "t", "seed": 1, "incomplete": False, "gp_rmse": 1.},
            {"task": "t", "seed": 2, "incomplete": True, "gp_rmse": 3.},
        ])).iloc[0]
        self.assertEqual(result.gp_rmse_mean, 2)
        self.assertAlmostEqual(result.gp_rmse_sd, np.sqrt(2))
        self.assertEqual(result.n_complete, 1)

    def test_inventory_missing_seeds_filters_and_read_only_report(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "raw"
            source.mkdir()
            self.fixture(source)
            before = {p: digest_file(p) for p in source.iterdir()}
            with patch("hdbo_benchmark.evaluation.cowboys_analysis.make_plots", return_value=[]):
                result = analyze(source, root / "report", planned_steps=3)
            inventory = pd.read_csv(root / "report" / "run_inventory.csv")
            self.assertEqual(len(inventory), 5)
            self.assertEqual((inventory.status == "missing_run").sum(), 4)
            self.assertEqual(result["validation_error_count"], 0)
            self.assertTrue(result["source_files_unchanged"])
            self.assertEqual(before, {p: digest_file(p) for p in source.iterdir()})
            report = (root / "report" / "report.md").read_text()
            self.assertIn("selection-biased", report)
            self.assertIn("future experiment", report)
            with patch("hdbo_benchmark.evaluation.cowboys_analysis.make_plots", return_value=[]):
                analyze(source, root / "filtered", tasks=["test_task"], seeds=[1], planned_steps=3)
            self.assertEqual(len(pd.read_csv(root / "filtered" / "run_inventory.csv")), 1)

    def test_duplicate_attempts_are_never_independent_seeds(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "raw"
            for attempt in ("one", "two"):
                (source / attempt).mkdir(parents=True)
                self.fixture(source / attempt)
            with self.assertRaisesRegex(ValueError, "No supported, unambiguous"):
                analyze(source, root / "report", planned_steps=3)
            inventory = pd.read_csv(root / "report" / "run_inventory.csv")
            self.assertEqual((inventory.status == "ambiguous_duplicate_attempt").sum(), 2)
            self.assertFalse(inventory.supported.any())

    def test_greedy_centers_accept_exact_distance_boundary(self):
        with tempfile.TemporaryDirectory() as directory:
            path, _ = self.fixture(Path(directory))
            run = load_run(path, 3)
            with patch("hdbo_benchmark.evaluation.cowboys_analysis.distance", return_value=.4):
                add_chemistry(run, {"tau": .1, "epsilon": .01})
            # One initial molecule and three distinct selections, all at boundary.
            self.assertEqual(run["rows"][-1]["circles"], 4)


if __name__ == "__main__":
    unittest.main()
