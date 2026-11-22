"""Distribution checks on saved toy scores, never objectives or BO runs."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from hdbo_benchmark.evaluation.cowboys_explainer import (
    create_explainer, evaluated_scores, plain_metrics, score_histograms,
)
from hdbo_benchmark.evaluation.cowboys_analysis import load_run, thresholds_for, add_chemistry


class ExplainerTests(unittest.TestCase):
    def test_histogram_counts_initial_points_and_missing_values(self):
        d = pd.DataFrame({"task": ["t"] * 4, "seed": [1] * 4,
                          "iteration": [1, 2, 101, 201], "y": [0., 1., np.nan, .5],
                          "canonical_smiles": ["C"] * 4})
        runs = [{"inventory": {"task": "t", "seed": 1}, "initial": [("C", .25, None, None)]}]
        all_scores = evaluated_scores(d, runs)
        absolute, seeds, tasks, dist = score_histograms(all_scores, d, ["t"], n_bins=2)
        self.assertEqual(len(all_scores), 5)
        self.assertEqual(absolute["count"].sum(), 4)
        self.assertEqual(absolute.query("phase == 'initial'")["count"].sum(), 1)
        self.assertEqual(dist.iloc[0].n_missing, 1)
        self.assertEqual(dist.iloc[0].n_bo, 3)
        early = seeds.query("stage == 'early'")
        np.testing.assert_array_equal(early["count"], [1, 1])
        self.assertEqual(early.percent.sum(), 100)
        middle = seeds.query("stage == 'middle'")
        self.assertTrue(middle.percent.isna().all())
        self.assertTrue(tasks.query("stage == 'middle'").mean_percent.isna().all())

    def test_seed_percentages_are_not_length_weighted(self):
        d = pd.DataFrame({"task": ["t"] * 4, "seed": [1, 1, 1, 2],
                          "iteration": [1, 2, 3, 1], "y": [0., 0., 0., 1.],
                          "canonical_smiles": ["C"] * 4})
        all_scores = evaluated_scores(d, [])
        _, _, tasks, _ = score_histograms(all_scores, d, ["t"], n_bins=2)
        early = tasks.query("stage == 'early'")
        np.testing.assert_array_equal(early.mean_percent, [50, 50])
        np.testing.assert_allclose(early.sd_percent, [np.sqrt(5000)] * 2)
        self.assertTrue((early.n_seeds == 2).all())
        self.assertTrue((tasks.query("stage == 'late'").n_seeds == 0).all())

    def test_plain_metrics_chronology_and_small_report(self):
        from tests.test_cowboys_analysis import AnalysisTests
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output = root / "raw", root / "out"
            source.mkdir()
            (output / "tables").mkdir(parents=True)
            path, _ = AnalysisTests().fixture(source)
            run = load_run(path, 3)
            add_chemistry(run, thresholds_for([run])["test_task"])
            frame = pd.DataFrame(run["rows"])
            frame["initial_circles"] = 0
            seeds, _ = plain_metrics(frame)
            self.assertAlmostEqual(seeds.iloc[0].record_wins_per100, 200 / 3)
            self.assertAlmostEqual(seeds.iloc[0].new_quality_centers_per100, 100 / 3)
            self.assertAlmostEqual(seeds.iloc[0].high_quality_per100, 100 / 3)
            with patch("matplotlib.figure.Figure.savefig") as save:
                result = create_explainer(frame, [run], output, ["test_task", "missing_task"], {})
            self.assertEqual(save.call_count, 6)
            self.assertEqual(len(result["figures"]), 3)
            self.assertEqual(result["n_valid_evaluations"], 4)
            text = (output / "insights.md").read_text()
            self.assertIn("rejected", text.lower())
            self.assertIn("selection-biased", text)


if __name__ == "__main__":
    unittest.main()
