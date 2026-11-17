"""Offline unit/integration checks; never construct a solver or call an oracle."""

import copy
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from hdbo_benchmark.evaluation.aggregate_runs import aggregate_directories, compatibility, describe
from hdbo_benchmark.evaluation.export_runs import build_run, export_directory, validate_run
from hdbo_benchmark.evaluation.metric_io import digest_file, io_path, read_json, write_json
from hdbo_benchmark.evaluation.molecular_metrics import (
    accepts_representative, compute_metrics, distance, fingerprint, molecule_info, normalized_auc,
)


FIXTURE = Path(__file__).parent / "fixtures/cowboys_checkpoint"
STEM = "cowboys_albuterol_similarity_1"


class MetricTests(unittest.TestCase):
    def test_hand_computed_quality_and_auc_include_initialization(self):
        _, curve, _ = compute_metrics([.1, .4, .8, .2], ["CCO", "CCN", "CCC", "CO"], 2,
                                       {0: 2, 1: 3, 2: 4})
        self.assertEqual([p["bo_step"] for p in curve], [0, 1, 2])
        self.assertEqual([p["quality"]["best_so_far"] for p in curve], [.4, .8, .8])
        self.assertAlmostEqual(curve[0]["quality"]["top_10"], .25)
        self.assertAlmostEqual(curve[1]["quality"]["top_10"], 1.3 / 3)
        self.assertAlmostEqual(curve[2]["quality"]["top_10"], 1.5 / 4)
        self.assertAlmostEqual(normalized_auc(curve, lambda p: p["quality"]["top_1"])[0], .7)
        self.assertAlmostEqual(normalized_auc(curve, lambda p: p["quality"]["top_10"])[0],
                               (.25 / 2 + 1.3 / 3 + .375 / 2) / 2)

    def test_10_plus_300_has_301_states_not_310(self):
        _, curve, _ = compute_metrics([float(i) for i in range(310)], ["CCO"] * 310, 10,
                                       {t: 10 + t for t in range(301)})
        self.assertEqual(len(curve), 301)
        self.assertEqual(curve[0]["quality"]["best_so_far"], 9)
        self.assertEqual(curve[-1]["cumulative_observed_evaluations"], 310)
        self.assertIsNone(curve[-1]["cumulative_oracle_calls"])
        self.assertAlmostEqual(normalized_auc(curve, lambda p: p["quality"]["top_1"])[0], 159)

    def test_duplicates_invalid_nonfinite_and_no_future_leakage(self):
        observations, curve, representatives = compute_metrics(
            [.1, .4, .8, .9, 100, float("nan")],
            ["CCO", "CCN", "OCC", "CCC", "invalid_smiles", "C"], 2,
            {t: 2 + t for t in range(5)}, theta=.8, delta=1.)
        self.assertEqual(curve[0]["quality"]["top_1"], .4)
        self.assertEqual(curve[0]["diversity_by_profile"]["primary"]["unique_hits"], 0)
        self.assertTrue(observations[2]["is_duplicate"])
        self.assertFalse(observations[4]["valid_molecule"])
        self.assertIsNone(observations[5]["raw_oracle_score"])
        self.assertEqual(curve[-1]["quality"]["best_so_far"], .9)
        self.assertAlmostEqual(curve[2]["quality"]["top_10"], (.8 + .4 + .9) / 3)
        self.assertEqual(curve[-1]["diversity_by_profile"]["primary"]["unique_hits"], 2)
        self.assertEqual(curve[-1]["diversity_by_profile"]["primary"]["online_diverse_hits"], 1)
        self.assertEqual(representatives[0]["observation_index"], 2)
        for point in curve:
            self.assertEqual(point["quality"]["top_1"], point["quality"]["best_so_far"])
        json.dumps(observations, allow_nan=False)

    def test_tanimoto_and_strict_greedy_boundary(self):
        a = fingerprint(molecule_info("[Na+]")["mol"])
        b = fingerprint(molecule_info("[Cl-]")["mol"])
        self.assertEqual(distance(a, a), 0)
        self.assertEqual(distance(a, b), 1)
        self.assertFalse(accepts_representative(a, [a], 0))
        self.assertFalse(accepts_representative(b, [a], 1))
        self.assertTrue(accepts_representative(b, [a], .7))
        self.assertTrue(accepts_representative(a, [], 1))

    def test_threshold_initial_hits_scaffolds_and_first_crossing(self):
        _, curve, reps = compute_metrics([.5, .5, .6, .5], ["CCO", "c1ccccc1", "OCC", "C1CCCCC1"],
                                         2, {0: 2, 1: 3, 2: 4}, theta=.5, delta=0)
        self.assertEqual(curve[0]["diversity_by_profile"]["primary"]["unique_hits"], 2)
        self.assertEqual(curve[0]["diversity_by_profile"]["primary"]["distinct_hit_scaffolds"], 1)
        self.assertEqual(curve[-1]["diversity_by_profile"]["primary"]["distinct_hit_scaffolds"], 2)
        self.assertEqual(len(reps), 3)
        self.assertEqual(reps[0]["bo_step"], 0)
        self.assertEqual(reps[0]["score_at_acceptance"], .5)
        counts = [p["diversity_by_profile"]["primary"]["online_diverse_hits"] for p in curve]
        self.assertEqual(counts, sorted(counts))

    def test_missing_structures_empty_and_zero_horizon(self):
        observations, curve, _ = compute_metrics([.2, .6], None, 1, {0: 1, 1: 2})
        self.assertEqual(curve[-1]["quality"]["best_so_far"], .6)
        self.assertIsNone(curve[-1]["quality"]["top_1"])
        self.assertIsNone(observations[0]["valid_molecule"])
        _, empty, _ = compute_metrics([], [], 0, {0: 0})
        self.assertIsNone(empty[0]["quality"]["best_so_far"])
        self.assertIsNone(normalized_auc(empty, lambda p: 0)[0])
        self.assertIsNotNone(normalized_auc(empty, lambda p: 0)[1])
        _, invalid, _ = compute_metrics([float("inf")], ["bad"], 1, {0: 1})
        self.assertIsNone(invalid[0]["top_10_internal_diversity"])

    def test_minimization_is_explicit_posthoc_negation(self):
        observations, curve, _ = compute_metrics([3, 2, 1], ["C", "CC", "CCC"], 1,
                                                 {0: 1, 1: 2, 2: 3}, direction="minimize", theta=-2)
        self.assertEqual([p["quality"]["best_so_far"] for p in curve], [-3, -2, -1])
        self.assertEqual(observations[0]["raw_oracle_score"], 3)
        self.assertEqual(observations[0]["score_for_metrics"], -3)

    def test_completed_batch_boundaries_and_missing_grid_auc(self):
        observations, curve, _ = compute_metrics([.1, .2, .3, .4, .5], ["C", "CC", "CCC", "CO", "CN"],
                                                 1, {0: 1, 1: 3, 2: 5})
        self.assertEqual([p["cumulative_observed_evaluations"] for p in curve], [1, 3, 5])
        self.assertEqual([e["bo_step"] for e in observations], [0, 1, 1, 2, 2])
        value, reason = normalized_auc([curve[0], curve[2]], lambda p: p["quality"]["top_1"])
        self.assertIsNone(value)
        self.assertIn("every state", reason)

    def test_nonfinite_configuration_rejected_and_large_finite_auc_supported(self):
        for kwargs in ({"theta": float("nan")}, {"delta": float("inf")}, {"delta": -.1},
                       {"delta": True}, {"direction": "unknown"}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                compute_metrics([], [], 0, {}, **kwargs)
        _, curve, _ = compute_metrics([1e308, 1e308], ["CCO", "CCN"], 1, {0: 1, 1: 2})
        self.assertEqual(normalized_auc(curve, lambda p: p["quality"]["top_1"])[0], 1e308)


class FixtureCase(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "group"
        shutil.copytree(FIXTURE, self.root)
        self.status_path = self.root / f"{STEM}_status.json"

    def run_fixture(self, **kwargs):
        return build_run(self.status_path, **kwargs)


class ArtifactTests(FixtureCase):
    def test_real_export_is_repeatable_input_preserving_and_schema_valid(self):
        originals = {p.name: p.read_bytes() for p in self.root.iterdir() if p.is_file()}
        manifest = export_directory(self.root, planned_bo_steps=1)
        self.assertFalse(manifest["issues"])
        output = Path(manifest["runs"][0]["path"])
        first = io_path(output).read_bytes()
        export_directory(self.root, planned_bo_steps=1)
        self.assertEqual(first, io_path(output).read_bytes())
        self.assertEqual(originals, {p.name: p.read_bytes() for p in self.root.iterdir() if p.is_file()})
        run = read_json(output)
        validate_run(run)
        import jsonschema
        from hdbo_benchmark.evaluation import export_runs
        jsonschema.validate(run, read_json(Path(export_runs.__file__).with_name("run_schema.json")))
        self.assertEqual(len(run["evaluations"]), 11)
        self.assertEqual(len(run["trajectory"]), 2)
        self.assertEqual(run["status"], "completed")
        self.assertEqual(run["summary"]["auc_scope"], "full_budget")
        self.assertAlmostEqual(run["summary"]["final"]["best_so_far"], .32941176470588235)
        self.assertAlmostEqual(run["trajectory"][0]["quality"]["top_10"], .2477345817466027)
        self.assertIsNone(run["summary"]["n_oracle_calls_total"])
        self.assertIsNone(run["trajectory"][0]["diversity_by_profile"]["primary"]["unique_hits"])
        self.assertIn("recorded_acquisition", run["evaluations"][-1])
        for source in run["provenance"]["sources"]:
            self.assertEqual(source["sha256"], digest_file(Path(source["path"])))

    def test_unverified_partial_failed_and_status_only(self):
        unknown = self.run_fixture()
        self.assertEqual(unknown["summary"]["auc_scope"], "unverified_budget")
        partial = self.run_fixture(planned_bo_steps=300)
        self.assertEqual(partial["status"], "partial")
        self.assertEqual(partial["summary"]["auc_scope"], "partial")
        self.assertIsNotNone(partial["summary"]["auc"]["top_1"])
        status = read_json(self.status_path)
        status.update(status="failed", completed_iterations=2)
        write_json(self.status_path, status)
        mismatch = self.run_fixture(planned_bo_steps=300)
        self.assertEqual(mismatch["summary"]["n_completed_bo_steps"], 1)
        self.assertTrue(any("disagree" in warning for warning in mismatch["warnings"]))
        (self.root / f"{STEM}_history.npz").unlink()
        (self.root / f"{STEM}_iteration_results.json").unlink()
        failed = self.run_fixture(planned_bo_steps=300)
        self.assertEqual(failed["status"], "failed")
        self.assertEqual(failed["trajectory"], [])
        self.assertEqual(failed["evaluations"], [])
        self.assertIsNone(failed["summary"]["auc"]["top_1"])
        self.assertIsNotNone(failed["summary"]["final"]["best_so_far"])

    def test_scores_only_no_original_structure_reconstruction(self):
        with np.load(self.root / f"{STEM}_history.npz") as source:
            payload = {key: source[key] for key in source.files if key != "structures"}
        np.savez(self.root / f"{STEM}_history.npz", **payload)
        run = self.run_fixture(planned_bo_steps=1)
        self.assertIsNotNone(run["summary"]["auc"]["best_so_far"])
        self.assertIsNone(run["summary"]["auc"]["top_10"])
        self.assertIn("molecular_metrics", run["missing_reason"])
        self.assertTrue(all(e["canonical_smiles"] is None for e in run["evaluations"]))

    def test_iteration_only_retains_observations_without_inventing_initialization(self):
        (self.root / f"{STEM}_history.npz").unlink()
        run = self.run_fixture(planned_bo_steps=1)
        self.assertEqual(len(run["evaluations"]), 1)
        observation = run["evaluations"][0]
        self.assertEqual(observation["bo_step"], 1)
        self.assertEqual(observation["observation_index"], 10)
        self.assertIsNotNone(observation["canonical_smiles"])
        self.assertIsNone(observation["is_duplicate"])
        self.assertEqual(run["trajectory"], [])
        self.assertIsNone(run["design"]["initial_content_sha256"])
        validate_run(run)
        status = read_json(self.status_path)
        status.pop("solver_configuration")
        write_json(self.status_path, status)
        legacy = self.run_fixture()
        self.assertIsNone(legacy["evaluations"][0]["observed_structure"])
        self.assertIsNotNone(legacy["evaluations"][0]["raw_oracle_score"])

    def test_scope_excludes_preflight_and_reports_failures_without_checkpoint(self):
        shutil.copytree(FIXTURE, self.root / "preflight")
        manifest = export_directory(self.root, recursive=True)
        self.assertEqual(len(manifest["runs"]), 1)
        (self.root / "sweep_status.tsv").write_text("task\tseed\texit_code\nmissing_task\t2\t1\n")
        manifest = export_directory(self.root)
        self.assertEqual(len(manifest["issues"]), 1)
        self.assertIn("No checkpoint", manifest["issues"][0]["missing_reason"])

    def test_missing_boundaries_and_malformed_status_are_explicit(self):
        status = read_json(self.status_path)
        status.pop("initial_history_size")
        write_json(self.status_path, status)
        run = self.run_fixture()
        self.assertEqual(run["trajectory"], [])
        self.assertTrue(all(e["phase"] is None and e["bo_step"] is None for e in run["evaluations"]))
        write_json(self.status_path, [])
        manifest = export_directory(self.root)
        self.assertFalse(manifest["runs"])
        self.assertEqual(len(manifest["issues"]), 1)

    def test_unsupported_task_config_and_corrupt_structures_are_not_silent(self):
        with self.assertRaises(ValueError):
            self.run_fixture(task_config={"theta": .5})
        with np.load(self.root / f"{STEM}_history.npz") as source:
            payload = {key: source[key] for key in source.files}
        payload["structures"] = payload["structures"][:-1]
        np.savez(self.root / f"{STEM}_history.npz", **payload)
        run = self.run_fixture()
        self.assertTrue(any("length mismatch" in warning for warning in run["warnings"]))
        self.assertIsNone(run["summary"]["final"]["top_10"])
        self.assertIsNotNone(run["summary"]["final"]["best_so_far"])

    def test_invalid_schema_and_tampered_auc_rejected(self):
        run = self.run_fixture()
        run["summary"]["auc"]["top_1"] = 100
        with self.assertRaises(ValueError):
            validate_run(run)
        run = self.run_fixture()
        run["status"] = "fabricated"
        with self.assertRaises(ValueError):
            validate_run(run)

    def test_export_never_imports_solver_or_objective(self):
        code = """
import sys
from pathlib import Path
from hdbo_benchmark.evaluation.export_runs import build_run
build_run(Path(sys.argv[1]))
assert not {'torch', 'poli', 'botorch', 'gpytorch'} & sys.modules.keys()
assert not any(n.startswith('hdbo_benchmark.solvers') for n in sys.modules)
"""
        result = subprocess.run([sys.executable, "-c", code, str(self.status_path)],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


class AggregationTests(FixtureCase):
    def test_aggregation_independent_seeds_and_no_padding(self):
        run = self.run_fixture(planned_bo_steps=1, task_config={"oracle_version": "fixture-runtime", "hit_threshold": .3,
                                                              "initialization_policy": "fixture_random_10"})
        for seed, value in ((1, .3), (2, .5)):
            other = copy.deepcopy(run)
            other["run_id"] = f"run-{seed}"
            other["design"]["seed"] = seed
            for point in other["trajectory"]:
                point["quality"] = {key: value for key in point["quality"]}
            for key in other["summary"]["auc"]:
                other["summary"]["auc"][key] = value
                other["summary"]["final"][key] = value
            write_json(self.root / f"metrics/seed{seed}/run.json", other)
        partial = copy.deepcopy(run)
        partial["run_id"] = "partial"
        partial["design"]["seed"] = 3
        partial["status"] = "partial"
        partial["summary"].update(full_budget_completed=False, auc_scope="partial")
        partial["trajectory"] = partial["trajectory"][:1]
        partial["summary"].update(auc_horizon=0, n_completed_bo_steps=0)
        partial["summary"]["final"].update(partial["trajectory"][0]["quality"])
        for key in partial["summary"]["auc"]:
            partial["summary"]["auc"][key] = None
        for key in ("unique_hits", "online_diverse_hits", "distinct_hit_scaffolds"):
            partial["summary"]["diversity_by_profile"]["primary"][f"auc_{key}"] = None
        write_json(self.root / "metrics/partial/run.json", partial)
        result = aggregate_directories([self.root])
        row = next(r for r in result["metric_summary_rows"] if r["metric"] == "final_top_10")
        self.assertEqual(row["n"], 2)
        self.assertAlmostEqual(row["mean"], .4)
        self.assertAlmostEqual(row["sample_sd"], .1414213562373095)
        self.assertEqual((row["ci95_low"], row["ci95_high"]), (.3, .5))
        excluded = next(r for r in result["run_summary_rows"] if r["run_id"] == "partial")
        self.assertFalse(excluded["included_in_full_budget_statistics"])
        self.assertTrue(all(r["n"] == 2 for r in result["trajectory_summary_rows"] if r["metric"] == "top_1"))

    def test_incompatible_protocols_are_partitioned_and_repeated_seeds_excluded(self):
        base = self.run_fixture(planned_bo_steps=1, task_config={"oracle_version": "fixture-runtime",
                                                               "initialization_policy": "fixture_random_10"})
        for index, threshold in enumerate((.3, .3, .5)):
            run = copy.deepcopy(base)
            run["run_id"] = str(index)
            run["metric_policy"]["diversity_profiles"]["primary"]["hit_threshold"] = threshold
            write_json(self.root / f"metrics/{index}/run.json", run)
        result = aggregate_directories([self.root], plots=False)
        self.assertEqual(len(result["compatibility_groups"]), 2)
        self.assertEqual(sum(r["included_in_full_budget_statistics"] for r in result["run_summary_rows"]), 1)
        for field, replacement in (("bo_steps_planned", 300), ("initialization_policy", "different")):
            other = copy.deepcopy(base)
            other["design"][field] = replacement
            self.assertNotEqual(compatibility(base), compatibility(other))

    def test_single_seed_has_no_invented_uncertainty(self):
        row = describe([.5])
        self.assertEqual(row["n"], 1)
        self.assertIsNone(row["sample_sd"])
        self.assertIsNone(row["ci95_low"])

    def test_unknown_oracle_is_reported_and_real_plot_created(self):
        manifest = export_directory(self.root, planned_bo_steps=1)
        result = aggregate_directories([self.root])
        self.assertFalse(result["run_summary_rows"][0]["included_in_full_budget_statistics"])
        self.assertIn("oracle_version", result["run_summary_rows"][0]["exclusion_reason"])
        generated = [p for p in result["plot_manifest"] if p["status"] == "generated"]
        self.assertTrue(generated)
        self.assertTrue(all(io_path(Path(p["path"])).exists() for p in generated))
        self.assertTrue(io_path(self.root / "metrics/aggregate/experiment_summary.json").exists())
        self.assertEqual(len(manifest["runs"]), 1)


if __name__ == "__main__":
    unittest.main()
