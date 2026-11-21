"""Validate sweep selection without loading models or evaluating objectives."""

import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from uuid import uuid4


ROOT = Path(__file__).resolve().parents[1]
GIT_BASH = Path(os.environ.get("ProgramFiles", "C:/Program Files")) / "Git/bin/bash.exe"
BASH = str(GIT_BASH) if GIT_BASH.is_file() else shutil.which("bash")


@unittest.skipUnless(BASH, "Bash is required for the sweep selection checks")
class SweepSelectionTests(unittest.TestCase):
    def dry_run(self, **settings):
        env = os.environ.copy()
        for key in ("RUNS_FILE", "FUNCTIONS", "SEEDS"):
            env.pop(key, None)
        run_id = f"selection_check_{uuid4().hex}"
        env.update(DRY_RUN="1", RUN_ID=run_id, PYTHON_BIN=Path(sys.executable).as_posix())
        env.update(settings)
        result = subprocess.run(
            [BASH, "scripts/experiments/run.sh"], cwd=ROOT, env=env,
            capture_output=True, text=True, timeout=30,
        )
        self.assertFalse((ROOT / "results/cowboys" / run_id).exists())
        return result

    def test_missing_runs_are_exact_pairs_without_cartesian_expansion(self):
        result = self.dry_run(RUNS_FILE="configs/cowboys_pending.tsv")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.splitlines(), [
            "task\tseed", "albuterol_similarity\t1",
            "albuterol_similarity\t2", "albuterol_similarity\t3",
            "albuterol_similarity\t4", "albuterol_similarity\t5",
        ])

    def test_existing_function_seed_cartesian_order_is_preserved(self):
        result = self.dry_run(FUNCTIONS="drd2_docking scaffold_hop", SEEDS="2 4")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.splitlines(), [
            "task\tseed", "drd2_docking\t2", "scaffold_hop\t2",
            "drd2_docking\t4", "scaffold_hop\t4",
        ])

    def test_ambiguous_or_duplicate_plans_are_rejected_before_launch(self):
        result = self.dry_run(RUNS_FILE="configs/cowboys_pending.tsv", SEEDS="1")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Use RUNS_FILE alone", result.stderr)
        with tempfile.TemporaryDirectory() as folder:
            plan = Path(folder) / "duplicate.tsv"
            plan.write_bytes(b"task\tseed\r\nalbuterol_similarity\t1\r\nalbuterol_similarity\t01\r\n")
            result = self.dry_run(RUNS_FILE=plan.as_posix())
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("duplicate task/seed", result.stderr)


if __name__ == "__main__":
    unittest.main()
