"""Lightweight layout checks: python -m unittest tests.test_structure."""

import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class StructureTests(unittest.TestCase):
    def test_namespace_imports_do_not_load_scientific_libraries(self):
        code = """
import sys
import hdbo_benchmark
import hdbo_benchmark.solvers
import hdbo_benchmark.benchmarks
import hdbo_benchmark.evaluation
assert not {'torch', 'poli', 'botorch', 'matplotlib'} & sys.modules.keys()
"""
        result = subprocess.run(
            [sys.executable, "-c", "import sys; sys.path.insert(0, 'src');" + code],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_local_methods_are_packaged(self):
        for name in (
            "cowboys", "cowboys_flow", "cowboys_flow_2",
            "cowboys_diffusion", "latent_space_bo",
        ):
            with self.subTest(method=name):
                self.assertTrue(
                    (ROOT / "src" / "hdbo_benchmark" / "solvers" / f"{name}.py").is_file()
                )

    def test_shell_launchers_reference_existing_runner(self):
        self.assertTrue((ROOT / "run.py").is_file())
        scripts = list((ROOT / "scripts" / "experiments").glob("run*.sh"))
        scripts.extend((ROOT / "scripts" / "slurm").glob("batch*.sh"))
        self.assertTrue(scripts)
        for script in scripts:
            with self.subTest(script=script.name):
                self.assertIn("python run.py", script.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
