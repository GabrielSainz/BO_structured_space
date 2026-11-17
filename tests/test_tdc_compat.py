"""Regression checks for repeated PMO initialization and hidden TDC errors."""

import importlib
import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from hdbo_benchmark.utils.tdc_compat import (
    ensure_tdc_is_available,
    expose_tdc_evaluation_errors,
    patch_broken_tdc_chem_utils,
)


class FakeOracle:
    """Reproduce TDC's different scalar/list exception handling."""

    def __init__(self):
        self.num_called = 0
        self.name = "test"

    def evaluate(self, smiles):
        if smiles == "broken_model":
            raise ValueError("incompatible model")
        return 0.25 if smiles == "CCO" else 0.0

    def __call__(self, smiles):
        if isinstance(smiles, list):
            self.num_called += len(smiles)
            return [self.evaluate(s) for s in smiles]
        self.num_called += 1
        try:
            return self.evaluate(smiles)
        except ValueError:
            return 0.0


class TDCCompatibilityTests(unittest.TestCase):
    def test_fallback_package_supports_repeated_initialization(self):
        with tempfile.TemporaryDirectory() as folder, patch.dict(sys.modules):
            package = Path(folder) / "tdc"
            oracle = package / "chem_utils" / "oracle"
            oracle.mkdir(parents=True)
            (package / "__init__.py").write_text("raise ImportError('broken exports')\n")
            (package / "chem_utils" / "__init__.py").write_text("")
            (oracle / "__init__.py").write_text("")
            (oracle / "oracle.py").write_text("def example(): return 1\n")
            (oracle / "filter.py").write_text("class MolFilter: pass\n")
            (package / "oracles.py").write_text("class Oracle: pass\n")
            for name in list(sys.modules):
                if name == "tdc" or name.startswith("tdc."):
                    sys.modules.pop(name)
            sys.path.insert(0, folder)
            importlib.invalidate_caches()
            try:
                patch_broken_tdc_chem_utils()
                first = importlib.import_module("tdc")
                ensure_tdc_is_available()
                patch_broken_tdc_chem_utils()
                self.assertIs(importlib.import_module("tdc"), first)
                for name in ["tdc", "tdc.chem_utils", "tdc.chem_utils.oracle"]:
                    self.assertIsNotNone(importlib.util.find_spec(name))
                self.assertEqual(importlib.import_module("tdc.chem_utils").example(), 1)
            finally:
                sys.path.remove(folder)
                importlib.invalidate_caches()

    def test_model_failure_is_exposed_without_extra_evaluations(self):
        original = FakeOracle()
        black_box = types.SimpleNamespace(inner_function=types.SimpleNamespace(oracle=original))
        self.assertTrue(expose_tdc_evaluation_errors(black_box))
        wrapped = black_box.inner_function.oracle
        with self.assertRaisesRegex(ValueError, "incompatible model"):
            wrapped("broken_model")
        self.assertEqual(original.num_called, 1)
        self.assertEqual(wrapped.name, original.name)
        self.assertFalse(expose_tdc_evaluation_errors(black_box))

    def test_valid_scores_and_invalid_defaults_are_preserved(self):
        original = FakeOracle()
        black_box = types.SimpleNamespace(inner_function=types.SimpleNamespace(oracle=original))
        expose_tdc_evaluation_errors(black_box)
        wrapped = black_box.inner_function.oracle
        self.assertEqual(wrapped("CCO"), 0.25)
        self.assertEqual(wrapped("invalid"), 0.0)
        self.assertEqual(wrapped(["CCO", "invalid"]), [0.25, 0.0])
        self.assertEqual(original.num_called, 4)
        wrapped.num_called = 0
        self.assertEqual(original.num_called, 0)
        wrapped("CCO")
        self.assertEqual(wrapped.num_called, 1)

    def test_non_tdc_black_box_is_not_modified(self):
        black_box = types.SimpleNamespace()
        self.assertFalse(expose_tdc_evaluation_errors(black_box))
        self.assertFalse(hasattr(black_box, "inner_function"))


if __name__ == "__main__":
    unittest.main()
