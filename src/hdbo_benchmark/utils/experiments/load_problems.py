"""Compatibility alias for :mod:`hdbo_benchmark.benchmarks.load_problems`."""

import importlib as _importlib
import sys as _sys

_implementation = _importlib.import_module("hdbo_benchmark.benchmarks.load_problems")

_sys.modules[__name__] = _implementation
