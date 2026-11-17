"""Compatibility alias for :mod:`hdbo_benchmark.solvers.cowboys_flow`."""

import importlib as _importlib
import sys as _sys

_implementation = _importlib.import_module("hdbo_benchmark.solvers.cowboys_flow")

_sys.modules[__name__] = _implementation
