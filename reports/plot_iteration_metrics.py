"""Compatibility alias for :mod:`hdbo_benchmark.evaluation.plot_iteration_metrics`."""

import importlib as _importlib
import sys as _sys

_implementation = _importlib.import_module("hdbo_benchmark.evaluation.plot_iteration_metrics")

if __name__ == "__main__":
    raise SystemExit(_implementation.main())
else:
    _sys.modules[__name__] = _implementation
