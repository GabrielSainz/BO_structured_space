"""Compatibility alias for :mod:`hdbo_benchmark.experiments.run`."""

import importlib as _importlib
import sys as _sys

_implementation = _importlib.import_module("hdbo_benchmark.experiments.run")

if __name__ == "__main__":
    _implementation.main()
else:
    _sys.modules[__name__] = _implementation
