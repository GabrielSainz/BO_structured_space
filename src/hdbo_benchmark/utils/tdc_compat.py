from __future__ import annotations

import importlib
import importlib.util
import sys
import types
from pathlib import Path


def ensure_tdc_is_available() -> None:
    """Raise a helpful error before poli falls back to Conda isolation.

    PMO tasks import `tdc` directly in-process when it is available. If it is
    missing, poli tries a Conda-based fallback that is confusing on systems
    such as Colab where Conda is not present.
    """

    if importlib.util.find_spec("tdc") is None:
        raise RuntimeError(
            "PMO tasks require the `tdc` module in the active Python "
            "environment. Install it with "
            '`python -m pip install PyTDC "huggingface_hub<1" "rdkit<2024.03"` '
            'or install this repo with `python -m pip install -e ".[tdc]"`.'
        )


def patch_broken_tdc_chem_utils() -> None:
    """Work around broken `tdc.chem_utils` package exports.

    Some TDC releases ship a `tdc.chem_utils.__init__` that re-exports names
    missing from `tdc.chem_utils.oracle.oracle`, which breaks PMO oracle
    creation before we even start optimization. For the PMO tasks in this repo
    we only need the oracle callables, so we construct a minimal
    `tdc.chem_utils` module from the underlying oracle implementation when the
    upstream package import fails.
    """

    spec = importlib.util.find_spec("tdc")
    if spec is None or not spec.submodule_search_locations:
        return

    tdc_root = Path(next(iter(spec.submodule_search_locations))).resolve()
    should_patch = False

    try:
        importlib.import_module("tdc")
    except ModuleNotFoundError as exc:
        if exc.name == "huggingface_hub":
            should_patch = True
        else:
            raise
    except ImportError:
        should_patch = True

    if not should_patch:
        try:
            importlib.import_module("tdc.chem_utils")
            return
        except ImportError as exc:
            message = str(exc)
            if (
                "rmsd" in message
                or "kabsch_rmsd" in message
                or "smina" in message
            ):
                should_patch = True
            else:
                raise

    if not should_patch:
        return

    for name in list(sys.modules):
        if name == "tdc" or name.startswith("tdc."):
            sys.modules.pop(name, None)

    tdc_module = types.ModuleType("tdc")
    tdc_module.__file__ = str(tdc_root / "__init__.py")
    tdc_module.__package__ = "tdc"
    tdc_module.__path__ = [str(tdc_root)]
    sys.modules["tdc"] = tdc_module

    chem_utils_dir = tdc_root / "chem_utils"
    oracle_dir = chem_utils_dir / "oracle"

    chem_utils_module = types.ModuleType("tdc.chem_utils")
    chem_utils_module.__file__ = str(chem_utils_dir / "__init__.py")
    chem_utils_module.__package__ = "tdc.chem_utils"
    chem_utils_module.__path__ = [str(chem_utils_dir)]
    sys.modules["tdc.chem_utils"] = chem_utils_module

    oracle_pkg = types.ModuleType("tdc.chem_utils.oracle")
    oracle_pkg.__file__ = str(oracle_dir / "__init__.py")
    oracle_pkg.__package__ = "tdc.chem_utils.oracle"
    oracle_pkg.__path__ = [str(oracle_dir)]
    sys.modules["tdc.chem_utils.oracle"] = oracle_pkg

    oracle_module = importlib.import_module("tdc.chem_utils.oracle.oracle")
    filter_module = importlib.import_module("tdc.chem_utils.oracle.filter")

    for name in dir(oracle_module):
        if not name.startswith("_"):
            setattr(chem_utils_module, name, getattr(oracle_module, name))

    if hasattr(filter_module, "MolFilter"):
        chem_utils_module.MolFilter = filter_module.MolFilter

    oracles_module = importlib.import_module("tdc.oracles")
    tdc_module.Oracle = oracles_module.Oracle
