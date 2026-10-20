"""
Work around a SciPy wheel incompatibility with newer macOS dyld.

SciPy 1.15.x wheels contain several Fortran-compiled extensions (PROPACK,
COBYLA, VODE/DOP853/LSODA) whose Mach-O binaries are rejected by newer
macOS versions (confirmed on Darwin 27) due to a malformed
__DATA,__thread_bss section. This causes SciPy imports to fail even though
FishEye and FilterPy do not use these solver backends.

FishEye currently uses Python 3.10, for which SciPy 1.15.3 is the latest
supported release, so upgrading SciPy alone cannot resolve the issue.

This module tests each affected extension before SciPy imports it. If the
extension loads successfully, nothing is changed. If it fails to load, a
minimal stub is installed so SciPy's eager imports can complete.

IMPORTANT:
The stubbed solver functions raise RuntimeError if called. If FishEye
begins using PROPACK, COBYLA, VODE, DOP853, or LSODA, do not extend this
workaround. Replace it with a working SciPy build instead.

TODO (madivanhorn): Remove this module and its call from fisheye/__init__.py once supported
SciPy wheels load these extensions successfully on affected macOS versions.

Upstream: https://github.com/scipy/scipy
Tests: tests/test_macos_scipy_compat.py
"""

import importlib.machinery
import importlib.util
import os
import sys
import types

import numpy as np

# Sentinel marking an attribute that's a *callable* nothing should ever
# invoke (as opposed to `types.SimpleNamespace(...)` below, which is a
# real value some of these modules' own import-time code legitimately
# reads before any of this runs).
_UNUSED = object()

# (module name, {attribute: placeholder value}) that fisheye's dependency
# chain would touch if the real extension loaded (verified against
# scipy 1.15.x source, not guessed). _vode/_dop/_lsoda's
# `types.intvar.dtype` access happens unconditionally at import time
# (scipy/integrate/_ode.py), so it needs a real dtype-bearing object
# rather than a loud failure.
_KNOWN_BROKEN = [
    (
        "scipy.sparse.linalg._propack._spropack",
        {"slansvd": _UNUSED, "slansvd_irl": _UNUSED},
    ),
    (
        "scipy.sparse.linalg._propack._dpropack",
        {"dlansvd": _UNUSED, "dlansvd_irl": _UNUSED},
    ),
    (
        "scipy.sparse.linalg._propack._cpropack",
        {"clansvd": _UNUSED, "clansvd_irl": _UNUSED},
    ),
    (
        "scipy.sparse.linalg._propack._zpropack",
        {"zlansvd": _UNUSED, "zlansvd_irl": _UNUSED},
    ),
    ("scipy.optimize._cobyla", {"minimize": _UNUSED}),
    (
        "scipy.integrate._vode",
        {"types": types.SimpleNamespace(intvar=np.zeros(1, dtype=np.int32))},
    ),
    (
        "scipy.integrate._dop",
        {"types": types.SimpleNamespace(intvar=np.zeros(1, dtype=np.int32))},
    ),
    (
        "scipy.integrate._lsoda",
        {"types": types.SimpleNamespace(intvar=np.zeros(1, dtype=np.int32))},
    ),
]


def _loud_failure(module_name: str, attr_name: str):
    """A stand-in that fails immediately and explains itself, instead of
    silently returning None and letting the real bug surface later as an
    unrelated-looking TypeError/AttributeError far from this file."""

    def _raise(*_args, **_kwargs):
        raise RuntimeError(
            f"{module_name}.{attr_name} was called, but it's stubbed out by "
            f"fisheye/_macos_scipy_compat.py - a workaround for a macOS dyld "
            f"bug that breaks this compiled extension (see that file's "
            f"docstring). It was stubbed because nothing called it at the "
            f"time this workaround was written but something does now. This "
            f"needs a real fix (a scipy build that doesn't carry the bug), "
            f"not a wider stub."
        )

    return _raise


def _loads(module_name: str) -> bool:
    """Whether the named extension can actually be dlopen'd. Loaded by
    file path in isolation rather than via importlib.import_module(name)
    - importing the dotted name would first run the parent package's
    __init__.py (e.g. scipy.sparse.linalg), which is itself the chain
    that crashes, so a clean ImportError would never reach us to catch."""
    import scipy  # safe: scipy's own top-level __init__ doesn't touch the broken submodules

    rel_parts = module_name.split(".")[1:]
    base_dir = os.path.join(os.path.dirname(scipy.__file__), *rel_parts[:-1])
    leaf = rel_parts[-1]
    for suffix in importlib.machinery.EXTENSION_SUFFIXES:
        path = os.path.join(base_dir, leaf + suffix)
        if os.path.exists(path):
            break
    else:
        return False

    try:
        loader = importlib.machinery.ExtensionFileLoader(module_name, path)
        spec = importlib.util.spec_from_loader(module_name, loader)
        loader.exec_module(importlib.util.module_from_spec(spec))
        return True
    except ImportError:
        return False


def apply() -> None:
    if sys.platform != "darwin":
        return
    for module_name, attrs in _KNOWN_BROKEN:
        if _loads(module_name):
            continue  # real extension loaded fine on this machine - leave it alone
        stub = types.ModuleType(module_name)
        for attr_name, value in attrs.items():
            if value is _UNUSED:
                value = _loud_failure(module_name, attr_name)
            setattr(stub, attr_name, value)
        sys.modules[module_name] = stub
