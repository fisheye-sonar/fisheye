"""Preload native libraries before matplotlib/pyexpat in frozen builds."""

from __future__ import annotations

import ctypes
import sys
from pathlib import Path


def preload_shared_libraries() -> None:
    if not getattr(sys, "frozen", False):
        return

    base = Path(getattr(sys, "_MEIPASS", ""))
    if not base.is_dir():
        return

    if sys.platform == "win32":
        patterns = ("libexpat*.dll", "expat.dll")
        loader = ctypes.WinDLL
    else:
        patterns = ("libexpat.so*", "libexpat*.so*")
        loader = lambda path: ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)

    candidates: list[Path] = []
    for pattern in patterns:
        candidates.extend(base.glob(pattern))
    candidates.sort(key=lambda path: len(path.name), reverse=True)

    for path in candidates:
        if not path.is_file():
            continue
        try:
            loader(str(path))
            return
        except OSError:
            continue

