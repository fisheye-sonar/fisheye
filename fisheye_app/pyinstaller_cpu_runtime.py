"""PyInstaller runtime hook for CPU-focused frozen builds."""

from __future__ import annotations

import os
import sys

os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "")

if getattr(sys, "frozen", False):
    base = getattr(sys, "_MEIPASS", "")
    if base and sys.platform != "win32":
        existing = os.environ.get("LD_LIBRARY_PATH", "")
        os.environ["LD_LIBRARY_PATH"] = (
            base if not existing else f"{base}{os.pathsep}{existing}"
        )

    from fisheye_app.frozen_preload import preload_shared_libraries

    preload_shared_libraries()

    if base:
        plugins = os.path.join(base, "PySide6", "Qt", "plugins")
        platforms = os.path.join(plugins, "platforms")
        if os.path.isdir(platforms):
            os.environ["QT_PLUGIN_PATH"] = plugins
            os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = platforms
        if sys.platform != "win32":
            os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

