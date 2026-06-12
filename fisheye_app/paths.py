"""Resolve install and development paths for desktop app resources."""

from __future__ import annotations

import os
import sys
from pathlib import Path

APP_SLUG = "fisheye-aris-salmon-detection"


def is_frozen() -> bool:
    return getattr(sys, "frozen", False)


def app_root() -> Path:
    if is_frozen():
        return Path(sys._MEIPASS)  # type: ignore[attr-defined]
    return Path(__file__).resolve().parent.parent


def user_data_dir() -> Path:
    if not is_frozen():
        base = app_root() / "logs"
        base.mkdir(parents=True, exist_ok=True)
        return base

    if sys.platform == "win32":
        local = os.environ.get("LOCALAPPDATA")
        if local:
            base = Path(local) / APP_SLUG
        else:
            base = Path.home() / "AppData" / "Local" / APP_SLUG
    else:
        base = Path.home() / ".local" / "share" / APP_SLUG
    base.mkdir(parents=True, exist_ok=True)
    return base


def crash_log_path() -> Path:
    return user_data_dir() / "crash.log"


def default_weights_dir() -> Path:
    if is_frozen():
        exe_dir = Path(sys.executable).resolve().parent
        bundled = exe_dir / "weights"
        if bundled.is_dir():
            return bundled
    return app_root() / "weights"


def list_bundled_checkpoints() -> list[Path]:
    weights = default_weights_dir()
    if not weights.is_dir():
        return []
    return sorted(weights.glob("*.pt"))


def log_dir() -> Path:
    base = user_data_dir()
    base.mkdir(parents=True, exist_ok=True)
    return base

