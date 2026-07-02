import warnings
from pathlib import Path
from typing import Optional

import tomli

ULTRALYTICS_DETECTOR_TYPES = {"yolov11", "yolov26"}


_APP_VERSION = "1.0.0-beta.5"


def get_app_version_from_pyproject():
    """Get version from pyproject.toml."""
    pyproject_path = Path(__file__).parent.parent / "pyproject.toml"
    if pyproject_path.exists():
        with open(pyproject_path, "rb") as f:
            pyproject = tomli.load(f)

        v = pyproject["tool"]["poetry"].get("version", "")
        return v.lstrip("v") or None

    return None


def _get_app_version() -> Optional[str]:
    # Check installed package metadata
    try:
        from importlib.metadata import PackageNotFoundError
        from importlib.metadata import version as pkg_version

        return pkg_version("fisheye")

    except Exception:
        pass

    # Running from source (local dev and/or editable install without metadata)
    v = get_app_version_from_pyproject()
    if v:
        return v

    return _APP_VERSION


def get_version_from_detector(path: str, detector_type: Optional[str] = None):
    """Get version of object detector."""
    try:
        project_root = Path(__file__).resolve().parents[1]
        model_path = str((project_root / path).resolve())

        if detector_type in ULTRALYTICS_DETECTOR_TYPES:
            from ultralytics import YOLO

            model = YOLO(model_path).model
        else:
            from torch.serialization import SourceChangeWarning
            from yolov5.models.experimental import attempt_load

            warnings.filterwarnings("ignore", category=SourceChangeWarning)
            model = attempt_load(model_path, inplace=True)

    except FileNotFoundError as e:
        raise FileNotFoundError(f"Could not find {path}: {e}")

    return (
        model.fisheye_version
        if hasattr(model, "fisheye_version")
        else Path(model_path).name
    )


__app_version__ = _get_app_version()
