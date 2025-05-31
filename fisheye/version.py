from pathlib import Path

import tomli
from yolov5.models.experimental import attempt_load

from fisheye.common.generic import load_model_config


def get_app_version_from_pyproject():
    """Get version from pyproject.toml."""
    pyproject_path = Path(__file__).parent.parent / "pyproject.toml"
    with open(pyproject_path, "rb") as f:
        pyproject = tomli.load(f)

    return pyproject["tool"]["poetry"]["version"]


def get_version_from_detector(path: str):
    """Get version of object detector."""
    try:
        project_root = Path(__file__).resolve().parents[1]
        model_path = str((project_root / path).resolve())
        model = attempt_load(model_path, inplace=True)

    except FileNotFoundError as e:
        raise FileNotFoundError(f"Could not find {path}: {e}")

    return model.fisheye_version if hasattr(model, "fisheye_version") else None


__app_version__ = get_app_version_from_pyproject()
__detector_version__ = get_version_from_detector(
    load_model_config()["detector"]["path"]
)
