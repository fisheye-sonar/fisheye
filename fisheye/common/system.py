import shutil
import uuid
from pathlib import Path
from typing import Union

from fisheye.common.exceptions import LowDiskSpaceError


def detect_platform() -> str:
    """Return the recommended platform name for the current hardware.

    Returns one of: 'cuda', 'mps', 'cpu'.
    """
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass

    return "cpu"


def check_disk_space(path: Union[str, Path] = "/", threshold: float = 10.0) -> None:
    """
    Checks if the available disk space is below the threshold. Don't want to run inference and then find out you
    can't save the results due to storage issues.

    Args:
        path (str | Path): Filesystem path to check. Default is root.
        threshold (float): Minimum free space percentage to consider "safe".

    Returns:
        bool: Raise LowDiskSpaceError if disk space is low. Return False otherwise.
    """
    path = Path(path).resolve()
    total, used, free = shutil.disk_usage(path)
    percent_free = (free / total) * 100

    if percent_free < threshold:
        raise LowDiskSpaceError(
            f"Low disk space: Only {percent_free:.2f}% free. Suggest moving unused files to external "
            f"storage."
        )


def generate_job_id() -> str:
    """Generate a unique job ID."""
    return str(uuid.uuid4())
