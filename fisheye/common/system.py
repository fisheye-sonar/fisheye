import logging
import shutil
from pathlib import Path

from fisheye.common.exceptions import LowDiskSpaceError

logger = logging.getLogger(__name__)


def check_disk_space(path: str | Path = "/", threshold: float = 10.0) -> None:
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
