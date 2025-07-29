import gc
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor
from functools import wraps
from pathlib import Path
from typing import Callable, List, Any

import numpy as np
import structlog
import torch

from fisheye.enums import ValidExtensions, IGNORED_FILE_PREFIXES, IGNORED_DIR_NAMES

logger = structlog.get_logger()


def safe_execution(
    default_return=None, max_retries=1, delay=0, exceptions=(Exception,)
):
    """
    A decorator to catch and log exceptions, with retry logic and exponential backoff.

    Args:
        default_return: Value to return if all retries fail.
        max_retries (int): Number of times to retry before failing.
        delay (float): Initial delay in seconds before retrying. Doubles each retry (exponential backoff).
        exceptions (tuple): Exceptions to catch. Enables fine-grained control over which exceptions the decorator
            handles and retries on.

    Returns:
        Function return value or default_return on failure.
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(1, max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    logger.error(
                        "safe_execution_exception",
                        function=func.__name__,
                        attempt=attempt,
                        error=str(e),
                    )

                    if attempt < max_retries and delay:
                        backoff = delay * (2 ** (attempt - 1))
                        time.sleep(backoff)

            logger.error(
                "safe_execution_failed",
                function=func.__name__,
                error=str(last_exception),
                retries=max_retries,
            )
            return default_return

        return wrapper

    return decorator


def run_with_threads(func: Callable, inputs: List[Any], max_workers: int) -> List[Any]:
    """
    Runs a function across inputs using multithreading.

    Args:
        func (Callable): The function to run.
        inputs (List[Any]): Inputs to run the function on.
        max_workers (int): Number of threads to use.

    Returns:
        List[Any]: List of function outputs.
    """
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        return list(executor.map(func, inputs))


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def cleanup():
    torch.cuda.empty_cache()
    gc.collect()


def _is_valid_file(file_path: Path) -> bool:
    """Check if the file is valid based on extension."""
    return file_path.is_file() and file_path.suffix in {
        e.value for e in ValidExtensions
    }


def _is_valid_dir(dir_path: Path) -> bool:
    """Check if the directory is valid."""
    return dir_path.is_dir()


def get_all_valid_files_in_dir(path: Path) -> List[Path]:
    """Get all valid files in a directory."""
    valid_files = []
    for root, dirs, files in os.walk(path):
        # Skip ignored system directories
        dirs[:] = [
            d for d in dirs if d not in IGNORED_DIR_NAMES and not d.startswith(".")
        ]

        for file in files:
            # Skip files with ignored prefixes
            if file.startswith(".") or any(
                file.startswith(prefix) for prefix in IGNORED_FILE_PREFIXES
            ):
                continue

            file_path = Path(root) / file
            if _is_valid_file(file_path):
                valid_files.append(file_path)

    return valid_files
