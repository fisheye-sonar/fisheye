import gc
import random
import traceback
from concurrent.futures import ThreadPoolExecutor
from functools import wraps
from pathlib import Path
from typing import Callable, List, Any

import numpy as np
import structlog
import torch
import yaml

from fisheye.enums import ValidExtensions

logger = structlog.get_logger()


def safe_execution(default_return=None):
    """Generic decorator to catch exceptions and return a default value."""

    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            try:
                return fn(*args, **kwargs)
            except Exception as e:
                logger.error(
                    "function_execution_failed",
                    function=fn.__name__,
                    error=str(e),
                )
                logger.debug(
                    "stack_trace",
                    function=fn.__name__,
                    traceback=traceback.format_exc(),
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
    """Check if the directory contains valid files."""
    return dir_path.is_dir() and any(
        file.suffix in {e.value for e in ValidExtensions} for file in dir_path.iterdir()
    )


def load_model_config():
    """Load model configuration from YAML file."""
    base_dir = Path(__file__).resolve().parents[2]
    config_path = base_dir / "config" / "model.yaml"

    with config_path.open("r") as f:
        config = yaml.safe_load(f)

    return config
