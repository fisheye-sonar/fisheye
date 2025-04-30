import gc
import logging
import random
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from functools import wraps
from typing import Callable, List, Any

import numpy as np
import torch

logger = logging.getLogger(__name__)


def safe_execution(default_return=None):
    """Generic decorator to catch exceptions and return a default value."""

    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            try:
                return fn(*args, **kwargs)
            except Exception as e:
                logger.error(f"Error in {fn.__name__}: {e}")
                logger.debug("Stack trace:\n" + traceback.format_exc())
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
