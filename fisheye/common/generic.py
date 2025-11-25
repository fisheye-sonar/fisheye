import gc
import random
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from functools import wraps
from typing import Callable, List, Any

import numpy as np
import structlog
import torch

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
                # MAH 2025-11-24 12:30:34 TODO put back in try/except
                print(f"MAH TODO put back in try/except")
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    tb = traceback.extract_tb(e.__traceback__)
                    # Get the last traceback entry (where the error occurred)
                    last_call = tb[-1] if tb else None

                    logger.error(
                        "safe_execution_exception",
                        function=func.__name__,
                        attempt=attempt,
                        error=str(e),
                        code_line=last_call.line if last_call else None,
                        file=last_call.filename if last_call else None,
                        function_name=last_call.name if last_call else None,
                        line=last_call.lineno if last_call else None,
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
