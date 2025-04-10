import logging
import traceback
from functools import wraps

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
