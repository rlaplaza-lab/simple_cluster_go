"""DB-oriented retries and a rename-based file probe (not for SQLite DB paths)."""

from __future__ import annotations

import sqlite3
import time
from collections.abc import Callable
from typing import Any

from scgo.utils.logging import get_logger

logger = get_logger(__name__)

HPC_DATABASE_EXCEPTIONS = (sqlite3.OperationalError, OSError)


def database_retry(
    operation: Callable[[], Any],
    max_retries: int = 5,
    initial_delay: float = 0.2,
    backoff_factor: float = 2.0,
    operation_name: str = "database operation",
    log_level: str = "debug",
) -> Any:
    """Run ``operation`` with :func:`retry_with_backoff` and SQLite-friendly defaults."""
    return retry_with_backoff(
        operation=operation,
        max_retries=max_retries,
        initial_delay=initial_delay,
        backoff_factor=backoff_factor,
        exception_types=HPC_DATABASE_EXCEPTIONS,
        operation_name=operation_name,
        log_level=log_level,
    )


def retry_with_backoff(
    operation: Callable[[], Any],
    max_retries: int = 3,
    initial_delay: float = 0.1,
    backoff_factor: float = 2.0,
    exception_types: tuple[type[BaseException], ...] = (OSError,),
    operation_name: str = "operation",
    log_level: str = "debug",
) -> Any:
    """Retry an operation with exponential backoff.

    Useful for filesystem operations that might fail transiently
    on slow network filesystems.

    Args:
        operation: Callable to execute
        max_retries: Maximum retry attempts (coerced to at least 1)
        initial_delay: Initial delay in seconds
        backoff_factor: Delay multiplier for each retry
        exception_types: Exceptions to catch and retry on
        operation_name: Name for logging purposes
        log_level: Logging level for retry messages ('debug', 'info', 'warning')

    Returns:
        Result of the successful operation.
    """
    max_retries = max(1, max_retries)
    delay = initial_delay

    log_methods = {
        "debug": logger.debug,
        "info": logger.info,
        "warning": logger.warning,
        "error": logger.error,
    }
    log_func = log_methods.get(log_level.lower(), logger.debug)

    for attempt in range(max_retries):
        try:
            return operation()
        except exception_types as e:
            if attempt < max_retries - 1:
                log_func(
                    f"{operation_name}: failed (attempt {attempt + 1}/{max_retries}): {e}. "
                    f"Retrying in {delay:.2f}s..."
                )
                time.sleep(delay)
                delay *= backoff_factor
            else:
                logger.error(
                    f"{operation_name}: failed after {max_retries} attempts: {e}"
                )
                raise
