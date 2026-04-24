"""Database retry helpers for transient SQLite and filesystem errors."""

from __future__ import annotations

import functools
import sqlite3
import time
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

from scgo.utils.logging import get_logger

logger = get_logger(__name__)

HPC_DATABASE_EXCEPTIONS = (sqlite3.OperationalError, OSError)
F = Callable[..., Any]


@dataclass(frozen=True)
class RetryConfig:
    """Retry counts and exponential backoff for transient SQLite / I/O errors."""

    max_retries: int = 3
    initial_delay: float = 0.1
    max_delay: float = 5.0
    backoff_factor: float = 2.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "max_retries", max(1, self.max_retries))
        object.__setattr__(self, "initial_delay", max(0.0, self.initial_delay))
        object.__setattr__(self, "max_delay", max(self.initial_delay, self.max_delay))
        object.__setattr__(self, "backoff_factor", max(1.0, self.backoff_factor))

    def get_delay(self, attempt: int) -> float:
        delay = self.initial_delay * (self.backoff_factor**attempt)
        return min(delay, self.max_delay)


PRESET_AGGRESSIVE = RetryConfig(max_retries=5, initial_delay=0.1, backoff_factor=2.0)
PRESET_CONSERVATIVE = RetryConfig(max_retries=3, initial_delay=0.5, backoff_factor=1.5)


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


def is_retryable_error(error: Exception) -> bool:
    """True for sqlite OperationalError messages that often clear after a short wait."""
    if not isinstance(error, sqlite3.OperationalError):
        return False

    msg = str(error).lower()
    if "locked" in msg or "readonly" in msg:
        return True

    return any(
        kw in msg
        for kw in (
            "disk i/o error",
            "resource temporarily unavailable",
            "input/output error",
        )
    )


def retry_on_lock(
    config: RetryConfig | None = None,
    operation_name: str = "database operation",
    log_retries: bool = True,
) -> Callable[[F], F]:
    """Decorator to retry callable on sqlite locked/readonly OperationalError."""
    effective_config = config or PRESET_CONSERVATIVE

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(effective_config.max_retries):
                try:
                    return func(*args, **kwargs)
                except sqlite3.OperationalError as e:
                    if not is_retryable_error(e):
                        raise

                    if attempt < effective_config.max_retries - 1:
                        delay = effective_config.get_delay(attempt)
                        if log_retries:
                            logger.warning(
                                f"{operation_name}: database locked, retrying in {delay:.2f}s "
                                f"(attempt {attempt + 1}/{effective_config.max_retries})"
                            )
                        time.sleep(delay)
                    else:
                        if log_retries:
                            logger.error(
                                f"{operation_name}: database locked after {effective_config.max_retries} attempts"
                            )
                        raise
            raise RuntimeError(f"{operation_name} failed unexpectedly")

        return wrapper  # type: ignore[return-value]

    return decorator


@contextmanager
def retry_transaction(
    db_connection,
    config: RetryConfig | None = None,
    operation_name: str = "transaction",
):
    """Retry opening ``database_transaction`` on transient lock errors."""
    effective_config = config or PRESET_AGGRESSIVE
    for attempt in range(effective_config.max_retries):
        try:
            # Lazy import avoids circular imports.
            from scgo.database.transactions import database_transaction

            with database_transaction(db_connection) as conn:
                yield conn
                return
        except sqlite3.OperationalError as e:
            if not is_retryable_error(e):
                raise
            if attempt < effective_config.max_retries - 1:
                delay = effective_config.get_delay(attempt)
                logger.warning(
                    f"{operation_name}: database locked, retrying in {delay:.2f}s "
                    f"(attempt {attempt + 1}/{effective_config.max_retries})"
                )
                time.sleep(delay)
            else:
                logger.error(
                    f"{operation_name}: database locked after {effective_config.max_retries} attempts"
                )
                raise


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
