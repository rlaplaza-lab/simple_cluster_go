"""Retry helpers for database operations.

Provides RetryConfig presets, a decorator to retry on sqlite lock errors,
and a context manager to retry transactions using existing transaction helpers.
"""

from __future__ import annotations

import functools
import sqlite3
import time
from collections.abc import Callable, Generator
from contextlib import contextmanager
from typing import Any, TypeVar

from scgo.database.exceptions import DatabaseLockError
from scgo.utils.logging import get_logger

logger = get_logger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


class RetryConfig:
    """Retry counts and exponential backoff for transient SQLite / I/O errors."""

    def __init__(
        self,
        max_retries: int = 3,
        initial_delay: float = 0.1,
        max_delay: float = 5.0,
        backoff_factor: float = 2.0,
    ):
        self.max_retries = max(1, max_retries)
        self.initial_delay = max(0.0, initial_delay)
        self.max_delay = max(initial_delay, max_delay)
        self.backoff_factor = max(1.0, backoff_factor)

    def get_delay(self, attempt: int) -> float:
        delay = self.initial_delay * (self.backoff_factor**attempt)
        return min(delay, self.max_delay)


# Presets
PRESET_AGGRESSIVE = RetryConfig(max_retries=5, initial_delay=0.1, backoff_factor=2.0)
PRESET_CONSERVATIVE = RetryConfig(max_retries=3, initial_delay=0.5, backoff_factor=1.5)


def is_retryable_error(error: Exception) -> bool:
    """True for sqlite OperationalError messages that often clear after a short wait."""
    if not isinstance(error, sqlite3.OperationalError):
        return False

    msg = str(error).lower()
    # Standard SQLite lock/readonly errors
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
    if config is None:
        config = PRESET_CONSERVATIVE

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_error = None

            for attempt in range(config.max_retries):
                try:
                    return func(*args, **kwargs)
                except sqlite3.OperationalError as e:
                    last_error = e

                    if not is_retryable_error(e):
                        raise

                    if attempt < config.max_retries - 1:
                        delay = config.get_delay(attempt)
                        if log_retries:
                            logger.warning(
                                f"{operation_name}: database locked, retrying in {delay:.2f}s "
                                f"(attempt {attempt + 1}/{config.max_retries})"
                            )
                        time.sleep(delay)
                    else:
                        if log_retries:
                            logger.error(
                                f"{operation_name}: database locked after {config.max_retries} attempts"
                            )
                        raise DatabaseLockError(
                            f"{operation_name} failed: database locked"
                        ) from e

            if last_error:
                raise last_error

        return wrapper  # type: ignore

    return decorator


@contextmanager
def retry_transaction(
    db_connection,
    config: RetryConfig | None = None,
    operation_name: str = "transaction",
) -> Generator[Any, None, None]:
    """Retry opening :func:`database_transaction` on transient lock errors."""
    if config is None:
        config = PRESET_AGGRESSIVE

    last_error = None

    for attempt in range(config.max_retries):
        try:
            # Import here to avoid circular imports
            from scgo.database.transactions import database_transaction

            with database_transaction(db_connection) as conn:
                yield conn
                return

        except sqlite3.OperationalError as e:
            last_error = e

            if not is_retryable_error(e):
                raise

            if attempt < config.max_retries - 1:
                delay = config.get_delay(attempt)
                logger.warning(
                    f"{operation_name}: database locked, retrying in {delay:.2f}s "
                    f"(attempt {attempt + 1}/{config.max_retries})"
                )
                time.sleep(delay)
            else:
                logger.error(
                    f"{operation_name}: database locked after {config.max_retries} attempts"
                )
                raise DatabaseLockError(
                    f"{operation_name} failed: database locked"
                ) from e

    if last_error:
        raise last_error
