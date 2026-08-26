"""Root logging setup, verbosity levels, and TRACE support for SCGO."""

from __future__ import annotations

import contextlib
import logging
import os
import re
import sys
from collections.abc import Iterator
from typing import TextIO

from scgo.exceptions import SCGOValidationError
from scgo.utils.runtime_warnings import apply_scgo_runtime_warning_filters

# Define custom TRACE logging level (below DEBUG)
TRACE = 5
logging.addLevelName(TRACE, "TRACE")


def _trace(self, message, *args, **kwargs):
    """Log at TRACE; defers formatting when TRACE is disabled."""
    if self.isEnabledFor(TRACE):
        self._log(TRACE, message, args, **kwargs)


# Install trace method immediately on import (modern logging support)
logging.Logger.trace = _trace  # type: ignore[attr-defined]


# Verbosity level mapping: user-friendly integers to Python logging levels
VERBOSITY_LEVELS: dict[int, int] = {
    0: logging.WARNING,  # quiet - only warnings and errors
    1: logging.INFO,  # normal - key updates + warnings (default)
    2: logging.DEBUG,  # verbose - detailed information
    3: TRACE,  # ultra-verbose - trace-level diagnostics
}

_TORCHINDUCTOR_LOCK_PATH_RE = re.compile(r" on (\S+\.lock)\s*$")


class _InductorFileLockHandler(logging.Handler):
    """Count TorchInductor filelock DEBUG records; do not propagate to console."""

    def __init__(self) -> None:
        super().__init__(level=logging.DEBUG)
        self._events = 0
        self._paths: set[str] = set()

    def emit(self, record: logging.LogRecord) -> None:
        message = record.getMessage()
        if "torchinductor" not in message.lower():
            return
        match = _TORCHINDUCTOR_LOCK_PATH_RE.search(message)
        with self.lock:
            self._events += 1
            if match is not None:
                self._paths.add(match.group(1))

    def drain(self) -> tuple[int, int]:
        """Return (event_count, unique_lock_files) and reset."""
        with self.lock:
            count, n_paths = self._events, len(self._paths)
            self._events = 0
            self._paths.clear()
        return count, n_paths


_filelock_handler: _InductorFileLockHandler | None = None


def get_logger(name: str) -> logging.Logger:
    """Return a logger for ``name`` (typically ``__name__``)."""
    return logging.getLogger(name)


DEFAULT_LOG_FORMAT = "%(message)s"
FULL_LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
LOG_FORMAT_ENV_VAR = "SCGO_LOG_FORMAT"


def resolve_log_format(format_string: str | None = None) -> str:
    """Resolve the console log format.

    Precedence: an explicit ``format_string`` wins; otherwise ``SCGO_LOG_FORMAT``
    selects a named preset (``full`` → ``FULL_LOG_FORMAT``, ``plain`` →
    ``DEFAULT_LOG_FORMAT``) or is used verbatim as a ``%``-style format
    string; otherwise the default ``"%(message)s"`` is used.
    """
    if format_string is not None:
        return format_string
    env = os.environ.get(LOG_FORMAT_ENV_VAR, "").strip()
    if not env:
        return DEFAULT_LOG_FORMAT
    preset = env.lower()
    if preset == "full":
        return FULL_LOG_FORMAT
    if preset in ("plain", "default", "message"):
        return DEFAULT_LOG_FORMAT
    return env


def configure_logging(
    verbosity: int = 1,
    format_string: str | None = None,
    hpc_mode: bool | None = None,
) -> None:
    """Configure the root logger for the SCGO package.

    SCGO targets HPC batch jobs: by default third-party loggers are suppressed
    aggressively. Set environment variable ``SCGO_LOCAL_DEV=1`` to use milder
    suppression when ``hpc_mode`` is omitted, or pass ``hpc_mode=False``
    explicitly.

    Args:
        verbosity: Verbosity level (0=quiet, 1=normal, 2=debug, 3=trace).
        format_string: Custom ``%``-style format string. When ``None``, the
            ``SCGO_LOG_FORMAT`` environment variable is consulted
            (``full`` gives ``"%(asctime)s %(levelname)s %(name)s: %(message)s"``,
            ``plain`` gives the default ``"%(message)s"``, and any other value is
            used verbatim). Defaults to ``"%(message)s"``.
        hpc_mode: If True, suppresses more third-party logs (default when None,
            unless ``SCGO_LOCAL_DEV=1``). If False, only WARNING+ for most libs.
    """
    if hpc_mode is None:
        hpc_mode = os.environ.get("SCGO_LOCAL_DEV") != "1"
    apply_scgo_runtime_warning_filters()
    if verbosity not in VERBOSITY_LEVELS:
        raise SCGOValidationError(
            f"Invalid verbosity level: {verbosity}. Must be 0, 1, 2, or 3."
        )

    level = VERBOSITY_LEVELS[verbosity]
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    for handler in root_logger.handlers[:]:
        handler.close()
        root_logger.removeHandler(handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)

    formatter = logging.Formatter(resolve_log_format(format_string))
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    _install_filelock_sink()
    _suppress_third_party_loggers(level, hpc_mode=hpc_mode)


def _install_filelock_sink() -> None:
    """Capture TorchInductor filelock DEBUG without propagating to root.

    ``filelock`` is a sibling of ``torch``, so torch suppression does not cover
    it. Keep level DEBUG so the counting handler sees events.
    """
    global _filelock_handler
    filelock_logger = logging.getLogger("filelock")
    filelock_logger.setLevel(logging.DEBUG)
    filelock_logger.propagate = False
    if _filelock_handler is None:
        _filelock_handler = _InductorFileLockHandler()
    if _filelock_handler not in filelock_logger.handlers:
        filelock_logger.addHandler(_filelock_handler)


def drain_inductor_filelock_summary(logger: logging.Logger) -> None:
    """Emit one INFO summary if TorchInductor filelock events were captured."""
    if _filelock_handler is None:
        return
    count, n_paths = _filelock_handler.drain()
    if count <= 0:
        return
    logger.info(
        "TorchInductor: %d compile-cache lock event(s) (%d lock file%s)",
        count,
        n_paths,
        "" if n_paths == 1 else "s",
    )


def _suppress_third_party_loggers(level: int, hpc_mode: bool = False) -> None:
    """Suppress or control third-party library loggers to prevent noise.

    Args:
        level: The logging level to set for third-party loggers.
        hpc_mode: If True, suppresses more aggressively for HPC environments.
    """
    third_party_loggers = [
        "ase",
        "ase.calculators",
        "ase.optimize",
        "torch",
        "torch_sim",
        "mace",
        "tqdm",
        "urllib3",
        "requests",
        "numpy",
        "scipy",
        "matplotlib",
        "pandas",
        "h5py",
        "netCDF4",
    ]

    for logger_name in third_party_loggers:
        logger = logging.getLogger(logger_name)
        suppression_level = logging.ERROR if hpc_mode else max(level, logging.WARNING)
        logger.setLevel(suppression_level)
        logger.propagate = False


def should_show_progress(verbosity: int) -> bool:
    """True when verbosity >= 1 (progress bars enabled for normal+)."""
    return verbosity >= 1


# ---------------------------------------------------------------------------
# Verbosity-gated logging helpers
# ---------------------------------------------------------------------------
# Style contract:
# - After ``configure_logging``, prefer ``logger.info`` / ``logger.debug`` /
#   ``logger.warning`` with %-style formatting (lazy args).
# - Use ``log_*_v`` only when a function takes integer ``verbosity``
#   and needs extra gating beyond the root logger level.
# - v2+ per-item detail is DEBUG (``log_debug_v``),
#   never INFO gated to v2.
# - Phase rollups and banners live in ``scgo.utils.phase_logging``.
# - Prefer %-style: logger.info("Processing %s", item)
# - Avoid f-strings: logger.info(f"Processing {item}")
# - Use logger.exception() for unexpected errors with automatic traceback
# - Use exc_info=(verbosity >= 2) for handled errors with conditional traceback


def log_debug_v(
    logger: logging.Logger,
    message: str,
    *args: object,
    verbosity: int = 1,
    min_verbosity: int = 2,
) -> None:
    """Log debug message if verbosity >= min_verbosity (default 2).

    Uses lazy %-style formatting. Message is only formatted if it will be logged.

    Args:
        logger: The logger instance.
        message: Format string for the message.
        *args: Arguments for the format string.
        verbosity: Current verbosity level (0-3).
        min_verbosity: Minimum verbosity to log (default 2 = DEBUG).
    """
    if verbosity >= min_verbosity:
        logger.debug(message, *args)


def log_info_v(
    logger: logging.Logger,
    message: str,
    *args: object,
    verbosity: int = 1,
    min_verbosity: int = 1,
) -> None:
    """Log info message if verbosity >= min_verbosity (default 1).

    Uses lazy %-style formatting. Message is only formatted if it will be logged.

    Args:
        logger: The logger instance.
        message: Format string for the message.
        *args: Arguments for the format string.
        verbosity: Current verbosity level (0-3).
        min_verbosity: Minimum verbosity to log (default 1 = INFO).
    """
    if verbosity >= min_verbosity:
        logger.info(message, *args)


def log_warning_v(
    logger: logging.Logger,
    message: str,
    *args: object,
    verbosity: int = 1,
    min_verbosity: int = 1,
) -> None:
    """Log warning message if verbosity >= min_verbosity (default 1).

    Warnings are typically always shown, but this allows conditional suppression.

    Args:
        logger: The logger instance.
        message: Format string for the message.
        *args: Arguments for the format string.
        verbosity: Current verbosity level (0-3).
        min_verbosity: Minimum verbosity to log (default 1).
    """
    if verbosity >= min_verbosity:
        logger.warning(message, *args)


class _MatchingStdoutFilter:
    """Line-buffered stdout wrapper that drops lines matching a pattern."""

    def __init__(
        self,
        underlying: TextIO,
        pattern: re.Pattern[str],
        captured: list[str] | None,
    ) -> None:
        self._underlying = underlying
        self._pattern = pattern
        self._captured = captured
        self._buf = ""

    def write(self, data: str) -> int:
        if not isinstance(data, str):
            data = str(data)
        self._buf += data
        written = len(data)
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            full = line + "\n"
            if self._pattern.search(line):
                if self._captured is not None:
                    self._captured.append(full)
            else:
                self._underlying.write(full)
        return written

    def flush(self) -> None:
        if self._buf:
            if self._pattern.search(self._buf):
                if self._captured is not None:
                    self._captured.append(self._buf)
            else:
                self._underlying.write(self._buf)
            self._buf = ""
        self._underlying.flush()

    def __getattr__(self, name: str):
        return getattr(self._underlying, name)


@contextlib.contextmanager
def suppress_matching_stdout(
    pattern: str,
    *,
    captured: list[str] | None = None,
) -> Iterator[None]:
    """Temporarily drop stdout lines that contain ``pattern``.

    Logging handlers created by ``configure_logging`` already hold a reference
    to the original ``sys.stdout``, so SCGO log records still appear. Optional
    ``captured`` collects suppressed lines for DEBUG re-emission.

    Args:
        pattern: Literal substring matched against each line.
        captured: If provided, append each suppressed line (including newline).
    """
    compiled = re.compile(re.escape(pattern))
    filter_stream = _MatchingStdoutFilter(sys.stdout, compiled, captured)
    with contextlib.redirect_stdout(filter_stream):
        try:
            yield
        finally:
            filter_stream.flush()
