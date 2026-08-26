"""Tests for verbosity management system."""

import logging

import pytest

from scgo.exceptions import SCGOValidationError
from scgo.utils.logging import (
    TRACE,
    VERBOSITY_LEVELS,
    configure_logging,
    get_logger,
    log_debug_v,
    log_info_v,
    log_warning_v,
    should_show_progress,
    suppress_matching_stdout,
)


class TestVerbosityLevels:
    """Test verbosity level mapping and configuration."""

    def test_verbosity_levels_mapping(self):
        """Test that verbosity levels map to correct logging levels."""
        assert VERBOSITY_LEVELS[0] == logging.WARNING
        assert VERBOSITY_LEVELS[1] == logging.INFO
        assert VERBOSITY_LEVELS[2] == logging.DEBUG
        assert VERBOSITY_LEVELS[3] == TRACE

    @pytest.mark.parametrize("verbosity", [0, 1, 2, 3])
    def test_configure_logging_valid_levels(self, verbosity):
        """Test logging configuration with valid verbosity levels."""
        configure_logging(verbosity, hpc_mode=True)
        root_logger = logging.getLogger()
        assert root_logger.level == VERBOSITY_LEVELS[verbosity]

    def test_hpc_mode_default_suppresses_numpy(self, monkeypatch):
        monkeypatch.delenv("SCGO_LOCAL_DEV", raising=False)
        configure_logging(1)
        assert logging.getLogger("numpy").level == logging.ERROR

    def test_local_dev_env_relaxes_third_party(self, monkeypatch):
        monkeypatch.setenv("SCGO_LOCAL_DEV", "1")
        configure_logging(1)
        assert logging.getLogger("numpy").level == logging.WARNING

    def test_configure_logging_invalid_level(self):
        """Test that invalid verbosity levels raise ValueError."""
        with pytest.raises(SCGOValidationError, match="Invalid verbosity level"):
            configure_logging(4)

    def test_configure_logging_removes_existing_handlers(self):
        """Test that configure_logging removes existing handlers."""
        # Get initial handler count (pytest may have added some)
        root_logger = logging.getLogger()
        initial_count = len(root_logger.handlers)

        # Add a handler first
        handler = logging.StreamHandler()
        root_logger.addHandler(handler)
        assert len(root_logger.handlers) == initial_count + 1

        # Configure logging should remove existing handlers and add exactly one new one
        configure_logging(1)
        assert len(root_logger.handlers) == 1  # Should have exactly one new handler


class TestLoggerFactory:
    """Test logger factory functions."""

    def test_get_logger_returns_logger(self):
        """Test that get_logger returns a proper logger instance."""
        logger = get_logger("test.module")
        assert isinstance(logger, logging.Logger)
        assert logger.name == "test.module"

    def test_get_logger_different_modules(self):
        """Test that different module names return different loggers."""
        logger1 = get_logger("module1")
        logger2 = get_logger("module2")
        assert logger1 is not logger2
        assert logger1.name == "module1"
        assert logger2.name == "module2"


class TestProgressBarControl:
    """Test progress bar visibility control."""

    def test_should_show_progress(self):
        """Test progress bar visibility logic."""
        assert not should_show_progress(0)  # quiet
        assert should_show_progress(1)  # normal
        assert should_show_progress(2)  # debug
        assert should_show_progress(3)  # trace

    def test_tqdm_disable_logic(self):
        """Test tqdm disable parameter logic (using not should_show_progress)."""
        assert (not should_show_progress(0)) is True  # quiet - disable progress bars
        assert (not should_show_progress(1)) is False  # normal - show progress bars
        assert (not should_show_progress(2)) is False  # debug - show progress bars
        assert (not should_show_progress(3)) is False  # trace - show progress bars


class TestLoggingOutput:
    """Test actual logging output behavior."""

    def test_quiet_mode_output(self, capfd):
        """Test that quiet mode (0) only shows warnings and errors."""
        configure_logging(0)
        logger = get_logger("test")

        # Log messages
        logger.debug("Debug message")
        logger.info("Info message")
        logger.warning("Warning message")
        logger.error("Error message")

        # Capture output
        captured = capfd.readouterr()
        output = captured.out

        # Should only contain warning and error messages
        assert "Debug message" not in output
        assert "Info message" not in output
        assert "Warning message" in output
        assert "Error message" in output

    def test_normal_mode_output(self, capfd):
        """Test that normal mode (1) shows info, warnings, and errors."""
        configure_logging(1)
        logger = get_logger("test")

        # Log messages
        logger.debug("Debug message")
        logger.info("Info message")
        logger.warning("Warning message")
        logger.error("Error message")

        # Capture output
        captured = capfd.readouterr()
        output = captured.out

        # Should contain info, warning, and error messages
        assert "Debug message" not in output
        assert "Info message" in output
        assert "Warning message" in output
        assert "Error message" in output

    def test_verbose_mode_output(self, capfd):
        """Test that verbose mode (2) shows all messages except trace."""
        configure_logging(2)
        logger = get_logger("test")

        # Log messages
        logger.trace("Trace message")
        logger.debug("Debug message")
        logger.info("Info message")
        logger.warning("Warning message")
        logger.error("Error message")

        # Capture output
        captured = capfd.readouterr()
        output = captured.out

        # Should contain debug, info, warning, and error messages (but not trace)
        assert "Trace message" not in output
        assert "Debug message" in output
        assert "Info message" in output
        assert "Warning message" in output
        assert "Error message" in output

    def test_trace_mode_output(self, capfd):
        """Test that trace mode (3) shows all messages including trace."""
        configure_logging(3)
        logger = get_logger("test")

        # Log messages
        logger.trace("Trace message")
        logger.debug("Debug message")
        logger.info("Info message")
        logger.warning("Warning message")
        logger.error("Error message")

        # Capture output
        captured = capfd.readouterr()
        output = captured.out

        # Should contain all messages including trace
        assert "Trace message" in output
        assert "Debug message" in output
        assert "Info message" in output
        assert "Warning message" in output
        assert "Error message" in output

    def test_clean_output_format(self, capfd):
        """Test that output format is clean (no timestamps/module names by default)."""
        configure_logging(1)
        logger = get_logger("test")

        # Log message
        logger.info("Test message")

        # Capture output
        captured = capfd.readouterr()
        output = captured.out.strip()

        # Should be just the message, no timestamps or module names
        assert output == "Test message"
        assert "test" not in output  # module name should not appear
        assert ":" not in output  # no timestamps or other formatting


class TestVerbosityGatedHelpers:
    """Test verbosity-gated logging helper functions."""

    def test_log_info_v_default(self, capfd):
        """Test log_info_v with default verbosity (1) and min_verbosity (1)."""
        configure_logging(1)
        logger = get_logger("test")
        log_info_v(logger, "Test info message", verbosity=1, min_verbosity=1)
        captured = capfd.readouterr()
        assert "Test info message" in captured.out

    def test_log_info_v_suppressed(self, capfd):
        """Test log_info_v suppressed when verbosity < min_verbosity."""
        configure_logging(0)
        logger = get_logger("test")
        log_info_v(logger, "Should not appear", verbosity=0, min_verbosity=1)
        captured = capfd.readouterr()
        assert "Should not appear" not in captured.out

    def test_log_debug_v_default(self, capfd):
        """Test log_debug_v with verbosity=2, min_verbosity=2."""
        configure_logging(2)
        logger = get_logger("test")
        log_debug_v(logger, "Test debug message", verbosity=2, min_verbosity=2)
        captured = capfd.readouterr()
        assert "Test debug message" in captured.out

    def test_log_debug_v_suppressed_at_verbosity_1(self, capfd):
        """Test log_debug_v suppressed when verbosity=1 < min_verbosity=2."""
        configure_logging(1)
        logger = get_logger("test")
        log_debug_v(logger, "Should not appear", verbosity=1, min_verbosity=2)
        captured = capfd.readouterr()
        assert "Should not appear" not in captured.out

    def test_log_warning_v_always_shown_at_0(self, capfd):
        """Test log_warning_v shown even at verbosity=0 with min_verbosity=0."""
        configure_logging(0)
        logger = get_logger("test")
        log_warning_v(logger, "Warning message", verbosity=0, min_verbosity=0)
        captured = capfd.readouterr()
        assert "Warning message" in captured.out

    def test_lazy_evaluation_with_logger_level(self, capfd):
        """Test that logger respects level filtering (lazy evaluation)."""
        configure_logging(0)  # WARNING level only
        logger = get_logger("test")

        # At WARNING level, INFO and DEBUG are filtered by the logger
        # The logging system itself does lazy evaluation
        logger.info("This should not appear")
        logger.debug("This should not appear either")
        logger.warning("This should appear")

        captured = capfd.readouterr()
        assert "This should not appear" not in captured.out
        assert "This should not appear either" not in captured.out
        assert "This should appear" in captured.out

    def test_format_string_with_args(self, capfd):
        """Test that format strings with args work correctly."""
        configure_logging(1)
        logger = get_logger("test")
        log_info_v(
            logger,
            "Processing %d of %d items",
            5,
            10,
            verbosity=1,
            min_verbosity=1,
        )
        captured = capfd.readouterr()
        assert "Processing 5 of 10 items" in captured.out


def test_suppress_matching_stdout(capfd):
    captured: list[str] = []
    with suppress_matching_stdout("Model Memory Estimation:", captured=captured):
        print("keep me")
        print("Model Memory Estimation: Running forward pass")
        print("also keep")
    out = capfd.readouterr().out
    assert "keep me" in out
    assert "also keep" in out
    assert "Model Memory Estimation" not in out
    assert len(captured) == 1
    assert "Running forward pass" in captured[0]


def test_inductor_filelock_events_suppressed_and_summarized(capfd):
    """TorchInductor filelock DEBUG spam becomes one INFO summary when drained."""
    from scgo.utils.logging import drain_inductor_filelock_summary

    configure_logging(2)
    filelock_logger = logging.getLogger("filelock")
    lock_path = "/tmp/torchinductor_test/locks/abc.lock"
    for msg in (
        "Attempting to acquire lock 123 on %s",
        "Lock 123 acquired on %s",
        "Attempting to release lock 123 on %s",
        "Lock 123 released on %s",
    ):
        filelock_logger.debug(msg, lock_path)
    filelock_logger.debug("Attempting to acquire lock 9 on /tmp/other.lock")

    out = capfd.readouterr().out
    assert "Attempting to acquire lock" not in out
    assert "torchinductor" not in out

    scgo_logger = get_logger("test.inductor_filelock")
    drain_inductor_filelock_summary(scgo_logger)
    out = capfd.readouterr().out
    assert out.count("TorchInductor:") == 1
    assert "4 compile-cache lock event(s)" in out
    assert "1 lock file" in out

    drain_inductor_filelock_summary(scgo_logger)
    assert "TorchInductor:" not in capfd.readouterr().out


class TestIntegration:
    """Integration tests for verbosity system."""

    def test_multiple_loggers_same_verbosity(self, capfd):
        """Test that multiple loggers respect the same verbosity setting."""
        configure_logging(1)
        logger1 = get_logger("module1")
        logger2 = get_logger("module2")

        # Log messages
        logger1.info("Message from module1")
        logger2.info("Message from module2")

        # Capture output
        captured = capfd.readouterr()
        output = captured.out

        assert "Message from module1" in output
        assert "Message from module2" in output

    def test_verbosity_change_affects_all_loggers(self, capfd):
        """Test that changing verbosity affects all existing loggers."""
        configure_logging(0)  # quiet
        logger = get_logger("test")

        # Test quiet mode
        logger.info("Should not appear")
        captured1 = capfd.readouterr()
        output1 = captured1.out
        assert "Should not appear" not in output1

        # Change verbosity
        configure_logging(1)  # normal

        # Test normal mode
        logger.info("Should appear now")
        captured2 = capfd.readouterr()
        output2 = captured2.out
        assert "Should appear now" in output2


def test_trace_always_available():
    """Trace method is now always available on import."""
    assert hasattr(logging.Logger, "trace")


def test_trace_method_works(capfd):
    # Configure logging and ensure trace method is installed and works
    configure_logging(3)
    assert hasattr(logging.Logger, "trace")

    logger = logging.getLogger("test_trace")
    logger.trace("Trace message")
    logger.debug("Debug message")

    captured = capfd.readouterr()
    output = captured.out

    assert "Trace message" in output
    assert "Debug message" in output
