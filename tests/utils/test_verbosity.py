"""Tests for verbosity management system."""

import logging

import pytest

from scgo.utils.logging import (
    TRACE,
    VERBOSITY_LEVELS,
    configure_logging,
    get_logger,
    should_show_progress,
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
        with pytest.raises(ValueError, match="Invalid verbosity level"):
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
