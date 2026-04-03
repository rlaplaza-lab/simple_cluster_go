import logging

from scgo.utils import logging as scgo_logging


def test_trace_always_available():
    """Trace method is now always available on import."""
    assert hasattr(logging.Logger, "trace")


def test_trace_method_works(capfd):
    # Configure logging and ensure trace method is installed and works
    scgo_logging.configure_logging(3)
    assert hasattr(logging.Logger, "trace")

    logger = logging.getLogger("test_trace")
    logger.trace("Trace message")
    logger.debug("Debug message")

    captured = capfd.readouterr()
    output = captured.out

    assert "Trace message" in output
    assert "Debug message" in output
