"""Keep third-party debug output from overwhelming stage progress logs."""

import re
import sys
import logging
from contextlib import contextmanager, redirect_stdout


_SHAPE = re.compile(r"\(\d+,\)(?: \(\d+,\))?")
_COUNTER = re.compile(r"\d+")
_GRID_DIAGNOSTIC = (
    "[datetime.datetime(",
    "collection times =",
    "data for table dt =",
    "data for time extra dt =",
    "{'stations':",
)


class _ProgressOutput:
    def __init__(self, logger):
        self.logger = logger
        self.pending = ""

    def write(self, text):
        self.pending += text
        while "\n" in self.pending:
            line, self.pending = self.pending.split("\n", 1)
            if not self._is_diagnostic(line):
                self.logger.info("%s", line)
        return len(text)

    def flush(self):
        pass

    def finish(self):
        if self.pending and not self._is_diagnostic(self.pending):
            self.logger.info("%s", self.pending)
        self.pending = ""

    @staticmethod
    def _is_diagnostic(line):
        line = line.strip()
        return bool(_SHAPE.fullmatch(line) or _COUNTER.fullmatch(line)) or line.startswith(
            _GRID_DIAGNOSTIC
        )


class LibraryDiagnosticFilter(logging.Filter):
    """Remove lmatools progress counters from its separate file logger."""

    def filter(self, record):
        return not _ProgressOutput._is_diagnostic(record.getMessage())


def filter_library_logger():
    logger = logging.getLogger("FlashAutorunLogger")
    if not any(isinstance(item, LibraryDiagnosticFilter) for item in logger.filters):
        logger.addFilter(LibraryDiagnosticFilter())


@contextmanager
def readable_library_output(logger):
    """Pass through library progress and warnings, hiding known numeric debug lines."""
    output = _ProgressOutput(logger)
    with redirect_stdout(output):
        try:
            yield
        finally:
            output.finish()


def configure_stage_logging(verbose=False):
    """Configure a single stdout handler for the stage CLI loggers."""
    logger = logging.getLogger("lma_scripts")
    if not any(getattr(handler, "_lma_stage_handler", False) for handler in logger.handlers):
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
        handler._lma_stage_handler = True
        logger.addHandler(handler)
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    logger.propagate = False
