"""File-only application logging, including library output and warnings."""

import logging
from pathlib import Path
import sys


class _LogStream:
    """Route legacy stdout/stderr writes through logging without console output."""

    def __init__(self, name: str, level: int):
        self.logger = logging.getLogger(name)
        self.level = level
        self.pending = ""
        self.encoding = "utf-8"

    def write(self, text: str) -> int:
        self.pending += text.replace("\r", "\n")
        while "\n" in self.pending:
            line, self.pending = self.pending.split("\n", 1)
            if line.strip():
                self.logger.log(self.level, line.rstrip())
        return len(text)

    def flush(self) -> None:
        if self.pending.strip():
            self.logger.log(self.level, self.pending.rstrip())
        self.pending = ""

    def isatty(self) -> bool:
        return False


def configure_logging() -> Path:
    """Append timestamped logs to app.log beside the project entry point."""
    path = Path(__file__).resolve().parent / "app.log"
    root = logging.getLogger()
    for handler in root.handlers[:]:
        root.removeHandler(handler)
        handler.close()
    handler = logging.FileHandler(path, mode="a", encoding="utf-8")
    handler.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    ))
    root.addHandler(handler)
    root.setLevel(logging.INFO)
    # Libraries can install their own console handlers; use the root file handler.
    for logger in logging.root.manager.loggerDict.values():
        if isinstance(logger, logging.Logger):
            for existing in logger.handlers[:]:
                logger.removeHandler(existing)
                existing.close()
            logger.propagate = True
    logging.captureWarnings(True)
    if not isinstance(sys.stdout, _LogStream):
        sys.stdout = _LogStream("stdout", logging.INFO)
    if not isinstance(sys.stderr, _LogStream):
        sys.stderr = _LogStream("stderr", logging.WARNING)
    return path
