import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import TextIO

LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

LOG_FORMAT = (
    "%(asctime)s - %(levelname)-8s - %(name)s - "
    "%(filename)s:%(lineno)d - %(message)s"
)
DATE_FORMAT = "%H:%M:%S"


class DailyFileHandler(logging.Handler):
    """Write each log record to the file named for its local calendar day."""

    def __init__(self, directory: Path) -> None:
        super().__init__()
        self.directory = directory
        self._current_date: str | None = None
        self._stream: TextIO | None = None

    def emit(self, record: logging.LogRecord) -> None:
        try:
            log_date = datetime.fromtimestamp(record.created).strftime("%Y-%m-%d")
            if log_date != self._current_date:
                if self._stream is not None:
                    self._stream.close()
                self._stream = (self.directory / f"{log_date}.log").open(
                    "a", encoding="utf-8"
                )
                self._current_date = log_date

            if self._stream is not None:
                self._stream.write(self.format(record) + "\n")
                self.flush()
        except Exception:
            self.handleError(record)

    def flush(self) -> None:
        if self._stream is not None:
            self._stream.flush()

    def close(self) -> None:
        self.acquire()
        try:
            if self._stream is not None:
                self._stream.close()
                self._stream = None
            super().close()
        finally:
            self.release()


class ColoredFormatter(logging.Formatter):
    COLORS = {
        logging.DEBUG: "\033[36m",
        logging.INFO: "\033[32m",
        logging.WARNING: "\033[33m",
        logging.ERROR: "\033[31m",
        logging.CRITICAL: "\033[1;31m",
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        message = super().format(record)
        color = self.COLORS.get(record.levelno)
        if color is None:
            return message
        return f"{color}{message}{self.RESET}"


class ColorStreamHandler(logging.StreamHandler):
    """Emit colored log records to stdout."""


def setup_logger(name: str = "Thyroid App") -> logging.Logger:
    """Return a configured logger that writes to the console and daily log files."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if not any(isinstance(handler, ColorStreamHandler) for handler in logger.handlers):
        console_handler = ColorStreamHandler(sys.stdout)
        console_handler.setFormatter(ColoredFormatter(LOG_FORMAT, datefmt=DATE_FORMAT))
        logger.addHandler(console_handler)

    if not any(isinstance(handler, DailyFileHandler) for handler in logger.handlers):
        file_handler = DailyFileHandler(LOG_DIR)
        file_handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))
        logger.addHandler(file_handler)

    return logger


logger = setup_logger()
