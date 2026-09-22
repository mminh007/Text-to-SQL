"""
logger.py
=========
Centralised logging helper for the Text-to-SQL project.

Usage:
    from src.helpers import TrainLogger

    logger = TrainLogger(name="train", log_dir="logs")
    logger.info("Starting training...")
"""

import os
import logging
from datetime import datetime


class TrainLogger:
    """
    A reusable logger that writes to both the console and a timestamped log file.

    Each instance creates its own named :class:`logging.Logger`, so different
    modules can have separate loggers (and separate log files) without
    interfering with each other.

    Parameters
    ----------
    name : str
        Logger name (appears in ``logging.getLogger(name)``).
        Using a unique name per module keeps loggers isolated.
    log_dir : str
        Directory where log files will be stored. Created automatically.
    console_level : int
        Minimum severity written to the console (default: ``logging.INFO``).
    file_level : int
        Minimum severity written to the log file (default: ``logging.DEBUG``).
    fmt : str | None
        Custom log format string. ``None`` falls back to the built-in default.
    """

    _DEFAULT_FMT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    _DEFAULT_DATEFMT = "%Y-%m-%d %H:%M:%S"

    def __init__(
        self,
        name: str = "app",
        log_dir: str = "logs",
        console_level: int = logging.INFO,
        file_level: int = logging.DEBUG,
        fmt: str | None = None,
    ) -> None:
        self.name = name
        self.log_dir = log_dir
        self.log_file = self._resolve_log_file()

        self._logger = logging.getLogger(name)
        self._logger.setLevel(logging.DEBUG)

        # Guard: only add handlers once per logger name
        if not self._logger.handlers:
            formatter = logging.Formatter(
                fmt=fmt or self._DEFAULT_FMT,
                datefmt=self._DEFAULT_DATEFMT,
            )
            self._add_console_handler(console_level, formatter)
            self._add_file_handler(file_level, formatter)

            self._logger.info("Logger [%s] ready. Log file → %s", name, self.log_file)

    # ------------------------------------------------------------------
    # Internal setup helpers
    # ------------------------------------------------------------------

    def _resolve_log_file(self) -> str:
        """Return the absolute path of the timestamped log file."""
        os.makedirs(self.log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return os.path.join(self.log_dir, f"{self.name}_{timestamp}.log")

    def _add_console_handler(self, level: int, formatter: logging.Formatter) -> None:
        ch = logging.StreamHandler()
        ch.setLevel(level)
        ch.setFormatter(formatter)
        self._logger.addHandler(ch)

    def _add_file_handler(self, level: int, formatter: logging.Formatter) -> None:
        fh = logging.FileHandler(self.log_file, encoding="utf-8")
        fh.setLevel(level)
        fh.setFormatter(formatter)
        self._logger.addHandler(fh)

    # ------------------------------------------------------------------
    # Public logging API  (mirrors logging.Logger)
    # ------------------------------------------------------------------

    def debug(self, msg: str, *args, **kwargs) -> None:
        self._logger.debug(msg, *args, **kwargs)

    def info(self, msg: str, *args, **kwargs) -> None:
        self._logger.info(msg, *args, **kwargs)

    def warning(self, msg: str, *args, **kwargs) -> None:
        self._logger.warning(msg, *args, **kwargs)

    def error(self, msg: str, *args, **kwargs) -> None:
        self._logger.error(msg, *args, **kwargs)

    def critical(self, msg: str, *args, **kwargs) -> None:
        self._logger.critical(msg, *args, **kwargs)

    def exception(self, msg: str, *args, **kwargs) -> None:
        """Log ERROR with full traceback (call inside an ``except`` block)."""
        self._logger.exception(msg, *args, **kwargs)
