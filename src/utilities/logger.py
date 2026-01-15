import sys
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path


class ColoredConsoleFormatter(logging.Formatter):
    """Simple formatter that colors everything except the message"""

    COLORS = {
        'DEBUG': '\033[90m',  # Dark gray
        'INFO': '\033[37m',  # White
        'WARNING': '\033[33m',  # Orange/Yellow
        'ERROR': '\033[31m',  # Red
        'CRITICAL': '\033[35m'  # Magenta
    }

    RESET = '\033[0m'

    def format(self, record):
        message = super().format(record)

        # Only add color if outputting to terminal
        if sys.stderr.isatty():
            color = self.COLORS.get(record.levelname, self.RESET)
            parts = message.rsplit(' | ', 1)
            if len(parts) == 2:
                metadata, actual_message = parts
                return f"{color}{metadata}{self.RESET} | {actual_message}"
            else:
                return f"{color}{message}{self.RESET}"

        return message


def setup_logging(log_file: str | Path | None = "log.txt", console_level: int = logging.INFO,
                  file_level: int = logging.DEBUG, max_bytes: int = 10 * 1024 * 1024, backup_count: int = 10,
                  colored_console: bool = True) -> None:
    """
    Set up logging configuration with both console and file handlers.

    Args:
        log_file: Path to log file. If None, only console logging will be enabled.
        console_level: Logging level for console output
        file_level: Logging level for file output
        max_bytes: Maximum size of log file before rotation
        backup_count: Number of backup files to keep
        colored_console: Whether to color console output using ColoredConsoleFormatter
    """
    fmt = "%(asctime)s | %(levelname)s | %(filename)s | %(message)s"
    formatter = logging.Formatter(fmt)

    # Console handler
    console = logging.StreamHandler()
    console.setLevel(console_level)
    console.setFormatter(ColoredConsoleFormatter(fmt) if colored_console else formatter)

    handlers = [console]

    # File handler (if log_file is provided)
    if log_file:
        # Ensure log directory exists
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8"
        )
        file_handler.setLevel(file_level)
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)

    # Configure logging
    logging.basicConfig(
        level=min(console_level, file_level),
        handlers=handlers,
    )


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the specified name.

    Args:
        name: Name for the logger

    Returns:
        logging.Logger: Configured logger instance
    """
    return logging.getLogger(name)