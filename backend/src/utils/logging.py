"""Logging configuration for FastRTC Voice Assistant.

This module provides centralized logging configuration with proper formatters,
handlers, and log levels. It extracts and improves upon the logging setup
from the original monolithic implementation.
"""

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime


# Default logging configuration
DEFAULT_LOG_LEVEL = logging.INFO
DEFAULT_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Component-specific log levels
COMPONENT_LOG_LEVELS: Dict[str, int] = {
    "backend.audio": logging.INFO,
    "backend.memory": logging.INFO,
    "backend.llm": logging.INFO,
    "backend.tts": logging.INFO,
    "backend.stt": logging.INFO,
    "backend.core": logging.INFO,
    "backend.integration": logging.INFO,
    # External libraries
    "httpx": logging.WARNING,
    "urllib3": logging.WARNING,
    "requests": logging.WARNING,
    "transformers": logging.WARNING,
    "torch": logging.WARNING,
}


class ColoredFormatter(logging.Formatter):
    """Colored console formatter for better readability."""
    
    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
        'RESET': '\033[0m'      # Reset
    }
    
    def format(self, record):
        """Format log record with colors."""
        log_color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        reset_color = self.COLORS['RESET']
        
        # Add color to level name
        record.levelname = f"{log_color}{record.levelname}{reset_color}"
        
        return super().format(record)


class VoiceAssistantFilter(logging.Filter):
    """Custom filter for voice assistant logs."""
    
    def __init__(self, component: Optional[str] = None):
        """Initialize filter.
        
        Args:
            component: Specific component to filter for (optional)
        """
        super().__init__()
        self.component = component
    
    def filter(self, record):
        """Filter log records based on component."""
        if self.component:
            return record.name.startswith(f"backend.{self.component}")
        return record.name.startswith("backend")


def setup_logging(
    log_level: Optional[str] = None,
    log_file: Optional[str] = None,
    console_output: bool = True,
    colored_output: bool = True,
    max_file_size: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5,
    component_levels: Optional[Dict[str, str]] = None
) -> logging.Logger:
    """Set up comprehensive logging for the voice assistant.
    
    Args:
        log_level: Global log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file (optional)
        console_output: Whether to output to console
        colored_output: Whether to use colored console output
        max_file_size: Maximum size of log file before rotation
        backup_count: Number of backup log files to keep
        component_levels: Component-specific log levels
        
    Returns:
        logging.Logger: Configured root logger
    """
    # Convert string log level to logging constant
    if log_level:
        numeric_level = getattr(logging, log_level.upper(), DEFAULT_LOG_LEVEL)
    else:
        numeric_level = DEFAULT_LOG_LEVEL
    
    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(numeric_level)
        
        if colored_output and sys.stdout.isatty():
            console_formatter = ColoredFormatter(DEFAULT_LOG_FORMAT, DEFAULT_DATE_FORMAT)
        else:
            console_formatter = logging.Formatter(DEFAULT_LOG_FORMAT, DEFAULT_DATE_FORMAT)
        
        console_handler.setFormatter(console_formatter)
        console_handler.addFilter(VoiceAssistantFilter())
        root_logger.addHandler(console_handler)
    
    # File handler with rotation
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=max_file_size,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(numeric_level)
        
        file_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s",
            DEFAULT_DATE_FORMAT
        )
        file_handler.setFormatter(file_formatter)
        file_handler.addFilter(VoiceAssistantFilter())
        root_logger.addHandler(file_handler)
    
    # Set component-specific log levels
    component_levels = component_levels or {}
    all_levels = {**COMPONENT_LOG_LEVELS, **component_levels}
    
    for component, level in all_levels.items():
        if isinstance(level, str):
            level = getattr(logging, level.upper(), logging.INFO)
        logging.getLogger(component).setLevel(level)
    
    # Log setup completion
    logger = logging.getLogger("backend.logging")
    logger.info(f"Logging configured - Level: {logging.getLevelName(numeric_level)}")
    if log_file:
        logger.info(f"Log file: {log_file}")
    
    return root_logger


def get_logger(name: str) -> logging.Logger:
    """Get a logger for a specific component.
    
    Args:
        name: Logger name (usually __name__)
        
    Returns:
        logging.Logger: Configured logger
    """
    return logging.getLogger(name)


def log_function_call(logger: logging.Logger, func_name: str, **kwargs):
    """Log function call with parameters.
    
    Args:
        logger: Logger instance
        func_name: Function name
        **kwargs: Function parameters to log
    """
    if logger.isEnabledFor(logging.DEBUG):
        params = ", ".join(f"{k}={v}" for k, v in kwargs.items())
        logger.debug(f"Calling {func_name}({params})")


def log_performance(logger: logging.Logger, operation: str, duration: float, **context):
    """Log performance metrics.
    
    Args:
        logger: Logger instance
        operation: Operation name
        duration: Duration in seconds
        **context: Additional context information
    """
    context_str = ", ".join(f"{k}={v}" for k, v in context.items())
    logger.info(f"Performance - {operation}: {duration:.3f}s ({context_str})")


def log_error_with_context(
    logger: logging.Logger, 
    error: Exception, 
    operation: str, 
    **context
):
    """Log error with full context information.
    
    Args:
        logger: Logger instance
        error: Exception that occurred
        operation: Operation that failed
        **context: Additional context information
    """
    context_str = ", ".join(f"{k}={v}" for k, v in context.items())
    logger.error(
        f"Error in {operation}: {type(error).__name__}: {error} ({context_str})",
        exc_info=True
    )


class LoggingContext:
    """Context manager for temporary logging configuration."""
    
    def __init__(self, logger: logging.Logger, level: int):
        """Initialize logging context.
        
        Args:
            logger: Logger to modify
            level: Temporary log level
        """
        self.logger = logger
        self.new_level = level
        self.original_level = logger.level
    
    def __enter__(self):
        """Enter context - set new log level."""
        self.logger.setLevel(self.new_level)
        return self.logger
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context - restore original log level."""
        self.logger.setLevel(self.original_level)


def with_debug_logging(logger: logging.Logger):
    """Context manager for temporary debug logging.
    
    Args:
        logger: Logger to enable debug logging for
        
    Returns:
        LoggingContext: Context manager
    """
    return LoggingContext(logger, logging.DEBUG)


def create_session_logger(session_id: str) -> logging.Logger:
    """Create a session-specific logger.
    
    Args:
        session_id: Unique session identifier
        
    Returns:
        logging.Logger: Session logger with session ID in name
    """
    logger_name = f"backend.session.{session_id}"
    logger = logging.getLogger(logger_name)
    
    # Add session ID to all log messages
    class SessionFilter(logging.Filter):
        def filter(self, record):
            record.session_id = session_id
            return True
    
    logger.addFilter(SessionFilter())
    return logger


# Pre-configured loggers for common components
def get_audio_logger() -> logging.Logger:
    """Get logger for audio components."""
    return get_logger("backend.audio")


def get_memory_logger() -> logging.Logger:
    """Get logger for memory components."""
    return get_logger("backend.memory")


def get_llm_logger() -> logging.Logger:
    """Get logger for LLM components."""
    return get_logger("backend.llm")


def get_core_logger() -> logging.Logger:
    """Get logger for core components."""
    return get_logger("backend.core")