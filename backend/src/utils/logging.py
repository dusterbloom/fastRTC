"""Logging configuration for FastRTC Voice Assistant.

This module provides centralized logging configuration with proper formatters,
handlers, and log levels. It extracts and improves upon the logging setup
from the original monolithic implementation.
"""

import logging
import os
import json
import sys
import time
import functools
from logging.handlers import RotatingFileHandler
from typing import Optional, Dict, Any, List, Callable, Union
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
    # External libraries - suppress in INFO mode
    "httpx": logging.WARNING,
    "urllib3": logging.WARNING,
    "requests": logging.WARNING,
    "transformers": logging.WARNING,
    "torch": logging.WARNING,
    "tensorflow": logging.ERROR,
    "absl": logging.ERROR,
    "onnxruntime": logging.WARNING,
    "faster_whisper": logging.WARNING,
    # Suppress CUDA/GPU warnings in INFO mode
    "numba": logging.WARNING,
    "cupy": logging.WARNING,
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


class ConversationLogger:
    """Special logger for prominent USER-ASSISTANT conversation display."""
    
    def __init__(self, logger: logging.Logger, show_conversation: bool = True):
        """Initialize conversation logger.
        
        Args:
            logger: Underlying logger instance
            show_conversation: Whether to display conversations (from environment)
        """
        self.logger = logger
        self.show_conversation = show_conversation
        
        # ANSI color codes for conversation
        self.USER_COLOR = '\033[94m'     # Blue
        self.ASSISTANT_COLOR = '\033[95m' # Magenta
        self.BOX_COLOR = '\033[90m'      # Gray
        self.RESET = '\033[0m'
    
    def log_user_input(self, text: str):
        """Log user input with prominent formatting.
        
        Args:
            text: User's input text
        """
        if not self.show_conversation or not text.strip():
            return
            
        # Create boxed user message
        message_lines = self._wrap_text(text, 70)
        box_width = max(len(line) for line in message_lines) + 4
        
        # Top border
        top_border = f"{self.BOX_COLOR}┌{'─' * (box_width - 2)}┐{self.RESET}"
        
        # Message lines
        formatted_lines = []
        for i, line in enumerate(message_lines):
            if i == 0:
                prefix = f"{self.USER_COLOR}💬 USER:{self.RESET} "
            else:
                prefix = "        "  # Indent continuation lines
            
            padding = " " * (box_width - len(prefix) - len(line) - 3)
            formatted_line = f"{self.BOX_COLOR}│{self.RESET} {prefix}{line}{padding}{self.BOX_COLOR}│{self.RESET}"
            formatted_lines.append(formatted_line)
        
        # Bottom border
        bottom_border = f"{self.BOX_COLOR}└{'─' * (box_width - 2)}┘{self.RESET}"
        
        # Log the complete box
        self.logger.info("")  # Empty line for separation
        self.logger.info(top_border)
        for line in formatted_lines:
            self.logger.info(line)
        self.logger.info(bottom_border)
    
    def log_assistant_response(self, text: str):
        """Log assistant response with prominent formatting.
        
        Args:
            text: Assistant's response text
        """
        if not self.show_conversation or not text.strip():
            return
            
        # Create boxed assistant message
        message_lines = self._wrap_text(text, 70)
        box_width = max(len(line) for line in message_lines) + 4
        
        # Top border
        top_border = f"{self.BOX_COLOR}┌{'─' * (box_width - 2)}┐{self.RESET}"
        
        # Message lines
        formatted_lines = []
        for i, line in enumerate(message_lines):
            if i == 0:
                prefix = f"{self.ASSISTANT_COLOR}🤖 ASSISTANT:{self.RESET} "
            else:
                prefix = "             "  # Indent continuation lines
            
            padding = " " * (box_width - len(prefix) - len(line) - 3)
            formatted_line = f"{self.BOX_COLOR}│{self.RESET} {prefix}{line}{padding}{self.BOX_COLOR}│{self.RESET}"
            formatted_lines.append(formatted_line)
        
        # Bottom border
        bottom_border = f"{self.BOX_COLOR}└{'─' * (box_width - 2)}┘{self.RESET}"
        
        # Log the complete box
        self.logger.info(top_border)
        for line in formatted_lines:
            self.logger.info(line)
        self.logger.info(bottom_border)
        self.logger.info("")  # Empty line for separation
    
    def _wrap_text(self, text: str, width: int) -> List[str]:
        """Wrap text to specified width.
        
        Args:
            text: Text to wrap
            width: Maximum line width
            
        Returns:
            List[str]: Wrapped lines
        """
        words = text.split()
        lines = []
        current_line = []
        current_length = 0
        
        for word in words:
            # Check if adding this word would exceed width
            word_length = len(word)
            if current_length + word_length + len(current_line) > width and current_line:
                # Start new line
                lines.append(" ".join(current_line))
                current_line = [word]
                current_length = word_length
            else:
                # Add to current line
                current_line.append(word)
                current_length += word_length
        
        # Add remaining words
        if current_line:
            lines.append(" ".join(current_line))
        
        return lines if lines else [""]


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
    
    # Set component-specific log levels (respecting global minimum)
    component_levels = component_levels or {}
    all_levels = {**COMPONENT_LOG_LEVELS, **component_levels}
    
    for component, level in all_levels.items():
        if isinstance(level, str):
            level = getattr(logging, level.upper(), logging.INFO)
        
        # Ensure component level is not lower than global level
        effective_level = max(level, numeric_level)
        logging.getLogger(component).setLevel(effective_level)
    
    # Enforce global level on all existing loggers
    for name in logging.Logger.manager.loggerDict:
        existing_logger = logging.getLogger(name)
        if existing_logger.level < numeric_level:
            existing_logger.setLevel(numeric_level)
    
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


def get_conversation_logger(name: str) -> ConversationLogger:
    """Get a conversation logger for prominent USER-ASSISTANT display.
    
    Args:
        name: Logger name (usually __name__)
        
    Returns:
        ConversationLogger: Configured conversation logger
    """
    import os
    base_logger = get_logger(name)
    show_conversation = os.getenv("SHOW_CONVERSATION", "true").lower() == "true"
    return ConversationLogger(base_logger, show_conversation)


# Enhanced Performance Monitoring Decorators
def time_it(logger: Optional[logging.Logger] = None, 
           operation: Optional[str] = None,
           log_level: int = logging.INFO,
           enabled_env_var: Optional[str] = None) -> Callable:
    """
    General timing decorator for any function.
    
    Args:
        logger: Logger instance to use (defaults to function's module logger)
        operation: Operation name for logging (defaults to function name)
        log_level: Log level for timing info (default: INFO)
        enabled_env_var: Environment variable to check if timing is enabled
        
    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Check if timing is enabled
            if enabled_env_var and not os.getenv(enabled_env_var, "false").lower() == "true":
                return func(*args, **kwargs)
            
            # Get logger
            func_logger = logger or get_logger(func.__module__)
            op_name = operation or func.__name__
            
            # Time the function
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                func_logger.log(log_level, f"⏱️ {op_name}: {duration:.3f}s")
                return result
            except Exception as e:
                duration = time.time() - start_time
                func_logger.log(log_level, f"⏱️ {op_name}: {duration:.3f}s (failed: {e})")
                raise
        
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Check if timing is enabled
            if enabled_env_var and not os.getenv(enabled_env_var, "false").lower() == "true":
                return await func(*args, **kwargs)
            
            # Get logger
            func_logger = logger or get_logger(func.__module__)
            op_name = operation or func.__name__
            
            # Time the async function
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                duration = time.time() - start_time
                func_logger.log(log_level, f"⏱️ {op_name}: {duration:.3f}s")
                return result
            except Exception as e:
                duration = time.time() - start_time
                func_logger.log(log_level, f"⏱️ {op_name}: {duration:.3f}s (failed: {e})")
                raise
        
        # Return appropriate wrapper based on function type
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return wrapper
    
    return decorator


def time_tts(operation: Optional[str] = None, log_level: int = logging.DEBUG) -> Callable:
    """
    TTS-specific timing decorator.
    
    Args:
        operation: Operation name (defaults to function name)
        log_level: Log level for timing info
        
    Returns:
        Decorator function
    """
    return time_it(
        logger=get_logger("backend.audio.tts"),
        operation=operation,
        log_level=log_level,
        enabled_env_var="DEBUG_TTS"
    )


def time_streaming(operation: Optional[str] = None, log_level: int = logging.DEBUG) -> Callable:
    """
    Streaming pipeline timing decorator.
    
    Args:
        operation: Operation name (defaults to function name)
        log_level: Log level for timing info
        
    Returns:
        Decorator function
    """
    return time_it(
        logger=get_logger("backend.integration.streaming"),
        operation=operation,
        log_level=log_level,
        enabled_env_var="DEBUG_STREAMING"
    )


def time_memory(operation: Optional[str] = None, log_level: int = logging.DEBUG) -> Callable:
    """
    Memory system timing decorator.
    
    Args:
        operation: Operation name (defaults to function name)
        log_level: Log level for timing info
        
    Returns:
        Decorator function
    """
    return time_it(
        logger=get_logger("backend.memory"),
        operation=operation,
        log_level=log_level,
        enabled_env_var="DEBUG_MEMORY"
    )


def time_general(operation: Optional[str] = None, log_level: int = logging.DEBUG) -> Callable:
    """
    General timing decorator enabled by DEBUG_TIMING.
    
    Args:
        operation: Operation name (defaults to function name)
        log_level: Log level for timing info
        
    Returns:
        Decorator function
    """
    return time_it(
        operation=operation,
        log_level=log_level,
        enabled_env_var="DEBUG_TIMING"
    )


class PerformanceTimer:
    """Context manager for timing code blocks with detailed logging."""
    
    def __init__(self, 
                 logger: logging.Logger,
                 operation: str,
                 log_level: int = logging.INFO,
                 enabled_env_var: Optional[str] = None):
        """
        Initialize performance timer.
        
        Args:
            logger: Logger instance
            operation: Operation name
            log_level: Log level for timing info
            enabled_env_var: Environment variable to check if enabled
        """
        self.logger = logger
        self.operation = operation
        self.log_level = log_level
        self.enabled = (not enabled_env_var or 
                       os.getenv(enabled_env_var, "false").lower() == "true")
        self.start_time = None
        
    def __enter__(self):
        """Start timing."""
        if self.enabled:
            self.start_time = time.time()
            self.logger.log(self.log_level, f"🚀 Starting {self.operation}")
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """End timing and log results."""
        if self.enabled and self.start_time:
            duration = time.time() - self.start_time
            if exc_type:
                self.logger.log(self.log_level, f"❌ {self.operation}: {duration:.3f}s (failed: {exc_val})")
            else:
                self.logger.log(self.log_level, f"✅ {self.operation}: {duration:.3f}s")
    
    def checkpoint(self, checkpoint_name: str):
        """Log an intermediate checkpoint."""
        if self.enabled and self.start_time:
            duration = time.time() - self.start_time
            self.logger.log(self.log_level, f"🔄 {self.operation} - {checkpoint_name}: {duration:.3f}s")


def create_tts_timer(operation: str) -> PerformanceTimer:
    """Create a TTS performance timer."""
    return PerformanceTimer(
        get_logger("backend.audio.tts"),
        operation,
        logging.DEBUG,
        "DEBUG_TTS"
    )


def create_streaming_timer(operation: str) -> PerformanceTimer:
    """Create a streaming pipeline performance timer."""
    return PerformanceTimer(
        get_logger("backend.integration.streaming"),
        operation,
        logging.DEBUG,
        "DEBUG_STREAMING"
    )


def create_memory_timer(operation: str) -> PerformanceTimer:
    """Create a memory system performance timer."""
    return PerformanceTimer(
        get_logger("backend.memory"),
        operation,
        logging.DEBUG,
        "DEBUG_MEMORY"
    )


def create_general_timer(operation: str) -> PerformanceTimer:
    """Create a general performance timer."""
    return PerformanceTimer(
        get_logger("backend.core"),
        operation,
        logging.DEBUG,
        "DEBUG_TIMING"
    )