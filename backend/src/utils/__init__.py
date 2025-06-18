"""Utility modules for FastRTC Voice Assistant."""

from .logging import setup_logging, get_logger
from .async_utils import (
    run_coro_from_sync_thread_with_timeout,
    AsyncEnvironmentManager
)

# Alias for backward compatibility
AsyncUtils = AsyncEnvironmentManager

__all__ = [
    "setup_logging",
    "get_logger",
    "run_coro_from_sync_thread_with_timeout",
    "AsyncEnvironmentManager",
    "AsyncUtils",
]