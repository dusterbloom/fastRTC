"""
Async Utilities Module

Provides utilities for running coroutines from sync contexts and handling
async operations in the voice assistant system.
"""

import asyncio
import threading
import time
from typing import Any, Optional, Coroutine
from ..utils.logging import get_logger

logger = get_logger(__name__)


def run_coro_from_sync_thread_with_timeout(
    coro: Coroutine,
    timeout: float = 4.0,
    event_loop: Optional[asyncio.AbstractEventLoop] = None
) -> Any:
    """
    Run a coroutine from a synchronous thread with timeout protection.
    
    This function is essential for preventing WebRTC disconnections by ensuring
    that async operations complete within reasonable time limits.
    
    Args:
        coro: The coroutine to execute
        timeout: Maximum time to wait for completion (seconds)
        event_loop: Optional event loop to use (if None, uses global loop)
        
    Returns:
        The result of the coroutine execution
        
    Raises:
        TimeoutError: If the operation exceeds the timeout
        RuntimeError: If no event loop is available
    """
    if not event_loop:
        # Try to get the running event loop from the current thread
        try:
            event_loop = asyncio.get_running_loop()
        except RuntimeError:
            # No running loop in current thread, this is expected for sync contexts
            event_loop = None
    
    if not event_loop or not event_loop.is_running():
        logger.error("❌ Event loop not available or not running")
        raise RuntimeError("Event loop not available for async operation")
    
    try:
        # Schedule the coroutine with timeout in the event loop
        future = asyncio.run_coroutine_threadsafe(
            asyncio.wait_for(coro, timeout=timeout),
            event_loop
        )
        
        # Wait for completion with additional buffer time
        result = future.result(timeout=timeout + 1.0)
        return result
        
    except asyncio.TimeoutError:
        logger.warning(f"❌ Async task timed out after {timeout}s")
        raise TimeoutError(f"Operation timed out after {timeout}s")
    except Exception as e:
        logger.error(f"❌ Error in async task: {e}")
        return "I encountered an error processing your request."


class AsyncEnvironmentManager:
    """
    Manages the async environment for the voice assistant system.
    
    This class handles the creation and lifecycle of the async event loop
    that runs in a separate thread to support real-time audio processing.
    """
    
    def __init__(self):
        """Initialize the async environment manager."""
        self.event_loop: Optional[asyncio.AbstractEventLoop] = None
        self.worker_thread: Optional[threading.Thread] = None
        self.is_running = False
        self._shutdown_event = threading.Event()
    
    def setup_async_environment(self, assistant_instance=None) -> bool:
        """
        Set up the async environment with a dedicated event loop thread.
        
        Args:
            assistant_instance: Optional assistant instance to initialize
            
        Returns:
            True if setup was successful, False otherwise
        """
        logger.info("🔧 Setting up async environment...")
        
        def run_async_loop_in_thread():
            """Run the async event loop in a dedicated thread."""
            try:
                self.event_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(self.event_loop)
                
                # Initialize assistant async components if provided
                if assistant_instance and hasattr(assistant_instance, 'initialize_async'):
                    self.event_loop.run_until_complete(
                        assistant_instance.initialize_async()
                    )
                
                self.is_running = True
                logger.info("✅ Async event loop started")
                
                # Run the event loop until shutdown
                self.event_loop.run_forever()
                
            except Exception as e:
                logger.error(f"❌ Error in async thread: {e}")
            finally:
                self._cleanup_loop(assistant_instance)
        
        # Start the async worker thread
        self.worker_thread = threading.Thread(
            target=run_async_loop_in_thread,
            daemon=True,
            name="AsyncWorkerThread"
        )
        self.worker_thread.start()
        
        # Wait for the event loop to be ready
        return self._wait_for_readiness(assistant_instance)
    
    def _wait_for_readiness(self, assistant_instance=None, max_wait_time: float = 10.0) -> bool:
        """
        Wait for the async environment to be ready.
        
        Args:
            assistant_instance: Optional assistant instance to check
            max_wait_time: Maximum time to wait for readiness
            
        Returns:
            True if ready, False if timeout
        """
        start_time = time.monotonic()
        
        while time.monotonic() - start_time < max_wait_time:
            if (self.event_loop and 
                self.event_loop.is_running() and 
                self.is_running):
                
                # Additional checks for assistant readiness if provided
                if assistant_instance:
                    if (hasattr(assistant_instance, 'http_session') and
                        assistant_instance.http_session and
                        hasattr(assistant_instance, 'amem_memory') and
                        assistant_instance.amem_memory):
                        
                        logger.info("✅ Async environment and components are ready")
                        return True
                else:
                    logger.info("✅ Async environment is ready")
                    return True
            
            time.sleep(0.1)
        
        logger.warning("⚠️ Async environment did not confirm readiness in time")
        return False
    
    def _cleanup_loop(self, assistant_instance=None):
        """Clean up the async event loop and resources."""
        try:
            if assistant_instance and hasattr(assistant_instance, 'cleanup_async'):
                if self.event_loop and not self.event_loop.is_closed():
                    logger.info("🧹 Cleaning up assistant resources...")
                    self.event_loop.run_until_complete(
                        assistant_instance.cleanup_async()
                    )
            
            if self.event_loop and not self.event_loop.is_closed():
                self.event_loop.close()
                logger.info("🧹 Async event loop closed")
                
        except Exception as e:
            logger.error(f"❌ Error during cleanup: {e}")
        finally:
            self.is_running = False
    
    def shutdown(self, timeout: float = 15.0):
        """
        Shutdown the async environment gracefully.
        
        Args:
            timeout: Maximum time to wait for shutdown
        """
        logger.info("🛑 Shutting down async environment...")
        
        if self.event_loop and not self.event_loop.is_closed():
            # Request the event loop to stop
            self.event_loop.call_soon_threadsafe(self.event_loop.stop)
        
        if self.worker_thread and self.worker_thread.is_alive():
            logger.info("⏳ Waiting for async worker thread to join...")
            self.worker_thread.join(timeout=timeout)
            
            if self.worker_thread.is_alive():
                logger.warning("⚠️ Async worker thread did not join in time")
            else:
                logger.info("✅ Async worker thread joined successfully")
        
        self.is_running = False
        logger.info("🏁 Async environment shutdown complete")
    
    def get_event_loop(self) -> Optional[asyncio.AbstractEventLoop]:
        """Get the current event loop."""
        return self.event_loop
    
    def is_ready(self) -> bool:
        """Check if the async environment is ready."""
        return (self.is_running and 
                self.event_loop and 
                self.event_loop.is_running())