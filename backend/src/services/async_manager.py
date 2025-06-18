"""Async lifecycle management for FastRTC Voice Assistant.

This module provides async component lifecycle management including
startup, shutdown, and health monitoring of async components.
"""

import asyncio
import threading
import time
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass

from ..core.interfaces import AsyncLifecycleManager as AsyncLifecycleManagerInterface
from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ManagedComponent:
    """A managed async component.
    
    Attributes:
        name: Component name
        instance: Component instance
        startup_func: Startup function (optional)
        shutdown_func: Shutdown function (optional)
        health_check_func: Health check function (optional)
        is_started: Whether component is started
        startup_timeout: Timeout for startup in seconds
        shutdown_timeout: Timeout for shutdown in seconds
    """
    name: str
    instance: Any
    startup_func: Optional[Callable] = None
    shutdown_func: Optional[Callable] = None
    health_check_func: Optional[Callable] = None
    is_started: bool = False
    startup_timeout: float = 30.0
    shutdown_timeout: float = 10.0


class AsyncManager(AsyncLifecycleManagerInterface):
    """Async lifecycle manager for managing component lifecycles.
    
    This class handles startup, shutdown, and health monitoring
    of async components with proper error handling and timeouts.
    """
    
    def __init__(self):
        """Initialize the async manager."""
        self.components: Dict[str, ManagedComponent] = {}
        self.event_loop: Optional[asyncio.AbstractEventLoop] = None
        self.worker_thread: Optional[threading.Thread] = None
        self.is_running = False
        self._shutdown_event = asyncio.Event()
        
        logger.info("Async manager initialized")
    
    async def initialize(self):
        """Initialize the async manager (async version of setup)."""
        logger.info("🔧 Initializing async manager...")
        # This method can be used for any async initialization
        # Currently, the main setup is done in setup_async_environment
        logger.info("✅ Async manager initialization complete")
    
    async def cleanup(self):
        """Cleanup async manager resources."""
        logger.info("🧹 Cleaning up async manager...")
        # Shutdown all components
        await self.shutdown()
        logger.info("✅ Async manager cleanup complete")
    
    def register_component(self,
                          name: str, 
                          instance: Any,
                          startup_func: Optional[Callable] = None,
                          shutdown_func: Optional[Callable] = None,
                          health_check_func: Optional[Callable] = None,
                          startup_timeout: float = 30.0,
                          shutdown_timeout: float = 10.0):
        """Register a component for lifecycle management.
        
        Args:
            name: Component name
            instance: Component instance
            startup_func: Optional startup function
            shutdown_func: Optional shutdown function
            health_check_func: Optional health check function
            startup_timeout: Startup timeout in seconds
            shutdown_timeout: Shutdown timeout in seconds
        """
        component = ManagedComponent(
            name=name,
            instance=instance,
            startup_func=startup_func,
            shutdown_func=shutdown_func,
            health_check_func=health_check_func,
            startup_timeout=startup_timeout,
            shutdown_timeout=shutdown_timeout
        )
        
        self.components[name] = component
        logger.info(f"Registered component: {name}")
    
    def setup_async_environment(self):
        """Setup async environment in a separate thread."""
        if self.worker_thread and self.worker_thread.is_alive():
            logger.warning("Async environment already running")
            return
        
        def run_async_loop():
            """Run the async event loop in a separate thread."""
            self.event_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.event_loop)
            
            try:
                # Start all components
                startup_task = self.event_loop.create_task(self.startup())
                self.event_loop.run_until_complete(startup_task)
                
                if not startup_task.result():
                    logger.error("Failed to start all components")
                    return
                
                self.is_running = True
                logger.info("Async environment ready, running event loop")
                
                # Run the event loop
                self.event_loop.run_forever()
                
            except KeyboardInterrupt:
                logger.info("Async loop interrupted")
            except Exception as e:
                logger.error(f"Error in async loop: {e}")
            finally:
                # Cleanup
                if self.is_running:
                    logger.info("Cleaning up async environment...")
                    shutdown_task = self.event_loop.create_task(self.shutdown())
                    self.event_loop.run_until_complete(shutdown_task)
                
                if not self.event_loop.is_closed():
                    self.event_loop.close()
                
                self.is_running = False
                logger.info("Async event loop closed")
        
        self.worker_thread = threading.Thread(
            target=run_async_loop, 
            daemon=True, 
            name="AsyncManagerThread"
        )
        self.worker_thread.start()
        
        # Wait for startup to complete
        max_wait_time = 100  # 10 seconds
        for _ in range(max_wait_time):
            if (self.event_loop and 
                self.event_loop.is_running() and 
                self.is_running):
                logger.info("✅ Async environment and components are ready")
                return
            time.sleep(0.1)
        
        logger.warning("⚠️ Async environment did not confirm readiness in time")
    
    async def startup(self) -> bool:
        """Start all managed components.
        
        Returns:
            bool: True if all components started successfully
        """
        logger.info("Starting async components...")
        
        success_count = 0
        total_components = len(self.components)
        
        for name, component in self.components.items():
            try:
                logger.info(f"Starting component: {name}")
                
                if component.startup_func:
                    # Call startup function with timeout
                    await asyncio.wait_for(
                        component.startup_func(),
                        timeout=component.startup_timeout
                    )
                
                component.is_started = True
                success_count += 1
                logger.info(f"✅ Component {name} started successfully")
                
            except asyncio.TimeoutError:
                logger.error(f"❌ Component {name} startup timed out after {component.startup_timeout}s")
            except Exception as e:
                logger.error(f"❌ Failed to start component {name}: {e}")
        
        success = success_count == total_components
        if success:
            logger.info(f"✅ All {total_components} components started successfully")
        else:
            logger.error(f"❌ Only {success_count}/{total_components} components started successfully")
        
        return success
    
    async def shutdown(self) -> bool:
        """Shutdown all managed components gracefully.
        
        Returns:
            bool: True if all components shut down successfully
        """
        logger.info("Shutting down async components...")
        
        success_count = 0
        total_components = len([c for c in self.components.values() if c.is_started])
        
        # Shutdown components in reverse order
        component_items = list(self.components.items())
        component_items.reverse()
        
        for name, component in component_items:
            if not component.is_started:
                continue
                
            try:
                logger.info(f"Shutting down component: {name}")
                
                if component.shutdown_func:
                    # Call shutdown function with timeout
                    await asyncio.wait_for(
                        component.shutdown_func(),
                        timeout=component.shutdown_timeout
                    )
                
                component.is_started = False
                success_count += 1
                logger.info(f"✅ Component {name} shut down successfully")
                
            except asyncio.TimeoutError:
                logger.error(f"❌ Component {name} shutdown timed out after {component.shutdown_timeout}s")
            except Exception as e:
                logger.error(f"❌ Failed to shutdown component {name}: {e}")
        
        success = success_count == total_components
        if success:
            logger.info(f"✅ All {total_components} components shut down successfully")
        else:
            logger.error(f"❌ Only {success_count}/{total_components} components shut down successfully")
        
        return success
    
    async def health_check(self) -> Dict[str, bool]:
        """Check health of all managed components.
        
        Returns:
            Dict[str, bool]: Component name to health status mapping
        """
        health_status = {}
        
        for name, component in self.components.items():
            if not component.is_started:
                health_status[name] = False
                continue
            
            try:
                if component.health_check_func:
                    # Call health check function with timeout
                    is_healthy = await asyncio.wait_for(
                        component.health_check_func(),
                        timeout=5.0
                    )
                    health_status[name] = bool(is_healthy)
                else:
                    # If no health check function, assume healthy if started
                    health_status[name] = True
                    
            except asyncio.TimeoutError:
                logger.warning(f"Health check for {name} timed out")
                health_status[name] = False
            except Exception as e:
                logger.warning(f"Health check for {name} failed: {e}")
                health_status[name] = False
        
        healthy_count = sum(health_status.values())
        total_count = len(health_status)
        logger.debug(f"Health check: {healthy_count}/{total_count} components healthy")
        
        return health_status
    
    def run_coroutine_threadsafe(self, coro, timeout: float = 4.0) -> Any:
        """Run coroutine from sync thread with timeout.
        
        Args:
            coro: Coroutine to run
            timeout: Timeout in seconds
            
        Returns:
            Any: Coroutine result
            
        Raises:
            TimeoutError: If operation times out
            RuntimeError: If event loop not available
        """
        if not self.event_loop or not self.event_loop.is_running():
            raise RuntimeError("Event loop not available")
        
        try:
            future = asyncio.run_coroutine_threadsafe(
                asyncio.wait_for(coro, timeout=timeout),
                self.event_loop
            )
            return future.result(timeout=timeout + 1.0)  # Add 1s buffer
            
        except asyncio.TimeoutError:
            logger.error(f"❌ Async task timed out after {timeout}s")
            raise TimeoutError(f"Operation timed out after {timeout}s")
        except Exception as e:
            logger.error(f"❌ Error in async task: {e}")
            raise
    
    def stop_async_environment(self):
        """Stop the async environment gracefully."""
        if not self.event_loop or not self.event_loop.is_running():
            logger.warning("Async environment not running")
            return
        
        logger.info("Stopping async environment...")
        
        # Schedule shutdown in the event loop
        def stop_loop():
            if self.event_loop and self.event_loop.is_running():
                self.event_loop.stop()
        
        self.event_loop.call_soon_threadsafe(stop_loop)
        
        # Wait for thread to finish
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=15.0)
            if self.worker_thread.is_alive():
                logger.warning("Worker thread did not finish in time")
        
        self.is_running = False
        logger.info("Async environment stopped")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get async manager statistics.
        
        Returns:
            Dict[str, Any]: Statistics dictionary
        """
        started_components = [c for c in self.components.values() if c.is_started]
        
        return {
            'total_components': len(self.components),
            'started_components': len(started_components),
            'is_running': self.is_running,
            'has_event_loop': self.event_loop is not None,
            'event_loop_running': (self.event_loop.is_running() 
                                 if self.event_loop else False),
            'worker_thread_alive': (self.worker_thread.is_alive() 
                                  if self.worker_thread else False),
            'component_names': list(self.components.keys()),
            'started_component_names': [c.name for c in started_components]
        }
    
    def is_available(self) -> bool:
        """Check if async manager is available and ready.
        
        Returns:
            bool: True if manager is ready, False otherwise
        """
        return (self.is_running and 
                self.event_loop is not None and 
                self.event_loop.is_running())
    
    def __del__(self):
        """Cleanup on destruction."""
        if self.is_running:
            try:
                self.stop_async_environment()
            except Exception as e:
                logger.error(f"Error during async manager cleanup: {e}")