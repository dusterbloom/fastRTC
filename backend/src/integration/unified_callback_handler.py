"""
Unified Callback Handler

Provides a unified interface that can switch between the original async callback
handler and the new threading-based handler based on configuration.
"""

import logging
from typing import Tuple, Generator, Any, Optional

from .callback_handler import StreamCallbackHandler
from .threading_callback_handler import ThreadingCallbackHandler
from ..config.threading_config import is_threading_enabled, should_fallback_to_async
from ..utils.logging import get_logger

logger = get_logger(__name__)


class UnifiedCallbackHandler:
    """
    Unified callback handler that switches between async and threading implementations.
    
    Provides seamless transition between old and new architectures with automatic
    fallback capabilities for safe deployment.
    """
    
    def __init__(
        self,
        voice_assistant,
        stt_engine,
        tts_engine,
        voice_mapper,
        event_loop=None
    ):
        """
        Initialize unified callback handler.
        
        Args:
            voice_assistant: Voice assistant instance
            stt_engine: STT engine
            tts_engine: TTS engine
            voice_mapper: Voice mapper
            event_loop: Event loop (for async handler)
        """
        self.voice_assistant = voice_assistant
        self.stt_engine = stt_engine
        self.tts_engine = tts_engine
        self.voice_mapper = voice_mapper
        self.event_loop = event_loop
        
        # Handler instances
        self.async_handler: Optional[StreamCallbackHandler] = None
        self.threading_handler: Optional[ThreadingCallbackHandler] = None
        self.active_handler = None
        self.handler_type = None
        
        # Initialize based on configuration
        self._initialize_handlers()
        
    def _initialize_handlers(self):
        """Initialize the appropriate handler based on configuration."""
        try:
            if is_threading_enabled():
                logger.info("🧵 Threading pipeline enabled - initializing threading handler")
                self._initialize_threading_handler()
            else:
                logger.info("🔄 Threading pipeline disabled - using async handler")
                self._initialize_async_handler()
                
        except Exception as e:
            logger.error(f"Error initializing handlers: {e}")
            self._handle_initialization_error()
            
    def _initialize_threading_handler(self):
        """Initialize threading-based handler."""
        try:
            self.threading_handler = ThreadingCallbackHandler(
                voice_assistant=self.voice_assistant,
                stt_engine=self.stt_engine,
                tts_engine=self.tts_engine,
                voice_mapper=self.voice_mapper
            )
            
            self.threading_handler.start()
            self.active_handler = self.threading_handler
            self.handler_type = "threading"
            
            logger.info("✅ Threading handler initialized and started")
            logger.info(f"🔍 THREADING HANDLER TYPE: {type(self.threading_handler).__name__}")
            
        except Exception as e:
            logger.error(f"Failed to initialize threading handler: {e}")
            
            if should_fallback_to_async():
                logger.info("🔄 Falling back to async handler")
                self._initialize_async_handler()
            else:
                raise
                
    def _initialize_async_handler(self):
        """Initialize async-based handler."""
        try:
            self.async_handler = StreamCallbackHandler(
                voice_assistant=self.voice_assistant,
                stt_engine=self.stt_engine,
                tts_engine=self.tts_engine,
                voice_mapper=self.voice_mapper,
                event_loop=self.event_loop
            )
            
            self.active_handler = self.async_handler
            self.handler_type = "async"
            
            logger.info("✅ Async handler initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize async handler: {e}")
            raise
            
    def _handle_initialization_error(self):
        """Handle initialization errors with fallback logic."""
        if should_fallback_to_async() and not self.async_handler:
            logger.warning("⚠️ Attempting fallback to async handler")
            try:
                self._initialize_async_handler()
            except Exception as fallback_error:
                logger.error(f"Fallback initialization failed: {fallback_error}")
                raise
        else:
            logger.error("No fallback available or fallback disabled")
            raise
            
    def process_audio_stream(self, audio_data_tuple: tuple) -> Generator[Tuple[Tuple[int, Any], Any], None, None]:
        """
        Process audio stream using the active handler.
        
        Args:
            audio_data_tuple: Audio data from FastRTC
            
        Yields:
            Audio response chunks
        """
        # DEBUG: Log that callback was invoked
        logger.info(f"🎤 UNIFIED CALLBACK INVOKED: handler_type={self.handler_type}, audio_data_type={type(audio_data_tuple)}")
        logger.info(f"🔍 UNIFIED: Active handler = {type(self.active_handler).__name__}")
        
        if not self.active_handler:
            logger.error("No active handler available")
            return
            
        try:
            # Delegate to active handler
            logger.info(f"🎤 Delegating to {self.handler_type} handler...")
            yield from self.active_handler.process_audio_stream(audio_data_tuple)
            
        except Exception as e:
            logger.error(f"Error in {self.handler_type} handler: {e}")
            
            # Attempt automatic recovery
            if self.handler_type == "threading" and should_fallback_to_async():
                logger.warning("🔄 Threading handler failed, switching to async")
                self._switch_to_async_handler()
                
                # Retry with async handler
                if self.active_handler:
                    yield from self.active_handler.process_audio_stream(audio_data_tuple)
            else:
                # Re-raise if no fallback available
                raise
                
    def _switch_to_async_handler(self):
        """Switch from threading to async handler."""
        try:
            # Stop threading handler
            if self.threading_handler:
                self.threading_handler.stop()
                
            # Initialize async handler if not already done
            if not self.async_handler:
                self._initialize_async_handler()
            else:
                self.active_handler = self.async_handler
                self.handler_type = "async"
                
            logger.info("✅ Switched to async handler")
            
        except Exception as e:
            logger.error(f"Failed to switch to async handler: {e}")
            self.active_handler = None
            self.handler_type = None
            
    def get_handler_stats(self) -> dict:
        """Get statistics from the active handler."""
        base_stats = {
            "handler_type": self.handler_type,
            "threading_enabled": is_threading_enabled(),
            "fallback_enabled": should_fallback_to_async(),
        }
        
        if self.active_handler and hasattr(self.active_handler, 'get_handler_stats'):
            handler_stats = self.active_handler.get_handler_stats()
            base_stats.update(handler_stats)
            
        return base_stats
        
    def stop(self):
        """Stop all handlers."""
        logger.info("🛑 Stopping unified callback handler")
        
        if self.threading_handler:
            try:
                self.threading_handler.stop()
            except Exception as e:
                logger.error(f"Error stopping threading handler: {e}")
                
        if self.async_handler:
            try:
                # Async handler doesn't have a stop method, but clean up if needed
                pass
            except Exception as e:
                logger.error(f"Error stopping async handler: {e}")
                
        self.active_handler = None
        self.handler_type = None
        
        logger.info("✅ Unified callback handler stopped")
        
    def __del__(self):
        """Cleanup on destruction."""
        try:
            self.stop()
        except:
            pass