"""
Main Application Entry Point

Application entry point for the FastRTC voice assistant system.
Handles dependency injection, component initialization, and application lifecycle.
"""

import sys
import signal
import asyncio
from typing import Optional

from .voice_assistant import VoiceAssistant
from ..integration import FastRTCBridge, StreamCallbackHandler
from ..utils.async_utils import AsyncEnvironmentManager
from ..utils.logging import get_logger, setup_logging
from ..config.settings import DEFAULT_SPEECH_THRESHOLD

logger = get_logger(__name__)


class VoiceAssistantApplication:
    """
    Main application class that orchestrates the entire voice assistant system.
    
    This class handles:
    - Component initialization and dependency injection
    - Async environment setup and lifecycle management
    - FastRTC integration and stream management
    - Graceful shutdown and cleanup
    """
    
    def __init__(self):
        """Initialize the voice assistant application."""
        self.voice_assistant: Optional[VoiceAssistant] = None
        self.fastrtc_bridge: Optional[FastRTCBridge] = None
        self.callback_handler: Optional[StreamCallbackHandler] = None
        self.async_env_manager: Optional[AsyncEnvironmentManager] = None
        self.is_running = False
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        logger.info(f"🛑 Received signal {signum}. Initiating graceful shutdown...")
        self.shutdown()
    
    async def initialize(self) -> bool:
        """
        Initialize all application components with dependency injection.
        
        Returns:
            True if initialization was successful, False otherwise
        """
        try:
            logger.info("🚀 Initializing FastRTC Voice Assistant Application...")
            
            # Initialize core voice assistant
            logger.info("🧠 Creating voice assistant instance...")
            self.voice_assistant = VoiceAssistant()
            
            # Initialize FastRTC bridge
            logger.info("🌐 Creating FastRTC bridge...")
            self.fastrtc_bridge = FastRTCBridge()
            
            # Initialize async environment manager
            logger.info("⚡ Setting up async environment...")
            self.async_env_manager = AsyncEnvironmentManager()
            
            # Setup async environment with voice assistant
            success = self.async_env_manager.setup_async_environment(self.voice_assistant)
            if not success:
                logger.error("❌ Failed to setup async environment")
                return False
            
            # Get the event loop for callback handler
            event_loop = self.async_env_manager.get_event_loop()
            
            # Initialize stream callback handler
            logger.info("🎤 Creating stream callback handler...")
            self.callback_handler = StreamCallbackHandler(
                voice_assistant=self.voice_assistant,
                stt_engine=self.voice_assistant.stt_engine,
                tts_engine=self.voice_assistant.tts_engine,
                language_detector=self.voice_assistant.language_detector,
                voice_mapper=self.voice_assistant.voice_mapper,
                event_loop=event_loop
            )
            
            logger.info("✅ Application initialization complete")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize application: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def run(
        self,
        server_name: str = "0.0.0.0",
        server_port: int = 7860,
        share: bool = False,
        speech_threshold: Optional[float] = None
    ):
        """
        Run the voice assistant application.
        
        Args:
            server_name: Server hostname for web interface
            server_port: Server port for web interface
            share: Whether to create a public share link
            speech_threshold: Custom speech detection threshold
        """
        try:
            # Determine speech threshold
            if speech_threshold is None:
                if (self.voice_assistant and 
                    self.voice_assistant.audio_processor.noise_floor):
                    speech_threshold = self.voice_assistant.audio_processor.noise_floor * 15
                else:
                    speech_threshold = DEFAULT_SPEECH_THRESHOLD
            
            logger.info(f"🎯 Using speech threshold: {speech_threshold}")
            
            # Create FastRTC stream with callback
            logger.info("🌐 Creating FastRTC stream...")
            stream = self.fastrtc_bridge.create_stream(
                callback_function=self.callback_handler.process_audio_stream,
                speech_threshold=speech_threshold,
                server_name=server_name,
                server_port=server_port,
                share=share
            )
            
            # Launch the stream
            logger.info("🚀 Launching FastRTC voice assistant...")
            self.is_running = True
            self.fastrtc_bridge.launch_stream(
                server_name=server_name,
                server_port=server_port,
                share=share,
                quiet=False
            )
            
        except KeyboardInterrupt:
            logger.info("🛑 KeyboardInterrupt received. Shutting down...")
            self.shutdown()
        except Exception as e:
            logger.error(f"❌ Application error: {e}")
            import traceback
            traceback.print_exc()
            self.shutdown()
    
    async def start(self):
        """Start the application asynchronously."""
        if self.is_running:
            logger.warning("Application is already running")
            return
        
        # Initialize if not already done
        if not self.voice_assistant:
            success = await self.initialize()
            if not success:
                raise RuntimeError("Failed to initialize application")
        
        # Start the async environment manager
        if self.async_env_manager:
            await self.async_env_manager.start()
        
        self.is_running = True
        logger.info("✅ Application started successfully")
    
    async def stop(self):
        """Stop the application asynchronously."""
        if not self.is_running:
            return
        
        logger.info("🏁 Starting application shutdown sequence...")
        self.is_running = False
        
        try:
            # Stop FastRTC stream
            if self.fastrtc_bridge:
                logger.info("🛑 Stopping FastRTC stream...")
                self.fastrtc_bridge.stop_stream()
            
            # Stop async environment manager
            if self.async_env_manager:
                logger.info("⚡ Stopping async environment...")
                await self.async_env_manager.stop()
            
            logger.info("👋 Voice assistant shutdown complete")
            
        except Exception as e:
            logger.error(f"❌ Error during shutdown: {e}")
    
    def shutdown(self):
        """Shutdown the application gracefully."""
        if not self.is_running:
            return
        
        logger.info("🏁 Starting application shutdown sequence...")
        self.is_running = False
        
        try:
            # Stop FastRTC stream
            if self.fastrtc_bridge:
                logger.info("🛑 Stopping FastRTC stream...")
                self.fastrtc_bridge.stop_stream()
            
            # Shutdown async environment
            if self.async_env_manager:
                logger.info("⚡ Shutting down async environment...")
                self.async_env_manager.shutdown(timeout=15)
            
            logger.info("👋 Voice assistant shutdown complete")
            
        except Exception as e:
            logger.error(f"❌ Error during shutdown: {e}")
        finally:
            sys.exit(0)
    
    def get_status(self) -> dict:
        """
        Get current application status.
        
        Returns:
            Dictionary containing application status information
        """
        status = {
            'is_running': self.is_running,
            'components': {
                'voice_assistant': self.voice_assistant is not None,
                'fastrtc_bridge': self.fastrtc_bridge is not None,
                'callback_handler': self.callback_handler is not None,
                'async_env_manager': self.async_env_manager is not None
            }
        }
        
        # Add component-specific status
        if self.voice_assistant:
            status['voice_assistant'] = self.voice_assistant.get_system_stats()
        
        if self.fastrtc_bridge:
            status['fastrtc_bridge'] = self.fastrtc_bridge.get_stream_status()
        
        if self.async_env_manager:
            status['async_environment'] = {
                'is_ready': self.async_env_manager.is_ready(),
                'event_loop_running': (
                    self.async_env_manager.event_loop and 
                    self.async_env_manager.event_loop.is_running()
                )
            }
        
        return status


def create_voice_assistant() -> VoiceAssistant:
    """
    Factory function to create a configured voice assistant instance.
    
    Returns:
        Configured VoiceAssistant instance
    """
    return VoiceAssistant()


async def create_application() -> VoiceAssistantApplication:
    """
    Factory function to create a configured voice assistant application.
    
    Returns:
        Configured VoiceAssistantApplication instance
    """
    # Setup logging
    setup_logging()
    
    # Create and return application
    return VoiceAssistantApplication()


async def main_async():
    """Async main function for the voice assistant application."""
    app = await create_application()
    
    # Initialize the application
    success = await app.initialize()
    if not success:
        logger.error("❌ Failed to initialize application")
        sys.exit(1)
    
    # Run the application
    app.run()


def main():
    """
    Main entry point for the voice assistant application.
    
    This function sets up the async environment and runs the application.
    """
    try:
        # Run the async main function
        asyncio.run(main_async())
    except KeyboardInterrupt:
        logger.info("🛑 Application interrupted by user")
    except Exception as e:
        logger.error(f"❌ Fatal application error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()