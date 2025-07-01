"""
FastRTC Bridge Module

Handles FastRTC stream setup, configuration, and WebRTC connection lifecycle.
Extracted from the original voice assistant implementation.
"""

import os
import logging
from typing import Optional, Dict, Any, Callable
from fastrtc import Stream, StreamHandler

from ..utils.logging import get_logger

logger = get_logger(__name__)


class WhisperLiveStreamHandler(StreamHandler):
    """
    StreamHandler that wraps the callback function for WhisperLive VAD mode.
    
    This handler simply forwards audio to the callback function without any VAD processing,
    since WhisperLive handles VAD directly.
    """
    
    def __init__(self, callback_function: Callable):
        """Initialize with the callback function."""
        self.callback_function = callback_function
        logger.debug("🎤 WhisperLiveStreamHandler initialized")
    
    def receive(self, audio_data_tuple):
        """Receive audio and forward to callback function."""
        logger.debug(f"🎤 WhisperLiveStreamHandler received audio: {type(audio_data_tuple)}")
        
        # Forward to callback function and yield results
        try:
            for result in self.callback_function(audio_data_tuple):
                yield result
        except Exception as e:
            logger.error(f"Error in WhisperLiveStreamHandler callback: {e}")
            # Yield empty audio to maintain connection
            yield ((16000, b""), None)
    
    def copy(self):
        """Create a copy of this handler."""
        return WhisperLiveStreamHandler(self.callback_function)
    
    def emit(self, frame):
        """Emit audio frame (required by StreamHandler interface)."""
        # This method is typically used for sending audio back to client
        # For our WhisperLive setup, we handle this in the callback
        return frame


class FastRTCBridge:
    """
    Manages FastRTC stream configuration and WebRTC connections.
    
    This class encapsulates all FastRTC-specific logic including:
    - Stream configuration with audio constraints
    - WebRTC connection lifecycle management
    - Audio processing pipeline integration
    """
    
    def __init__(self):
        """Initialize the FastRTC bridge."""
        self.stream: Optional[Stream] = None
        self.is_running = False
        
    def create_stream(
        self,
        callback_function: Callable,
        speech_threshold: float = 0.15,
        server_name: str = "0.0.0.0",
        server_port: int = 7860,
        share: bool = False
    ) -> Stream:
        """
        Create and configure a FastRTC stream with optimized audio settings.
        
        Args:
            callback_function: The audio processing callback function
            speech_threshold: Threshold for speech detection
            server_name: Server hostname for the web interface
            server_port: Server port for the web interface
            share: Whether to create a public share link
            
        Returns:
            Configured FastRTC Stream instance
        """
        logger.info("🌐 Creating FastRTC stream with optimized audio settings...")
        logger.debug(f"🌐 Callback function: {callback_function}")
        logger.debug(f"🌐 Speech threshold: {speech_threshold}")
        logger.debug(f"🌐 Server: {server_name}:{server_port}")
        
        try:
            # Create debug wrapper for callback to log invocations and handle generator
            def debug_callback_wrapper(audio_data_tuple):
                logger.info(f"🎤 FASTRTC CALLBACK INVOKED: audio_data_type={type(audio_data_tuple)}")
                if hasattr(audio_data_tuple, '__len__') and len(audio_data_tuple) >= 2:
                    audio_data, sample_rate = audio_data_tuple[0], audio_data_tuple[1]
                    logger.info(f"🎤 Audio data: {type(audio_data)}, samples: {getattr(audio_data, 'size', 'unknown')}, rate: {sample_rate}")
                
                # Our callback returns a generator, but FastRTC expects direct results
                # Consume the generator and yield each result
                for result in callback_function(audio_data_tuple):
                    yield result
            
            # Create WhisperLive stream handler (no VAD - handled by WhisperLive)
            logger.debug("🌐 Creating WhisperLiveStreamHandler (VAD disabled - handled by WhisperLive)...")
            stream_handler = WhisperLiveStreamHandler(debug_callback_wrapper)
            
            # Create stream with WhisperLive handler
            logger.debug("🌐 Creating Stream with WhisperLiveStreamHandler...")
            self.stream = Stream(
                stream_handler,  # Use proper StreamHandler
                modality="audio",
                mode="send-receive",
                track_constraints=self._get_audio_constraints()
            )
            

            
            logger.info("✅ FastRTC stream created successfully")
            logger.debug(f"✅ Stream object: {self.stream}")
            logger.debug(f"✅ Stream modality: {self.stream.modality if hasattr(self.stream, 'modality') else 'unknown'}")
            return self.stream
            
        except Exception as e:
            logger.error(f"❌ Failed to create FastRTC stream: {e}")
            raise
    
    def _get_audio_constraints(self) -> Dict[str, Any]:
        """
        Get optimized audio track constraints for real-time processing.
        
        Returns:
            Dictionary of audio constraints for WebRTC
        """
        return {
            "echoCancellation": True,
            "noiseSuppression": True,
            "autoGainControl": True,
            "sampleRate": {"ideal": 16000},
            "sampleSize": {"ideal": 16},
            "channelCount": {"exact": 1},
            "latency": {"ideal": 0.01},
        }
    
    def launch_stream(
        self,
        server_name: str = "0.0.0.0",
        server_port: int = 7860,
        share: bool = False,
        quiet: bool = False
    ) -> None:
        """
        Launch the FastRTC stream web interface.
        
        Args:
            server_name: Server hostname
            server_port: Server port
            share: Whether to create a public share link
            quiet: Whether to suppress launch messages
        """
        if not self.stream:
            raise RuntimeError("Stream not created. Call create_stream() first.")
        
        logger.info(f"🚀 Launching FastRTC stream on {server_name}:{server_port}")
        
        if not quiet:
            self._print_launch_info()
        
        try:
            self.is_running = True
            self.stream.ui.launch(
                server_name=server_name,
                server_port=server_port,
                quiet=quiet,
                share=share
            )
        except Exception as e:
            logger.error(f"❌ Failed to launch stream: {e}")
            self.is_running = False
            raise
    
    def _print_launch_info(self) -> None:
        """Print launch information and usage instructions."""
        print("=" * 70)
        print("🚀 FastRTC Voice Assistant Ready!")
        print("=" * 70)
        print("💡 Test Commands:")
        print("   • 'My name is [Your Name]'")
        print("   • 'What is my name?' / 'Who am I?'")
        print("   • 'I like [something interesting]'")
        print("   • Ask questions in supported languages.")
        print("\n🛑 To stop: Press Ctrl+C in the terminal")
        print("=" * 70)
    
    def stop_stream(self) -> None:
        """Stop the FastRTC stream."""
        if self.stream and self.is_running:
            logger.info("🛑 Stopping FastRTC stream...")
            self.is_running = False
            # Note: FastRTC doesn't have a direct stop method,
            # stopping is typically handled by the UI framework
    
    def get_stream_status(self) -> Dict[str, Any]:
        """
        Get current stream status information.
        
        Returns:
            Dictionary containing stream status
        """
        return {
            "is_running": self.is_running,
            "stream_created": self.stream is not None,
            "stream_type": "audio" if self.stream else None
        }