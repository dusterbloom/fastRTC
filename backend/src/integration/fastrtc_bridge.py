"""
FastRTC Bridge Module

Handles FastRTC stream setup, configuration, and WebRTC connection lifecycle.
Extracted from the original voice assistant implementation.
"""

import logging
from typing import Optional, Dict, Any, Callable
from fastrtc import Stream, ReplyOnPause, AlgoOptions, SileroVadOptions

from ..utils.logging import get_logger

logger = get_logger(__name__)


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
        
        try:
            # Create stream with ReplyOnPause for voice activity detection
            self.stream = Stream(
                ReplyOnPause(
                    callback_function,
                    can_interrupt=True,
                       algo_options=AlgoOptions(
                        # This is the GATEKEEPER. We are making it extremely sensitive.
                        speech_threshold=0.5,  # Drastically lower: will detect even quiet speech.
                        started_talking_threshold=0.10,
                        audio_chunk_duration=1.5 # Process audio in smaller chunks for responsiveness
                    ),
                    model_options=SileroVadOptions(
                        # This is the TIMER. Updated for faster interruption support.
                        threshold=0.2,                  # More sensitive model threshold
                        min_speech_duration_ms=350,     # Catches very short words like "a" or "I"
                        min_silence_duration_ms=3000,   # FASTER: Reduced from 4000ms to 1200ms for quicker interruption
                        speech_pad_ms=250              # Generous buffer at the end of your speech
                    )
                ),
                modality="audio",
                mode="send-receive",
                track_constraints=self._get_audio_constraints()
            )
            

            
            logger.info("✅ FastRTC stream created successfully")
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