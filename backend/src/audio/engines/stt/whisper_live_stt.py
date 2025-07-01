"""
WhisperLive STT Engine Implementation

Real-time streaming STT using WhisperLive client-server architecture
with faster-whisper backend for improved threading performance.
"""

import asyncio
import io
import time
import threading
import numpy as np
import wave
from typing import Optional, Dict, Any, Union
from pathlib import Path

from .base import BaseSTTEngine
from ....core.interfaces import AudioData, TranscriptionResult
from ....core.exceptions import STTError
from ....utils.logging import get_logger

try:
    from whisper_live.client import TranscriptionClient
    WHISPER_LIVE_AVAILABLE = True
except ImportError:
    WHISPER_LIVE_AVAILABLE = False
    # Don't log here - will log when engine is actually used

logger = get_logger(__name__)


class WhisperLiveSTTEngine(BaseSTTEngine):
    """
    WhisperLive STT engine using client-server architecture.
    
    Provides real-time streaming transcription with faster-whisper backend
    for improved performance in threading pipelines.
    """
    
    def __init__(
        self,
        server_host: str = "localhost",
        server_port: int = 9090,
        model: str = "small",
        language: str = "en",
        translate: bool = False,
        use_vad: bool = True,
        max_clients: int = 4,
        max_connection_time: int = 600,
        auto_start_server: bool = True,
        server_backend: str = "faster_whisper",
        server_model_path: Optional[str] = None
    ):
        """
        Initialize WhisperLive STT engine.
        
        Args:
            server_host: WhisperLive server host
            server_port: WhisperLive server port
            model: Whisper model size or HF model path
            language: Target language for transcription
            translate: Whether to translate to English
            use_vad: Whether to use Voice Activity Detection
            max_clients: Maximum number of concurrent clients
            max_connection_time: Maximum connection time in seconds
            auto_start_server: Whether to auto-start server if not running
            server_backend: Backend for server (faster_whisper, tensorrt, openvino)
            server_model_path: Custom model path for server
        """
        super().__init__()
        
        if not WHISPER_LIVE_AVAILABLE:
            raise STTError("whisper-live is not available. Install with: pip install whisper-live")
        
        self.server_host = server_host
        self.server_port = server_port
        self.model = model
        self.language = language
        self.translate = translate
        self.use_vad = use_vad
        self.max_clients = max_clients
        self.max_connection_time = max_connection_time
        self.auto_start_server = auto_start_server
        self.server_backend = server_backend
        self.server_model_path = server_model_path
        
        # Client management
        self.client: Optional[TranscriptionClient] = None
        self.server_process: Optional[Any] = None
        self.client_lock = threading.Lock()
        
        # Transcription state
        self.current_transcription = ""
        self.transcription_complete = threading.Event()
        self.transcription_lock = threading.Lock()
        
        # Performance tracking
        self.connection_retries = 0
        self.max_retries = 3
        
        logger.info(f"🎤 WhisperLive STT engine initialized (server: {server_host}:{server_port})")
    
    async def initialize(self) -> bool:
        """Initialize the WhisperLive STT engine."""
        try:
            # Check if server is running
            if not await self._check_server_health():
                if self.auto_start_server:
                    logger.info("🚀 Starting WhisperLive server...")
                    await self._start_server()
                else:
                    raise STTError(f"WhisperLive server not available at {self.server_host}:{self.server_port}")
            
            # Initialize client
            await self._initialize_client()
            
            self._is_available = True
            logger.info("✅ WhisperLive STT engine initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize WhisperLive STT engine: {e}")
            self._is_available = False
            return False
    
    async def _check_server_health(self) -> bool:
        """Check if WhisperLive server is running and healthy."""
        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                # Try to connect to server (basic connectivity check)
                async with session.get(
                    f"http://{self.server_host}:{self.server_port}/health",
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as response:
                    return response.status == 200
        except Exception:
            # Server not running or not accessible
            return False
    
    async def _start_server(self):
        """Start WhisperLive server if not running."""
        try:
            import subprocess
            import sys
            
            # Build server command
            cmd = [
                sys.executable, "-m", "whisper_live.server",
                "--port", str(self.server_port),
                "--backend", self.server_backend
            ]
            
            if self.server_model_path:
                if self.server_backend == "faster_whisper":
                    cmd.extend(["-fw", self.server_model_path])
                elif self.server_backend == "tensorrt":
                    cmd.extend(["-trt", self.server_model_path])
            
            # Start server process
            self.server_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Wait for server to start
            await asyncio.sleep(3)
            
            # Verify server is running
            if not await self._check_server_health():
                raise STTError("Failed to start WhisperLive server")
            
            logger.info(f"✅ WhisperLive server started on port {self.server_port}")
            
        except Exception as e:
            raise STTError(f"Failed to start WhisperLive server: {e}")
    
    async def _initialize_client(self):
        """Initialize WhisperLive client."""
        try:
            with self.client_lock:
                self.client = TranscriptionClient(
                    host=self.server_host,
                    port=self.server_port,
                    lang=self.language,
                    translate=self.translate,
                    model=self.model,
                    use_vad=self.use_vad,
                    max_clients=self.max_clients,
                    max_connection_time=self.max_connection_time,
                    save_output_recording=False  # Don't save recordings
                )
            
            logger.info("✅ WhisperLive client initialized")
            
        except Exception as e:
            raise STTError(f"Failed to initialize WhisperLive client: {e}")
    
    async def _transcribe_audio(self, audio: Union[AudioData, np.ndarray]) -> TranscriptionResult:
        """
        Transcribe audio using WhisperLive client.
        
        Args:
            audio: Audio data to transcribe
            
        Returns:
            TranscriptionResult: Transcription with metadata
        """
        if not self.client:
            raise STTError("WhisperLive client not initialized")
        
        try:
            # Convert audio to the format expected by WhisperLive
            audio_data = self._prepare_audio(audio)
            
            # Create temporary WAV file for WhisperLive client
            wav_path = await self._create_temp_wav(audio_data)
            
            try:
                # Reset transcription state
                with self.transcription_lock:
                    self.current_transcription = ""
                    self.transcription_complete.clear()
                
                # Set up transcription callback
                original_callback = getattr(self.client, '_on_message', None)
                self.client._on_message = self._transcription_callback
                
                # Run transcription in thread to avoid blocking
                transcription_task = asyncio.create_task(
                    self._run_transcription(wav_path)
                )
                
                # Wait for transcription with timeout
                try:
                    await asyncio.wait_for(
                        asyncio.to_thread(self.transcription_complete.wait),
                        timeout=30.0  # 30 second timeout
                    )
                except asyncio.TimeoutError:
                    transcription_task.cancel()
                    raise STTError("Transcription timeout")
                
                # Restore original callback
                if original_callback:
                    self.client._on_message = original_callback
                
                # Get final transcription
                with self.transcription_lock:
                    text = self.current_transcription.strip()
                
                return TranscriptionResult(
                    text=text,
                    confidence=0.9,  # WhisperLive doesn't provide confidence scores
                    language=self.language,
                    processing_time=0.0,  # Will be calculated by base class
                    metadata={
                        'engine': 'whisper-live',
                        'model': self.model,
                        'backend': self.server_backend,
                        'server': f"{self.server_host}:{self.server_port}"
                    }
                )
                
            finally:
                # Clean up temporary file
                try:
                    Path(wav_path).unlink(missing_ok=True)
                except Exception:
                    pass
                    
        except Exception as e:
            logger.error(f"WhisperLive transcription error: {e}")
            raise STTError(f"Transcription failed: {e}")
    
    def _prepare_audio(self, audio: Union[AudioData, np.ndarray]) -> np.ndarray:
        """Prepare audio data for WhisperLive processing."""
        if isinstance(audio, AudioData):
            audio_data = audio.data
            sample_rate = audio.sample_rate
        else:
            audio_data = audio
            sample_rate = 16000  # Default sample rate
        
        # Ensure audio is numpy array
        if not isinstance(audio_data, np.ndarray):
            audio_data = np.array(audio_data)
        
        # Ensure mono audio
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)
        
        # Normalize to float32 range [-1, 1]
        if audio_data.dtype == np.int16:
            audio_data = audio_data.astype(np.float32) / 32768.0
        elif audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)
        
        return audio_data
    
    async def _create_temp_wav(self, audio_data: np.ndarray, sample_rate: int = 16000) -> str:
        """Create temporary WAV file for WhisperLive client."""
        import tempfile
        
        # Create temporary file
        temp_fd, temp_path = tempfile.mkstemp(suffix='.wav')
        
        try:
            # Convert float32 to int16 for WAV file
            audio_int16 = (audio_data * 32767).astype(np.int16)
            
            # Write WAV file
            with wave.open(temp_path, 'wb') as wav_file:
                wav_file.setnchannels(1)  # Mono
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(audio_int16.tobytes())
            
            return temp_path
            
        finally:
            # Close file descriptor
            import os
            os.close(temp_fd)
    
    async def _run_transcription(self, wav_path: str):
        """Run transcription in a separate thread."""
        def transcribe():
            try:
                # Use WhisperLive client to transcribe file
                self.client(wav_path)
            except Exception as e:
                logger.error(f"Transcription thread error: {e}")
                self.transcription_complete.set()
        
        # Run in thread pool
        await asyncio.to_thread(transcribe)
    
    def _transcription_callback(self, message: Dict[str, Any]):
        """Handle transcription messages from WhisperLive client."""
        try:
            if isinstance(message, dict):
                # Extract text from message
                text = message.get('text', '')
                is_final = message.get('final', False)
                
                with self.transcription_lock:
                    if text:
                        self.current_transcription = text
                    
                    # Mark complete if final transcription received
                    if is_final:
                        self.transcription_complete.set()
            
        except Exception as e:
            logger.error(f"Transcription callback error: {e}")
            self.transcription_complete.set()
    
    async def cleanup(self):
        """Clean up WhisperLive resources."""
        try:
            # Close client connection
            if self.client:
                with self.client_lock:
                    # WhisperLive client doesn't have explicit close method
                    # Connection will be closed automatically
                    self.client = None
            
            # Stop server if we started it
            if self.server_process:
                self.server_process.terminate()
                try:
                    self.server_process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.server_process.kill()
                self.server_process = None
            
            self._is_available = False
            logger.info("🧹 WhisperLive STT engine cleaned up")
            
        except Exception as e:
            logger.error(f"Error during WhisperLive cleanup: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get engine statistics."""
        stats = super().get_stats()
        stats.update({
            'engine_type': 'whisper-live',
            'server_host': self.server_host,
            'server_port': self.server_port,
            'model': self.model,
            'backend': self.server_backend,
            'connection_retries': self.connection_retries
        })
        return stats
    
    def is_available(self) -> bool:
        """Check if the engine is available and ready."""
        return self._is_available and self.client is not None