"""Faster-whisper STT engine implementation with multiprocessing support."""

import os
import time
import asyncio
import threading
import multiprocessing as mp
from pathlib import Path
from typing import Optional

import numpy as np

from .base import BaseSTTEngine
from .transcription_worker import transcription_worker_main
from ....core.interfaces import TranscriptionResult, AudioData
from ....utils.logging import get_logger
from ....utils.safepipe import SafePipeFactory

logger = get_logger(__name__)

_MODEL_DIR = Path("/models/whisper-v3-ct2")   # baked into the image
_COMPUTE   = "int8_float16"                   # best perf/quality on Ampere


class FasterWhisperSTT(BaseSTTEngine):
    """
    Multiprocessing-based STT using faster-whisper + CTranslate2.
    
    Uses a separate process for transcription to avoid threading issues
    with faster-whisper and CUDA contexts.
    """

    def __init__(self):
        super().__init__()
        
        # Configuration
        self.model_path = os.environ.get("FASTER_WHISPER_MODEL_PATH", "Systran/faster-whisper-large-v3")
        # Device will be set in _initialize_worker_process based on availability
        self.device = None
        self.compute_type = None
        self.beam_size = 1
        self.batch_size = 0
        self.temperature = 0.0
        self.vad_filter = True
        
        # Multiprocessing components
        self.transcription_process = None
        self.parent_conn = None
        self.child_conn = None
        self.parent_stdout_pipe = None
        self.child_stdout_pipe = None
        self.ready_event = mp.Event()
        self.shutdown_event = mp.Event()
        
        # Initialize multiprocessing
        try:
            import torch
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                current_device = torch.cuda.current_device()
                gpu_name = torch.cuda.get_device_name(current_device)
                logger.info(f"🚀 GPU available: {gpu_name} (device {current_device}/{gpu_count})")
                # For multiprocessing with CUDA, we need to be more careful
                # Let's use CPU for multiprocessing to avoid CUDA context sharing issues
                logger.info("⚠️ Using CPU for multiprocessing to avoid CUDA context sharing issues")
                return False, "cpu", "int8"
            else:
                logger.info("⚠️ CUDA not available, falling back to CPU")
                return False, "cpu", "int8"
        except ImportError:
            logger.info("⚠️ PyTorch not available, falling back to CPU")
            return False, "cpu", "int8"
        except Exception as e:
            logger.warning(f"⚠️ Error checking GPU availability: {e}, falling back to CPU")
            return False, "cpu", "int8"
        
        try:
            import torch
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                current_device = torch.cuda.current_device()
                gpu_name = torch.cuda.get_device_name(current_device)
                logger.info(f"🚀 GPU available: {gpu_name} (device {current_device}/{gpu_count})")
                
                # Try GPU first - multiprocessing with spawn should handle CUDA contexts properly
                logger.info("🚀 Using GPU for multiprocessing transcription worker")
                return True, "cuda", "int8_float16"
            else:
                logger.info("⚠️ CUDA not available, falling back to CPU")
                return False, "cpu", "int8"
        except ImportError:
            logger.info("⚠️ PyTorch not available, falling back to CPU")
            return False, "cpu", "int8"
        except Exception as e:
            logger.warning(f"⚠️ Error checking GPU availability: {e}, falling back to CPU")
            return False, "cpu", "int8"
        
        try:
            import torch
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                current_device = torch.cuda.current_device()
                gpu_name = torch.cuda.get_device_name(current_device)
                logger.info(f"🚀 GPU available: {gpu_name} (device {current_device}/{gpu_count})")
                # For multiprocessing with CUDA, we need to be more careful
                # Let's use CPU for multiprocessing to avoid CUDA context issues
                logger.info("⚠️ Using CPU for multiprocessing to avoid CUDA context sharing issues")
                return False, "cpu", "int8"
            else:
                logger.info("⚠️ CUDA not available, falling back to CPU")
                return False, "cpu", "int8"
        except ImportError:
            logger.info("⚠️ PyTorch not available, falling back to CPU")
            return False, "cpu", "int8"
        except Exception as e:
            logger.warning(f"⚠️ Error checking GPU availability: {e}, falling back to CPU")
            return False, "cpu", "int8"

    def _initialize_worker_process(self):
        """Initialize the transcription worker process."""
        logger.info("🔧 Initializing transcription worker process...")
        
        # Create communication pipes using raw multiprocessing.Pipe
        self.parent_conn, self.child_conn = mp.Pipe()
        self.parent_stdout_pipe, self.child_stdout_pipe = mp.Pipe()
        
        # Check GPU availability and configure device
        gpu_available, device, compute_type = self._check_gpu_availability()
        self.device = device
        self.compute_type = compute_type
        
        # Start transcription worker process
        self.transcription_process = mp.Process(
            target=transcription_worker_main,
            args=(
                self.child_conn,
                self.child_stdout_pipe,
                self.model_path,
                self.device,
                self.compute_type,
                self.beam_size,
                self.batch_size,
                None,  # language (will be set per request)
                self.temperature,
                self.vad_filter
            )
        )
        self.transcription_process.start()
        
        # Start stdout reader thread
        self.stdout_thread = threading.Thread(target=self._read_stdout, daemon=True)
        self.stdout_thread.start()
        
        # Wait for worker to be ready
        logger.info("⏳ Waiting for transcription worker to initialize...")
        start_time = time.time()
        timeout = 60  # 60 second timeout for model loading
        
        while time.time() - start_time < timeout:
            if self.parent_conn.poll(0.1):
                message = self.parent_conn.recv()
                if message and len(message) == 2:
                    status, data = message
                    if status == 'ready':
                        logger.info("✅ Transcription worker ready")
                        return
                    elif status == 'error':
                        raise RuntimeError(f"Worker initialization failed: {data}")
            time.sleep(0.1)
        
        raise TimeoutError("Transcription worker failed to initialize within timeout")
    
    def _read_stdout(self):
        """Read stdout messages from worker process."""
        while not self.shutdown_event.is_set():
            try:
                if self.parent_stdout_pipe.poll(0.1):
                    message = self.parent_stdout_pipe.recv()
                    if message:
                        logger.debug(f"Worker: {message}")
            except (BrokenPipeError, EOFError, OSError):
                break
            except Exception as e:
                logger.warning(f"Error reading worker stdout: {e}")
                break
            time.sleep(0.1)

    async def _transcribe_audio(self, audio, target_language: str = None) -> TranscriptionResult:
        """Implement specific transcription logic using multiprocessing worker.
        
        Args:
            audio: Audio data to transcribe (AudioData object or numpy array)
            target_language: Target language code (Whisper format, e.g. 'en', 'it', 'es')
            
        Returns:
            TranscriptionResult: Transcription result
        """
        # Extract audio samples
        if isinstance(audio, AudioData):
            audio_samples = audio.samples
            sample_rate = audio.sample_rate
        else:
            # Assume it's a numpy array
            audio_samples = audio
            sample_rate = 16000  # Default sample rate
        
        # Ensure float32 format
        if audio_samples.dtype != np.float32:
            audio_samples = audio_samples.astype(np.float32)
        
        logger.debug(f"🔧 FasterWhisper transcribing via worker process: shape={audio_samples.shape}, lang={target_language}")
        
        try:
            # Check if worker process is alive
            if not self.transcription_process or not self.transcription_process.is_alive():
                logger.error("❌ Transcription worker process is not running")
                return TranscriptionResult(
                    text="",
                    language=target_language or "en",
                    confidence=0.0
                )
            
            # Send transcription request to worker
            request = (audio_samples, sample_rate, target_language)
            try:
                self.parent_conn.send(request)
            except Exception as e:
                logger.error(f"❌ Failed to send transcription request to worker: {e}")
                return TranscriptionResult(
                    text="",
                    language=target_language or "en",
                    confidence=0.0
                )
            
            # Wait for response with timeout
            timeout = 30.0  # 30 second timeout
            start_time = time.time()
            
            while time.time() - start_time < timeout:
                if self.parent_conn.poll(0.1):
                    response = self.parent_conn.recv()
                    if response and len(response) == 2:
                        status, data = response
                        
                        if status == 'success':
                            logger.debug(f"✅ Transcription successful: '{data['text']}'")
                            return TranscriptionResult(
                                text=data['text'],
                                language=data['language'],
                                confidence=data['confidence']
                            )
                        elif status == 'error':
                            logger.error(f"❌ Transcription error from worker: {data}")
                            return TranscriptionResult(
                                text="",
                                language=target_language or "en",
                                confidence=0.0
                            )
                
                # Check if process is still alive
                if not self.transcription_process.is_alive():
                    logger.error("❌ Transcription worker process died during transcription")
                    break
                
                await asyncio.sleep(0.1)
            
            logger.error("❌ Transcription timeout")
            return TranscriptionResult(
                text="",
                language=target_language or "en",
                confidence=0.0
            )
            
        except Exception as e:
            logger.error(f"❌ Transcription error: {e}")
            return TranscriptionResult(
                text="",
                language=target_language or "en",
                confidence=0.0
            )

    def _transcribe_sync(self, audio_samples: np.ndarray, target_language: str = None) -> TranscriptionResult:
        """Synchronous transcription method using multiprocessing worker.
        
        Args:
            audio_samples: Audio samples as numpy array
            target_language: Target language code (Whisper format, e.g. 'en', 'it', 'es')
            
        Returns:
            TranscriptionResult: Transcription result
        """
        logger.debug(f"🔧 FasterWhisper sync transcription: shape={audio_samples.shape}, lang={target_language}")
        
        try:
            # Check if worker process is alive
            if not self.transcription_process or not self.transcription_process.is_alive():
                logger.error("❌ Transcription worker process is not running")
                return TranscriptionResult(
                    text="",
                    language=target_language or "en",
                    confidence=0.0
                )
            
            # Send transcription request to worker
            sample_rate = 16000  # Default sample rate
            request = (audio_samples, sample_rate, target_language)
            try:
                self.parent_conn.send(request)
            except Exception as e:
                logger.error(f"❌ Failed to send transcription request to worker: {e}")
                return TranscriptionResult(
                    text="",
                    language=target_language or "en",
                    confidence=0.0
                )
            
            # Wait for response with timeout
            timeout = 30.0  # 30 second timeout
            start_time = time.time()
            
            while time.time() - start_time < timeout:
                if self.parent_conn.poll(0.1):
                    response = self.parent_conn.recv()
                    if response and len(response) == 2:
                        status, data = response
                        
                        if status == 'success':
                            logger.debug(f"✅ Sync transcription successful: '{data['text']}'")
                            return TranscriptionResult(
                                text=data['text'],
                                language=data['language'],
                                confidence=data['confidence']
                            )
                        elif status == 'error':
                            logger.error(f"❌ Sync transcription error from worker: {data}")
                            return TranscriptionResult(
                                text="",
                                language=target_language or "en",
                                confidence=0.0
                            )
                
                # Check if process is still alive
                if not self.transcription_process.is_alive():
                    logger.error("❌ Transcription worker process died during sync transcription")
                    break
                
                time.sleep(0.1)
            
            logger.error("❌ Sync transcription timeout")
            return TranscriptionResult(
                text="",
                language=target_language or "en",
                confidence=0.0
            )
            
        except Exception as e:
            logger.error(f"❌ Sync transcription error: {e}")
            return TranscriptionResult(
                text="",
                language=target_language or "en",
                confidence=0.0
            )

    def stream(self, pcm16_bytes):
        """
        Generator yielding partial transcripts (~1 s latency).
        For compatibility with streaming interfaces.
        """
        # Convert bytes to numpy array for worker process
        audio_array = np.frombuffer(pcm16_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        
        # Use sync transcription method
        result = self._transcribe_sync(audio_array)
        if result.text:
            yield result.text
    
    def shutdown(self):
        """Shutdown the transcription worker process and cleanup resources."""
        logger.info("🧹 Shutting down FasterWhisperSTT...")
        
        self.shutdown_event.set()
        
        # Close connections
        if self.parent_conn:
            self.parent_conn.close()
        if self.parent_stdout_pipe:
            self.parent_stdout_pipe.close()
        
        # Terminate worker process
        if self.transcription_process and self.transcription_process.is_alive():
            logger.info("🔄 Terminating transcription worker process...")
            self.transcription_process.terminate()
            self.transcription_process.join(timeout=5)
            
            if self.transcription_process.is_alive():
                logger.warning("⚠️ Force killing transcription worker process...")
                self.transcription_process.kill()
                self.transcription_process.join()
        
        # Wait for stdout thread
        if hasattr(self, 'stdout_thread') and self.stdout_thread.is_alive():
            self.stdout_thread.join(timeout=2)
        
        logger.info("✅ FasterWhisperSTT shutdown complete")
    
    def __del__(self):
        """Ensure cleanup on deletion."""
        try:
            self.shutdown()
        except Exception as e:
            logger.warning(f"Error during FasterWhisperSTT cleanup: {e}")
