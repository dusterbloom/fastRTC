"""GPU-enabled Faster-whisper STT engine implementation with threading support."""

import os
import time
import asyncio
import threading
import queue
from pathlib import Path
from typing import Optional

import numpy as np
from faster_whisper import WhisperModel, BatchedInferencePipeline

from .base import BaseSTTEngine
from ....core.interfaces import TranscriptionResult, AudioData
from ....utils.logging import get_logger

logger = get_logger(__name__)

_MODEL_DIR = Path("/models/whisper-v3-ct2")   # baked into the image
_COMPUTE   = "int8_float16"                   # best perf/quality on Ampere


class FasterWhisperGPUSTT(BaseSTTEngine):
    """
    GPU-enabled STT using faster-whisper + CTranslate2 with threading.
    
    Uses threading instead of multiprocessing to support CUDA contexts.
    Similar to RealtimeSTT's approach for GPU acceleration.
    """

    def __init__(self):
        super().__init__()
        
        # Configuration
        self.model_path = os.environ.get("FASTER_WHISPER_MODEL_PATH", "Systran/faster-whisper-large-v3")
        self.device = None
        self.compute_type = None
        self.beam_size = 1
        self.batch_size = 0
        self.temperature = 0.0
        self.vad_filter = True  # Enable VAD with proper settings
        
        # Threading components
        self.model = None
        self.transcription_queue = queue.Queue()
        self.result_queue = queue.Queue()
        self.worker_thread = None
        self.shutdown_event = threading.Event()
        
        try:
            self._check_and_configure_device()
            self._initialize_model()
            self._start_worker_thread()
            self._set_available(True)
            logger.info("✅ FasterWhisperGPUSTT initialized with GPU support")
        except Exception as e:
            logger.error(f"Failed to initialize FasterWhisperGPUSTT: {e}")
            self._set_available(False)
            raise
    
    def _check_and_configure_device(self):
        """Check GPU availability and configure device."""
        # Check if CPU mode is forced via environment variable
        force_cpu = os.environ.get("FASTER_WHISPER_FORCE_CPU", "false").lower() == "true"
        if force_cpu:
            logger.info("🔧 CPU mode forced via FASTER_WHISPER_FORCE_CPU environment variable")
            self.device = "cpu"
            self.compute_type = "int8"
            return
        
        try:
            import torch
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                current_device = torch.cuda.current_device()
                gpu_name = torch.cuda.get_device_name(current_device)
                logger.info(f"🚀 GPU available: {gpu_name} (device {current_device}/{gpu_count})")
                self.device = "cuda"
                self.compute_type = _COMPUTE
                logger.info("✅ Using CUDA device for transcription")
            else:
                logger.info("⚠️ CUDA not available, falling back to CPU")
                self.device = "cpu"
                self.compute_type = "int8"
        except ImportError:
            logger.info("⚠️ PyTorch not available, falling back to CPU")
            self.device = "cpu"
            self.compute_type = "int8"
        except Exception as e:
            logger.warning(f"⚠️ Error checking GPU availability: {e}, falling back to CPU")
            self.device = "cpu"
            self.compute_type = "int8"
    
    def _initialize_model(self):
        """Initialize the Whisper model."""
        logger.info(f"🔧 Initializing Whisper model on {self.device} with {self.compute_type}...")
        
        try:
            self.model = WhisperModel(
                model_size_or_path=self.model_path,
                device=self.device,
                compute_type=self.compute_type
            )
            
            # Create batched pipeline if batch_size > 0
            if self.batch_size > 0:
                self.model = BatchedInferencePipeline(model=self.model)
            
            # Warm up model with dummy audio
            dummy_audio = np.zeros(16000, dtype=np.float32)  # 1 second of silence
            segments, info = self.model.transcribe(
                dummy_audio,
                language="en",
                beam_size=1,
                vad_filter=False
            )
            _ = " ".join(segment.text for segment in segments)
            
            logger.info("✅ Whisper model initialized and warmed up successfully")
            
        except Exception as e:
            logger.error(f"❌ Error initializing Whisper model: {e}")
            raise
    
    def _start_worker_thread(self):
        """Start the transcription worker thread."""
        logger.info("🔧 Starting transcription worker thread...")
        
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()
        
        logger.info("✅ Transcription worker thread started")
    
    def _worker_loop(self):
        """Main worker thread loop for processing transcription requests."""
        logger.info("🔄 Worker thread started and waiting for requests")
        
        while not self.shutdown_event.is_set():
            try:
                # Get transcription request with timeout
                try:
                    logger.debug("🔍 Worker thread waiting for transcription request...")
                    request_id, audio_data, sample_rate, language = self.transcription_queue.get(timeout=0.1)
                    logger.info(f"🎤 Worker thread received transcription request: {request_id}")
                except queue.Empty:
                    continue
                
                try:
                    logger.info(f"🎯 Processing transcription request {request_id}")
                    logger.info(f"🔍 Audio analysis: shape={audio_data.shape}, dtype={audio_data.dtype}, "
                               f"min={audio_data.min():.6f}, max={audio_data.max():.6f}, "
                               f"mean={audio_data.mean():.6f}, rms={np.sqrt(np.mean(audio_data**2)):.6f}")
                    start_time = time.time()
                    
                    # Check for silent audio
                    rms = np.sqrt(np.mean(audio_data**2))
                    if rms < 1e-6:
                        logger.warning(f"⚠️ Audio appears to be silent (RMS: {rms:.8f}) for request {request_id}")
                    
                    # Ensure audio is float32
                    if audio_data.dtype != np.float32:
                        audio_data = audio_data.astype(np.float32)
                        logger.debug(f"🔧 Converted audio to float32 for request {request_id}")
                    
                    # Flatten audio if it's 2D (faster-whisper expects 1D)
                    if audio_data.ndim > 1:
                        original_shape = audio_data.shape
                        audio_data = audio_data.flatten()
                        logger.debug(f"🔧 Flattened audio from {original_shape} to {audio_data.shape} for request {request_id}")
                    
                    # Normalize audio if needed
                    max_val = np.max(np.abs(audio_data))
                    if max_val > 1.0:
                        audio_data = audio_data / max_val
                        logger.debug(f"🔧 Normalized audio by factor {max_val} for request {request_id}")
                    
                    # Transcribe audio
                    transcribe_kwargs = {
                        'language': language,
                        'beam_size': self.beam_size,
                        'temperature': self.temperature,
                        'vad_filter': self.vad_filter
                    }
                    
                    if self.batch_size > 0:
                        transcribe_kwargs['batch_size'] = self.batch_size
                    
                    logger.info(f"🚀 Starting transcription for request {request_id} with kwargs: {transcribe_kwargs}")
                    segments, info = self.model.transcribe(audio_data, **transcribe_kwargs)
                    
                    # Collect transcription text
                    text_parts = []
                    segment_count = 0
                    for segment in segments:
                        segment_count += 1
                        segment_text = segment.text.strip()
                        text_parts.append(segment_text)
                        logger.debug(f"📝 Segment {segment_count} for {request_id}: '{segment_text}'")
                    
                    full_text = " ".join(text_parts).strip()
                    
                    # Get language info
                    detected_language = getattr(info, 'language', language or 'en')
                    confidence = getattr(info, 'language_probability', 1.0)
                    
                    elapsed = time.time() - start_time
                    logger.info(f"✅ Transcription {request_id} completed: '{full_text}' (segments: {segment_count}, "
                               f"lang: {detected_language}, conf: {confidence:.3f}) in {elapsed:.3f}s")
                    
                    # Send result back
                    result = TranscriptionResult(
                        text=full_text,
                        language=detected_language,
                        confidence=confidence
                    )
                    self.result_queue.put((request_id, 'success', result))
                    
                except Exception as e:
                    logger.error(f"❌ Transcription error for request {request_id}: {e}")
                    self.result_queue.put((request_id, 'error', str(e)))
                
                finally:
                    self.transcription_queue.task_done()
                    
            except Exception as e:
                logger.error(f"❌ Error in worker thread: {e}")
        
        logger.debug("🔄 Worker thread stopped")

    async def _transcribe_audio(self, audio, target_language: str = None) -> TranscriptionResult:
        """Implement specific transcription logic using threading worker.
        
        Args:
            audio: Audio data to transcribe (AudioData object or numpy array)
            target_language: Target language code (Whisper format, e.g. 'en', 'it', 'es')
            
        Returns:
            TranscriptionResult: Transcription result
        """
        logger.debug(f"🚀 [FASTER_WHISPER_GPU] _transcribe_audio called with audio type: {type(audio)}")
        
        # Extract audio samples
        if isinstance(audio, AudioData):
            audio_samples = audio.samples
            sample_rate = audio.sample_rate
            logger.debug(f"🔍 [FASTER_WHISPER_GPU] AudioData object: samples shape={audio_samples.shape}, sr={sample_rate}")
        else:
            # Assume it's a numpy array
            audio_samples = audio
            sample_rate = 16000  # Default sample rate
            logger.debug(f"🔍 [FASTER_WHISPER_GPU] Raw numpy array: shape={audio_samples.shape}, sr={sample_rate}")
        
        # Ensure float32 format
        if audio_samples.dtype != np.float32:
            audio_samples = audio_samples.astype(np.float32)
        
        # Flatten audio if it's 2D (faster-whisper expects 1D)
        if audio_samples.ndim > 1:
            original_shape = audio_samples.shape
            audio_samples = audio_samples.flatten()
            logger.debug(f"🔧 Flattened audio from {original_shape} to {audio_samples.shape}")
        
        logger.info(f"🔧 FasterWhisperGPU transcribing: shape={audio_samples.shape}, lang={target_language}")
        logger.info(f"🔍 Audio analysis: min={audio_samples.min():.6f}, max={audio_samples.max():.6f}, "
                   f"mean={audio_samples.mean():.6f}, rms={np.sqrt(np.mean(audio_samples**2)):.6f}")
        
        try:
            # Generate unique request ID
            request_id = f"req_{int(time.time() * 1000000)}"
            logger.info(f"🚀 Sending transcription request {request_id} to worker thread")
            
            # Send transcription request to worker
            request = (request_id, audio_samples, sample_rate, target_language)
            self.transcription_queue.put(request)
            logger.info(f"✅ Request {request_id} queued successfully")
            
            # Wait for response with timeout
            timeout = 30.0  # 30 second timeout
            start_time = time.time()
            
            while time.time() - start_time < timeout:
                try:
                    result_id, status, data = self.result_queue.get(timeout=0.1)
                    
                    if result_id == request_id:
                        if status == 'success':
                            logger.debug(f"✅ Transcription successful: '{data.text}'")
                            return data
                        elif status == 'error':
                            logger.error(f"❌ Transcription error: {data}")
                            return TranscriptionResult(
                                text="",
                                language=target_language or "en",
                                confidence=0.0
                            )
                    else:
                        # Put back result for different request
                        self.result_queue.put((result_id, status, data))
                        
                except queue.Empty:
                    await asyncio.sleep(0.1)
                    continue
            
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
        """Synchronous transcription method using threading worker.
        
        Args:
            audio_samples: Audio samples as numpy array
            target_language: Target language code (Whisper format, e.g. 'en', 'it', 'es')
            
        Returns:
            TranscriptionResult: Transcription result
        """
        logger.debug(f"🔧 FasterWhisperGPU sync transcription: shape={audio_samples.shape}, lang={target_language}")
        
        try:
            # Generate unique request ID
            request_id = f"sync_req_{int(time.time() * 1000000)}"
            
            # Send transcription request to worker
            sample_rate = 16000  # Default sample rate
            request = (request_id, audio_samples, sample_rate, target_language)
            self.transcription_queue.put(request)
            
            # Wait for response with timeout
            timeout = 30.0  # 30 second timeout
            start_time = time.time()
            
            while time.time() - start_time < timeout:
                try:
                    result_id, status, data = self.result_queue.get(timeout=0.1)
                    
                    if result_id == request_id:
                        if status == 'success':
                            logger.debug(f"✅ Sync transcription successful: '{data.text}'")
                            return data
                        elif status == 'error':
                            logger.error(f"❌ Sync transcription error: {data}")
                            return TranscriptionResult(
                                text="",
                                language=target_language or "en",
                                confidence=0.0
                            )
                    else:
                        # Put back result for different request
                        self.result_queue.put((result_id, status, data))
                        
                except queue.Empty:
                    time.sleep(0.1)
                    continue
            
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
        # Convert bytes to numpy array for worker thread
        audio_array = np.frombuffer(pcm16_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        
        # Use sync transcription method
        result = self._transcribe_sync(audio_array)
        if result.text:
            yield result.text
    
    def shutdown(self):
        """Shutdown the transcription worker thread and cleanup resources."""
        logger.info("🧹 Shutting down FasterWhisperGPUSTT...")
        
        self.shutdown_event.set()
        
        # Wait for worker thread to finish
        if self.worker_thread and self.worker_thread.is_alive():
            logger.info("🔄 Waiting for transcription worker thread to finish...")
            self.worker_thread.join(timeout=5)
            
            if self.worker_thread.is_alive():
                logger.warning("⚠️ Worker thread did not finish gracefully")
        
        # Clear queues
        while not self.transcription_queue.empty():
            try:
                self.transcription_queue.get_nowait()
                self.transcription_queue.task_done()
            except queue.Empty:
                break
        
        while not self.result_queue.empty():
            try:
                self.result_queue.get_nowait()
            except queue.Empty:
                break
        
        logger.info("✅ FasterWhisperGPUSTT shutdown complete")
    
    def __del__(self):
        """Ensure cleanup on deletion."""
        try:
            self.shutdown()
        except Exception as e:
            logger.warning(f"Error during FasterWhisperGPUSTT cleanup: {e}")