"""HuggingFace STT engine implementation."""

import logging
import time
import asyncio
import io
import numpy as np
from typing import Optional, Dict, Any
import torch
from transformers import (
    AutoModelForSpeechSeq2Seq, 
    AutoProcessor, 
    pipeline
)

from .base import BaseSTTEngine
from ....core.interfaces import AudioData, TranscriptionResult
from ....core.exceptions import STTError
from ....config.settings import AudioConfig
from ....utils.logging import get_logger

logger = get_logger(__name__)
logger.setLevel(logging.INFO)


def get_device(force_cpu: bool = False) -> str:
    """Get the best available device for computation.
    
    Args:
        force_cpu: Force CPU usage even if GPU is available
        
    Returns:
        str: Device string ('cuda', 'mps', or 'cpu')
    """
    if force_cpu:
        return "cpu"
    
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def get_torch_and_np_dtypes(device: str, use_bfloat16: bool = False) -> tuple:
    """Get appropriate torch and numpy dtypes for the device.
    
    Args:
        device: Target device
        use_bfloat16: Whether to use bfloat16 precision
        
    Returns:
        tuple: (torch_dtype, np_dtype)
    """
    if device == "cuda":
        torch_dtype = torch.bfloat16 if use_bfloat16 else torch.float16
        np_dtype = np.float32  # Always use float32 for numpy
    else:
        torch_dtype = torch.float32
        np_dtype = np.float32
    
    return torch_dtype, np_dtype


def is_flash_attn_2_available() -> bool:
    """Check if Flash Attention 2 is available.
    
    Returns:
        bool: True if Flash Attention 2 is available
    """
    try:
        import flash_attn
        return True
    except ImportError:
        return False


def audio_to_bytes(audio_tuple: tuple) -> bytes:
    """Convert audio tuple to bytes for pipeline processing.
    
    Args:
        audio_tuple: (sample_rate, audio_array) tuple
        
    Returns:
        bytes: Audio data as bytes
    """
    sample_rate, audio_array = audio_tuple
    
    # Ensure audio is float32
    if audio_array.dtype != np.float32:
        audio_array = audio_array.astype(np.float32)
    
    # Convert to bytes using io.BytesIO
    buffer = io.BytesIO()
    # For pipeline, we need to return the raw audio array
    # The pipeline expects numpy array directly
    return audio_array


class HuggingFaceSTTEngine(BaseSTTEngine):
    """HuggingFace Transformers-based STT engine.
    
    This engine uses HuggingFace Transformers for speech-to-text transcription
    with automatic device detection and optimization.
    """
    
    def __init__(self, model_id: str = None, force_cpu: bool = False, use_bfloat16: bool = False):
        """Initialize the HuggingFace STT engine.
        
        Args:
            model_id: HuggingFace model ID to use
            force_cpu: Force CPU usage even if GPU is available
            use_bfloat16: Use bfloat16 precision for better performance
        """
        super().__init__()
        
        self.model_id = model_id or AudioConfig().hf_model_id
        self.force_cpu = force_cpu
        self.use_bfloat16 = use_bfloat16
        
        # Model components
        self.device = None
        self.torch_dtype = None
        self.np_dtype = None
        self.model = None
        self.processor = None
        self.pipeline = None
        
        # Initialize the model
        self._initialize_model()
    
    def _initialize_model(self) -> None:
        """Initialize the HuggingFace model and pipeline."""
        try:
            logger.info(f"🧠 Profiling: Starting STT model load (Hugging Face: {self.model_id})...")
            overall_start_time = time.monotonic()
            
            # Setup device and dtypes
            self.device = get_device(force_cpu=self.force_cpu)
            self.torch_dtype, self.np_dtype = get_torch_and_np_dtypes(
                self.device, use_bfloat16=self.use_bfloat16
            )
            
            logger.info(f"Using device: {self.device}, torch_dtype: {self.torch_dtype}, np_dtype: {self.np_dtype}")
            
            # Setup attention mechanism
            attention = "flash_attention_2" if is_flash_attn_2_available() else "sdpa"
            logger.info(f"Using attention implementation: {attention}")
            
            # Load model and processor
            logger.info(f"🧠 Profiling: Starting AutoModelForSpeechSeq2Seq.from_pretrained for {self.model_id}...")
            model_load_start_time = time.monotonic()
            self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
                self.model_id,
                torch_dtype=self.torch_dtype,
                low_cpu_mem_usage=True,
                use_safetensors=True,
                attn_implementation=attention
            )
            model_load_duration = time.monotonic() - model_load_start_time
            logger.info(f"🧠 Profiling: AutoModelForSpeechSeq2Seq.from_pretrained for {self.model_id} took {model_load_duration:.2f}s")

            logger.info(f"🧠 Profiling: Starting model.to({self.device}) for {self.model_id}...")
            model_to_device_start_time = time.monotonic()
            self.model.to(self.device)
            model_to_device_duration = time.monotonic() - model_to_device_start_time
            logger.info(f"🧠 Profiling: model.to({self.device}) for {self.model_id} took {model_to_device_duration:.2f}s")
            
            logger.info(f"🧠 Profiling: Starting AutoProcessor.from_pretrained for {self.model_id}...")
            processor_load_start_time = time.monotonic()
            self.processor = AutoProcessor.from_pretrained(self.model_id)
            processor_load_duration = time.monotonic() - processor_load_start_time
            logger.info(f"🧠 Profiling: AutoProcessor.from_pretrained for {self.model_id} took {processor_load_duration:.2f}s")
            
            # Create pipeline
            logger.info(f"🧠 Profiling: Starting STT pipeline creation for {self.model_id}...")
            pipeline_create_start_time = time.monotonic()
            self.pipeline = pipeline(
                task="automatic-speech-recognition",
                model=self.model,
                tokenizer=self.processor.tokenizer,
                feature_extractor=self.processor.feature_extractor,
                torch_dtype=self.torch_dtype,
                device=self.device,
            )
            pipeline_create_duration = time.monotonic() - pipeline_create_start_time
            logger.info(f"🧠 Profiling: STT pipeline creation for {self.model_id} took {pipeline_create_duration:.2f}s")
            
            overall_duration = time.monotonic() - overall_start_time
            logger.info(f"✅ STT model ({self.model_id}) loaded! Total time: {overall_duration:.2f}s")
            
            # Warm up the model
            self._warmup_model()
            
            # Mark as available
            self._set_available(True)
            
        except Exception as e:
            logger.error(f"❌ STT model ({self.model_id}) failed to load: {e}")
            self._set_available(False)
            raise STTError(f"Failed to initialize HuggingFace STT engine: {e}") from e
    
    def _warmup_model(self) -> None:
        """Warm up the model with dummy input."""
        try:
            logger.info(f"🧠 Profiling: Starting STT model warmup for {self.model_id}...")
            warmup_start_time = time.monotonic()
            warmup_audio = np.zeros((16000,), dtype=self.np_dtype)
            self.pipeline(warmup_audio)
            warmup_duration = time.monotonic() - warmup_start_time
            logger.info(f"✅ STT model warmup complete for {self.model_id}. Took {warmup_duration:.2f}s")
        except Exception as e:
            logger.warning(f"Model warmup failed: {e}")
    
    async def transcribe_with_sample_rate(self, audio_array: np.ndarray, sample_rate: int) -> TranscriptionResult:
        """Transcribe audio with explicit sample rate handling.
        
        Args:
            audio_array: Audio data as numpy array
            sample_rate: Sample rate of the audio
            
        Returns:
            TranscriptionResult: Transcription result with metadata
        """
        if not self.pipeline:
            raise STTError("Pipeline not initialized")
        
        try:
            logger.info(f"🔧 STT Engine Debug: Input array shape={audio_array.shape}, dtype={audio_array.dtype}, SR={sample_rate}")
            
            # Ensure audio array is valid
            if not isinstance(audio_array, np.ndarray):
                audio_array = np.array(audio_array, dtype=self.np_dtype)
            
            # Convert to target dtype if needed
            if audio_array.dtype != self.np_dtype:
                logger.info(f"🔧 STT Engine Debug: Converting dtype from {audio_array.dtype} to {self.np_dtype}")
                audio_array = audio_array.astype(self.np_dtype)
            
            logger.info(f"🔧 STT Engine Debug: Audio range=[{np.min(audio_array):.6f}, {np.max(audio_array):.6f}], size={audio_array.size}")
            
            # Run transcription in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            outputs = await loop.run_in_executor(
                None,
                self._run_pipeline,
                audio_array,
                sample_rate
            )
            
            # Extract transcription text
            text = outputs.get("text", "").strip()
            logger.info(f"📝 Transcribed: '{text}'")
            
            # Enhanced language detection using debug_audio_pipeline.py logic
            language = None
            confidence = None
            chunks = None
            
            # Method 1: Try to get language from Whisper outputs (like debug_audio_pipeline.py)
            if isinstance(outputs, dict):
                # Direct language field
                if 'language' in outputs:
                    language = outputs['language']
                    logger.info(f"🎤 Whisper detected language (direct): {language}")
                
                # From chunks if available
                elif 'chunks' in outputs and outputs['chunks']:
                    chunks = outputs['chunks']
                    for chunk in chunks:
                        if isinstance(chunk, dict) and 'language' in chunk:
                            language = chunk['language']
                            logger.info(f"🎤 Language from chunk: {language}")
                            break
                
                # Get chunks for metadata
                if 'chunks' in outputs:
                    chunks = outputs['chunks']
            
            # Method 2: Enhanced text-based language detection (from debug_audio_pipeline.py)
            if not language and text:
                text_lower = text.lower()
                
                # Enhanced Italian detection
                italian_words = [
                    'ciao', 'come', 'stai', 'sono', 'mi', 'chiamo', 'voglio', 'parlare', 'italiano',
                    'bene', 'grazie', 'prego', 'scusa', 'dove', 'quando', 'perché', 'cosa',
                    'casa', 'famiglia', 'lavoro', 'tempo', 'oggi', 'ieri', 'domani',
                    'mangiare', 'bere', 'dormire', 'andare', 'venire', 'fare', 'dire',
                    'molto', 'poco', 'grande', 'piccolo', 'bello', 'brutto', 'buono', 'cattivo'
                ]
                
                # Enhanced English detection
                english_words = [
                    'hello', 'how', 'are', 'you', 'my', 'name', 'is', 'want', 'speak', 'english',
                    'good', 'bad', 'yes', 'no', 'please', 'thank', 'sorry', 'where', 'when', 'why', 'what',
                    'home', 'family', 'work', 'time', 'today', 'yesterday', 'tomorrow',
                    'eat', 'drink', 'sleep', 'go', 'come', 'do', 'say', 'make', 'get', 'take',
                    'very', 'little', 'big', 'small', 'beautiful', 'ugly', 'good', 'bad'
                ]
                
                # Count matches for each language
                italian_matches = sum(1 for word in italian_words if word in text_lower)
                english_matches = sum(1 for word in english_words if word in text_lower)
                
                # Determine language based on matches
                if italian_matches > english_matches and italian_matches > 0:
                    language = 'it'
                    logger.info(f"🇮🇹 Language inferred from text (Italian words: {italian_matches}): {language}")
                elif english_matches > 0:
                    language = 'en'
                    logger.info(f"🇺🇸 Language inferred from text (English words: {english_matches}): {language}")
                else:
                    # Fallback: check for common patterns
                    if any(pattern in text_lower for pattern in ['mi chiamo', 'come stai', 'molto bene', 'va bene']):
                        language = 'it'
                        logger.info(f"🇮🇹 Language inferred from Italian patterns: {language}")
                    elif any(pattern in text_lower for pattern in ['my name', 'how are', 'very good', 'i am']):
                        language = 'en'
                        logger.info(f"🇺🇸 Language inferred from English patterns: {language}")
            
            return TranscriptionResult(
                text=text,
                language=language,
                confidence=confidence,
                chunks=chunks
            )
            
        except Exception as e:
            raise STTError(f"Transcription failed: {e}") from e

    async def _transcribe_audio(self, audio) -> TranscriptionResult:
        """Transcribe audio using HuggingFace pipeline.
        
        Args:
            audio: Audio data to transcribe (AudioData object or numpy array)
            
        Returns:
            TranscriptionResult: Transcription result with metadata
        """
        if not self.pipeline:
            raise STTError("Pipeline not initialized")
        
        try:
            # SIMPLIFIED: Handle both AudioData objects and raw numpy arrays efficiently
            logger.info(f"🔧 STT Engine Debug: Input type={type(audio)}")
            
            # Extract audio array from input
            if hasattr(audio, 'samples'):
                # AudioData object - extract samples
                audio_array = audio.samples
                logger.info(f"🔧 STT Engine Debug: AudioData object - samples shape={audio_array.shape}, dtype={audio_array.dtype}")
            else:
                # Raw numpy array - use directly
                audio_array = audio
                logger.info(f"🔧 STT Engine Debug: Raw array - shape={audio_array.shape}, dtype={audio_array.dtype}")
            
            # Ensure audio array is valid
            if not isinstance(audio_array, np.ndarray):
                audio_array = np.array(audio_array, dtype=self.np_dtype)
            
            logger.info(f"🔧 STT Engine Debug: Audio range=[{np.min(audio_array):.6f}, {np.max(audio_array):.6f}], size={audio_array.size}")
            
            # Convert to target dtype if needed
            if audio_array.dtype != self.np_dtype:
                logger.info(f"🔧 STT Engine Debug: Converting dtype from {audio_array.dtype} to {self.np_dtype}")
                audio_array = audio_array.astype(self.np_dtype)
            
            # Run transcription in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            outputs = await loop.run_in_executor(
                None,
                self._run_pipeline,
                audio_array,
                16000  # Default to 16kHz, will be overridden by gradio2.py
            )
            
            # Extract transcription text
            text = outputs.get("text", "").strip()
            logger.info(f"📝 Transcribed: '{text}'")
            
            # Enhanced language detection using debug_audio_pipeline.py logic
            language = None
            confidence = None
            chunks = None
            
            # Method 1: Try to get language from Whisper outputs (like debug_audio_pipeline.py)
            if isinstance(outputs, dict):
                # Direct language field
                if 'language' in outputs:
                    language = outputs['language']
                    logger.info(f"🎤 Whisper detected language (direct): {language}")
                
                # From chunks if available
                elif 'chunks' in outputs and outputs['chunks']:
                    chunks = outputs['chunks']
                    for chunk in chunks:
                        if isinstance(chunk, dict) and 'language' in chunk:
                            language = chunk['language']
                            logger.info(f"🎤 Language from chunk: {language}")
                            break
                
                # Get chunks for metadata
                if 'chunks' in outputs:
                    chunks = outputs['chunks']
            
            # Method 2: Enhanced text-based language detection (from debug_audio_pipeline.py)
            if not language and text:
                text_lower = text.lower()
                
                # Enhanced Italian detection
                italian_words = [
                    'ciao', 'come', 'stai', 'sono', 'mi', 'chiamo', 'voglio', 'parlare', 'italiano',
                    'bene', 'grazie', 'prego', 'scusa', 'dove', 'quando', 'perché', 'cosa',
                    'casa', 'famiglia', 'lavoro', 'tempo', 'oggi', 'ieri', 'domani',
                    'mangiare', 'bere', 'dormire', 'andare', 'venire', 'fare', 'dire',
                    'molto', 'poco', 'grande', 'piccolo', 'bello', 'brutto', 'buono', 'cattivo'
                ]
                
                # Enhanced English detection
                english_words = [
                    'hello', 'how', 'are', 'you', 'my', 'name', 'is', 'want', 'speak', 'english',
                    'good', 'bad', 'yes', 'no', 'please', 'thank', 'sorry', 'where', 'when', 'why', 'what',
                    'home', 'family', 'work', 'time', 'today', 'yesterday', 'tomorrow',
                    'eat', 'drink', 'sleep', 'go', 'come', 'do', 'say', 'make', 'get', 'take',
                    'very', 'little', 'big', 'small', 'beautiful', 'ugly', 'good', 'bad'
                ]
                
                # Count matches for each language
                italian_matches = sum(1 for word in italian_words if word in text_lower)
                english_matches = sum(1 for word in english_words if word in text_lower)
                
                # Determine language based on matches
                if italian_matches > english_matches and italian_matches > 0:
                    language = 'it'
                    logger.info(f"🇮🇹 Language inferred from text (Italian words: {italian_matches}): {language}")
                elif english_matches > 0:
                    language = 'en'
                    logger.info(f"🇺🇸 Language inferred from text (English words: {english_matches}): {language}")
                else:
                    # Fallback: check for common patterns
                    if any(pattern in text_lower for pattern in ['mi chiamo', 'come stai', 'molto bene', 'va bene']):
                        language = 'it'
                        logger.info(f"🇮🇹 Language inferred from Italian patterns: {language}")
                    elif any(pattern in text_lower for pattern in ['my name', 'how are', 'very good', 'i am']):
                        language = 'en'
                        logger.info(f"🇺🇸 Language inferred from English patterns: {language}")
            
            return TranscriptionResult(
                text=text,
                language=language,
                confidence=confidence,
                chunks=chunks
            )
            
        except Exception as e:
            raise STTError(f"Transcription failed: {e}") from e
    
    def _run_pipeline(self, audio_array: np.ndarray, sample_rate: int = 16000) -> Dict[str, Any]:
        """Run the pipeline synchronously with proper sample rate handling.
        
        Args:
            audio_array: Audio data as numpy array
            sample_rate: Sample rate of input audio
            
        Returns:
            Dict[str, Any]: Pipeline outputs
        """
        # CRITICAL FIX: Handle sample rate conversion
        if sample_rate != 16000:
            logger.info(f"🔧 STT Debug: Resampling from {sample_rate}Hz to 16000Hz")
            # Simple downsampling for 48kHz -> 16kHz (3:1 ratio)
            if sample_rate == 48000:
                audio_array = audio_array[::3]  # Take every 3rd sample
                logger.info(f"🔧 STT Debug: Downsampled to {audio_array.size} samples")
            else:
                # For other sample rates, use simple decimation
                ratio = sample_rate // 16000
                if ratio > 1:
                    audio_array = audio_array[::ratio]
        
        # SIMPLIFIED: Use audio array directly (no audio_to_bytes needed)
        logger.info(f"🔧 STT Debug: Using direct audio array for Whisper")
        
        # CRITICAL FIX: Use exact same pipeline call as V4 but with optimizations
        return self.pipeline(
            audio_array,
            chunk_length_s=30,
            batch_size=1,
            generate_kwargs={
                'task': 'transcribe',
                'language': None,  # Let Whisper auto-detect
            },
            return_timestamps=False,  # KEY FIX: Match V4 exactly
        )
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model.
        
        Returns:
            Dict[str, Any]: Model information
        """
        return {
            'model_id': self.model_id,
            'device': self.device,
            'torch_dtype': str(self.torch_dtype),
            'np_dtype': str(self.np_dtype),
            'is_available': self.is_available(),
            'model_loaded': self.model is not None,
            'pipeline_loaded': self.pipeline is not None
        }
    
    def is_available(self) -> bool:
        """Check if the STT engine is available and ready.
        
        Returns:
            bool: True if engine is ready, False otherwise
        """
        return (
            super().is_available() and 
            self.model is not None and 
            self.pipeline is not None
        )