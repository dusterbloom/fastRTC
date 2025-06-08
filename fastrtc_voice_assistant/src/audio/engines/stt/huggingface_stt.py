"""HuggingFace STT engine implementation."""

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
            logger.info(f"🧠 Loading STT model (Hugging Face: {self.model_id})...")
            
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
            self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
                self.model_id,
                torch_dtype=self.torch_dtype,
                low_cpu_mem_usage=True,
                use_safetensors=True,
                attn_implementation=attention
            )
            self.model.to(self.device)
            
            self.processor = AutoProcessor.from_pretrained(self.model_id)
            
            # Create pipeline
            self.pipeline = pipeline(
                task="automatic-speech-recognition",
                model=self.model,
                tokenizer=self.processor.tokenizer,
                feature_extractor=self.processor.feature_extractor,
                torch_dtype=self.torch_dtype,
                device=self.device,
            )
            
            logger.info(f"✅ STT model ({self.model_id}) loaded!")
            
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
            logger.info("Warming up STT model with dummy input...")
            warmup_audio = np.zeros((16000,), dtype=self.np_dtype)
            self.pipeline(warmup_audio)
            logger.info("✅ STT model warmup complete.")
        except Exception as e:
            logger.warning(f"Model warmup failed: {e}")
    
    async def _transcribe_audio(self, audio: AudioData) -> TranscriptionResult:
        """Transcribe audio using HuggingFace pipeline.
        
        Args:
            audio: Audio data to transcribe
            
        Returns:
            TranscriptionResult: Transcription result with metadata
        """
        if not self.pipeline:
            raise STTError("Pipeline not initialized")
        
        try:
            # Convert AudioData to format expected by pipeline
            audio_array = audio.samples
            if audio_array.dtype != self.np_dtype:
                audio_array = audio_array.astype(self.np_dtype)
            
            # Run transcription in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            outputs = await loop.run_in_executor(
                None,
                self._run_pipeline,
                audio_array
            )
            
            # Extract transcription text
            text = outputs.get("text", "").strip()
            logger.info(f"📝 Transcribed: '{text}'")
            
            # Extract language information if available
            language = None
            confidence = None
            chunks = None
            
            # Try to get language from outputs
            if hasattr(outputs, 'get'):
                if 'language' in outputs:
                    language = outputs['language']
                    logger.info(f"🎤 Whisper detected language: {language}")
                elif 'chunks' in outputs and outputs['chunks']:
                    # Try to get language from first chunk
                    first_chunk = outputs['chunks'][0]
                    if isinstance(first_chunk, dict) and 'language' in first_chunk:
                        language = first_chunk['language']
                        logger.info(f"🎤 Language from chunk: {language}")
                
                # Get chunks if available
                if 'chunks' in outputs:
                    chunks = outputs['chunks']
            
            return TranscriptionResult(
                text=text,
                language=language,
                confidence=confidence,
                chunks=chunks
            )
            
        except Exception as e:
            raise STTError(f"Transcription failed: {e}") from e
    
    def _run_pipeline(self, audio_array: np.ndarray) -> Dict[str, Any]:
        """Run the pipeline synchronously.
        
        Args:
            audio_array: Audio data as numpy array
            
        Returns:
            Dict[str, Any]: Pipeline outputs
        """
        return self.pipeline(
            audio_array,
            chunk_length_s=30,
            batch_size=1,
            generate_kwargs={'task': 'transcribe'},
            return_timestamps=False,
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