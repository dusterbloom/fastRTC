"""Faster-whisper STT engine implementation."""

import asyncio
from pathlib import Path
import numpy as np
from faster_whisper import WhisperModel

from .base import BaseSTTEngine
from ....core.interfaces import TranscriptionResult, AudioData
from ....utils.logging import get_logger

logger = get_logger(__name__)

_MODEL_DIR = Path("/models/whisper-v3-ct2")   # baked into the image
_COMPUTE   = "int8_float16"                   # best perf/quality on Ampere


class FasterWhisperSTT(BaseSTTEngine):
    """
    Streaming multilingual STT using faster-whisper + CTranslate2.
    """

    def __init__(self):
        super().__init__()
        try:
            # Check if model directory exists (for local development)
            if not _MODEL_DIR.exists():
                # Try alternative path for development
                dev_model_dir = Path("./models/whisper-v3-ct2")
                if dev_model_dir.exists():
                    model_path = str(dev_model_dir)
                else:
                    # Fall back to downloading if not available
                    model_path = "large-v3"
                    logger.warning("Model directory not found, will download on first use")
            else:
                model_path = str(_MODEL_DIR)
            
            self.model = WhisperModel(
                model_path,
                device="cuda",
                compute_type=_COMPUTE,
            )
            
            # Verify multilingual model
            if hasattr(self.model, 'hf_tokenizer') and hasattr(self.model.hf_tokenizer, 'lang_to_id'):
                if "de" not in self.model.hf_tokenizer.lang_to_id:
                    logger.warning("Model may not be multilingual")
            
            self._set_available(True)
            logger.info(f"Initialized FasterWhisperSTT with model at {model_path}")
            
        except Exception as e:
            logger.error(f"Failed to initialize FasterWhisperSTT: {e}")
            self._set_available(False)
            raise

    async def _transcribe_audio(self, audio) -> TranscriptionResult:
        """Implement specific transcription logic.
        
        Args:
            audio: Audio data to transcribe (AudioData object or numpy array)
            
        Returns:
            TranscriptionResult: Transcription result
        """
        # Extract audio samples
        if isinstance(audio, AudioData):
            audio_samples = audio.samples
        else:
            # Assume it's a numpy array
            audio_samples = audio
        
        # Ensure float32 format
        if audio_samples.dtype != np.float32:
            audio_samples = audio_samples.astype(np.float32)
        
        # Run transcription in thread pool (faster-whisper is sync)
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, 
            self._transcribe_sync, 
            audio_samples
        )
        
        return result

    def _transcribe_sync(self, audio_samples: np.ndarray) -> TranscriptionResult:
        """Synchronous transcription method.
        
        Args:
            audio_samples: Audio samples as numpy array
            
        Returns:
            TranscriptionResult: Transcription result
        """
        segments, info = self.model.transcribe(
            audio_samples,
            vad_filter=True,      # built-in Silero VAD
            beam_size=1,          # Greedy search for speed
            language=None,        # Auto-detect language
            temperature=0.0,      # Deterministic
            word_timestamps=False # Disable for speed
        )
        
        # Collect all segments
        text_parts = []
        for segment in segments:
            text_parts.append(segment.text.strip())
        
        full_text = " ".join(text_parts)
        
        # Get language info
        detected_language = info.language if hasattr(info, 'language') else None
        language_probability = info.language_probability if hasattr(info, 'language_probability') else None
        
        return TranscriptionResult(
            text=full_text,
            language=detected_language,
            confidence=language_probability
        )

    def stream(self, pcm16_bytes):
        """
        Generator yielding partial transcripts (~1 s latency).
        For compatibility with streaming interfaces.
        """
        segments, _ = self.model.transcribe(
            pcm16_bytes,
            vad_filter=True,    # built-in Silero VAD
            chunk_size=1.0,
            beam_size=1,
        )
        for s in segments:
            yield s.text
