"""Kokoro TTS engine implementation."""

import asyncio
import numpy as np
from typing import List, Dict, Any, Optional, Generator, Tuple
from dataclasses import dataclass

from .base import BaseTTSEngine
from ....core.interfaces import AudioData
from ....core.exceptions import TTSError
from ....config.language_config import KOKORO_VOICE_MAP, KOKORO_TTS_LANG_MAP, DEFAULT_LANGUAGE
from ....utils.logging import get_logger

logger = get_logger(__name__)

# Import KokoroONNX at module level for testing compatibility
try:
    from kokoro_onnx import KokoroONNX
except ImportError:
    from .kokoro_onnx_stub import KokoroONNX


@dataclass
class KokoroTTSOptions:
    """Options for Kokoro TTS synthesis."""
    speed: float = 1.0
    lang: str = "en-us"
    voice: Optional[str] = None


class KokoroTTSEngine(BaseTTSEngine):
    """Kokoro TTS engine implementation.
    
    This engine provides text-to-speech synthesis using the Kokoro TTS model
    with support for multiple languages and voices.
    """
    
    def __init__(self):
        """Initialize the Kokoro TTS engine."""
        super().__init__()
        
        self.tts_model = None
        self._initialize_model()
    
    def _initialize_model(self) -> None:
        """Initialize the Kokoro TTS model."""
        try:
            logger.info("🗣️ Loading TTS model (Kokoro)...")
            
            # Initialize Kokoro TTS using module-level import
            self.tts_model = KokoroONNX()
            
            # Log which implementation is being used
            if hasattr(self.tts_model, 'model') and hasattr(self.tts_model.model, 'voices'):
                logger.info("✅ Using real Kokoro ONNX implementation")
            else:
                logger.info("⚠️ Using Kokoro ONNX stub implementation (for testing)")
            
            # Check available voices
            try:
                if hasattr(self.tts_model, 'model') and hasattr(self.tts_model.model, 'voices'):
                    available_voices = getattr(self.tts_model.model, 'voices', [])
                    if available_voices:
                        logger.info(f"Kokoro TTS: Available voice names (first few): {list(available_voices)[:5]}")
                    else:
                        logger.warning("Kokoro TTS: No voices found in model")
                else:
                    logger.warning("Kokoro TTS: Could not access voice information")
            except Exception as e:
                logger.debug(f"Could not check voice information (likely mocked): {e}")
            
            logger.info("✅ Kokoro TTS model loaded successfully!")
            self._set_available(True)
            
        except ImportError as e:
            logger.error(f"❌ Kokoro TTS not available: {e}")
            logger.error("Please install kokoro-onnx: pip install kokoro-onnx")
            self._set_available(False)
        except Exception as e:
            logger.error(f"❌ Failed to load Kokoro TTS model: {e}")
            self._set_available(False)
    
    async def _synthesize_text(self, text: str, voice: str, language: str) -> AudioData:
        """Synthesize text using Kokoro TTS.
        
        Args:
            text: Text to synthesize
            voice: Voice identifier
            language: Language code
            
        Returns:
            AudioData: Synthesized audio
        """
        if not self.tts_model:
            raise TTSError("Kokoro TTS model not initialized")
        
        try:
            # Prepare TTS options
            options_params = {"speed": 1.05}
            kokoro_tts_lang = KOKORO_TTS_LANG_MAP.get(language, 'en-us')
            options_params["lang"] = kokoro_tts_lang
            
            if voice:
                options_params["voice"] = voice
            
            tts_options = KokoroTTSOptions(**options_params)
            logger.info(f"🔊 Synthesizing with voice '{voice}', lang '{kokoro_tts_lang}'")
            
            # Run synthesis in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            audio_chunks = await loop.run_in_executor(
                None,
                self._run_synthesis,
                text,
                tts_options
            )
            
            if not audio_chunks:
                raise TTSError("No audio generated")
            
            # Combine all chunks into single audio data
            combined_audio = self._combine_audio_chunks(audio_chunks)
            
            logger.info(f"✅ TTS synthesis completed. Chunks: {len(audio_chunks)}, Samples: {len(combined_audio.samples)}")
            
            return combined_audio
            
        except Exception as e:
            raise TTSError(f"Kokoro TTS synthesis failed: {e}") from e
    
    def _run_synthesis(self, text: str, options: KokoroTTSOptions) -> List[AudioData]:
        """Run synthesis synchronously and collect all chunks.
        
        Args:
            text: Text to synthesize
            options: TTS options
            
        Returns:
            List[AudioData]: List of audio chunks
        """
        audio_chunks = []
        
        try:
            for tts_output_item in self.tts_model.stream_tts_sync(text, options):
                if isinstance(tts_output_item, tuple) and len(tts_output_item) == 2:
                    sample_rate, audio_array = tts_output_item
                    if isinstance(audio_array, np.ndarray) and audio_array.size > 0:
                        duration = len(audio_array) / sample_rate if sample_rate > 0 else 0.0
                        audio_data = AudioData(
                            samples=audio_array.astype(np.float32),
                            sample_rate=sample_rate,
                            duration=duration
                        )
                        audio_chunks.append(audio_data)
                        
                elif isinstance(tts_output_item, np.ndarray) and tts_output_item.size > 0:
                    # Assume default sample rate if not provided
                    sample_rate = 24000  # Kokoro default
                    duration = len(tts_output_item) / sample_rate
                    audio_data = AudioData(
                        samples=tts_output_item.astype(np.float32),
                        sample_rate=sample_rate,
                        duration=duration
                    )
                    audio_chunks.append(audio_data)
                    
        except Exception as e:
            logger.error(f"Error during synthesis streaming: {e}")
            raise
        
        return audio_chunks
    
    def _combine_audio_chunks(self, chunks: List[AudioData]) -> AudioData:
        """Combine multiple audio chunks into a single AudioData object.
        
        Args:
            chunks: List of audio chunks to combine
            
        Returns:
            AudioData: Combined audio data
        """
        if not chunks:
            raise TTSError("No audio chunks to combine")
        
        if len(chunks) == 1:
            return chunks[0]
        
        # Ensure all chunks have the same sample rate
        sample_rate = chunks[0].sample_rate
        for chunk in chunks[1:]:
            if chunk.sample_rate != sample_rate:
                logger.warning(f"Sample rate mismatch: {chunk.sample_rate} vs {sample_rate}")
        
        # Concatenate all audio samples
        combined_samples = np.concatenate([chunk.samples for chunk in chunks])
        total_duration = sum(chunk.duration for chunk in chunks)
        
        return AudioData(
            samples=combined_samples,
            sample_rate=sample_rate,
            duration=total_duration
        )
    
    def get_available_voices(self, language: str) -> List[str]:
        """Get available voices for a language.
        
        Args:
            language: Language code
            
        Returns:
            List[str]: Available voice identifiers
        """
        voices = KOKORO_VOICE_MAP.get(language, KOKORO_VOICE_MAP.get(DEFAULT_LANGUAGE, []))
        return voices.copy() if voices else []
    
    def stream_synthesis(self, text: str, voice: str, language: str) -> Generator[Tuple[int, np.ndarray], None, None]:
        """Stream synthesis results as they are generated.
        
        Args:
            text: Text to synthesize
            voice: Voice identifier
            language: Language code
            
        Yields:
            Tuple[int, np.ndarray]: (sample_rate, audio_chunk)
        """
        if not self.tts_model:
            raise TTSError("Kokoro TTS model not initialized")
        
        try:
            # Prepare TTS options
            options_params = {"speed": 1.05}
            kokoro_tts_lang = KOKORO_TTS_LANG_MAP.get(language, 'en-us')
            options_params["lang"] = kokoro_tts_lang
            
            if voice:
                options_params["voice"] = voice
            
            tts_options = KokoroTTSOptions(**options_params)
            logger.info(f"🔊 Streaming synthesis with voice '{voice}', lang '{kokoro_tts_lang}'")
            
            chunk_count = 0
            total_samples = 0
            
            for tts_output_item in self.tts_model.stream_tts_sync(text, tts_options):
                if isinstance(tts_output_item, tuple) and len(tts_output_item) == 2:
                    sample_rate, audio_array = tts_output_item
                    if isinstance(audio_array, np.ndarray) and audio_array.size > 0:
                        chunk_count += 1
                        total_samples += audio_array.size
                        
                        # Yield smaller chunks to prevent timeouts
                        chunk_size = min(1024, audio_array.size)
                        for i in range(0, audio_array.size, chunk_size):
                            mini_chunk = audio_array[i:i+chunk_size]
                            if mini_chunk.size > 0:
                                yield (sample_rate, mini_chunk.astype(np.float32))
                                
                elif isinstance(tts_output_item, np.ndarray) and tts_output_item.size > 0:
                    chunk_count += 1
                    total_samples += tts_output_item.size
                    sample_rate = 24000  # Kokoro default
                    
                    chunk_size = min(1024, tts_output_item.size)
                    for i in range(0, tts_output_item.size, chunk_size):
                        mini_chunk = tts_output_item[i:i+chunk_size]
                        if mini_chunk.size > 0:
                            yield (sample_rate, mini_chunk.astype(np.float32))
            
            logger.info(f"✅ Streaming synthesis completed. Chunks: {chunk_count}, Samples: {total_samples}")
            
        except Exception as e:
            logger.error(f"❌ Streaming synthesis failed: {e}")
            raise TTSError(f"Streaming synthesis failed: {e}") from e
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model.
        
        Returns:
            Dict[str, Any]: Model information
        """
        info = {
            'model_type': 'Kokoro ONNX',
            'is_available': self.is_available(),
            'model_loaded': self.tts_model is not None,
            'supported_languages': list(KOKORO_TTS_LANG_MAP.keys()),
            'voice_map': KOKORO_VOICE_MAP
        }
        
        if self.tts_model and hasattr(self.tts_model, 'model'):
            if hasattr(self.tts_model.model, 'voices'):
                info['available_voices'] = list(getattr(self.tts_model.model, 'voices', []))
        
        return info
    
    def is_available(self) -> bool:
        """Check if the TTS engine is available and ready.
        
        Returns:
            bool: True if engine is ready, False otherwise
        """
        return super().is_available() and self.tts_model is not None