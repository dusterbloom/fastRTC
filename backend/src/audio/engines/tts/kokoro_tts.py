"""Kokoro TTS engine implementation."""

import time
import logging
import asyncio
import os
import numpy as np
from typing import List, Dict, Any, Optional, Generator, Tuple, TYPE_CHECKING
from dataclasses import dataclass

if TYPE_CHECKING:
    from typing import AsyncGenerator

from .base import BaseTTSEngine
from ....core.interfaces import AudioData
from ....core.exceptions import TTSError
from ....config.language_config import KOKORO_VOICE_MAP, KOKORO_TTS_LANG_MAP, DEFAULT_LANGUAGE
from ....utils.logging import get_logger, time_tts, create_tts_timer

logger = get_logger(__name__)
logger.setLevel(logging.INFO)

# Import real fastRTC TTS model like V4
try:
    from fastrtc import get_tts_model, KokoroTTSOptions
    USE_FASTRTC_TTS = True
except ImportError:
    # Fallback to stub only if fastRTC is not available
    from .kokoro_onnx_stub import KokoroONNX, KokoroTTSOptions
    USE_FASTRTC_TTS = False


class KokoroTTSEngine(BaseTTSEngine):
    """Kokoro TTS engine implementation.
    
    This engine provides text-to-speech synthesis using the Kokoro TTS model
    with support for multiple languages and voices.
    """
    
    def __init__(self):
        """Initialize the Kokoro TTS engine."""
        super().__init__()
        
        self.tts_model = None
        self.should_interrupt = False  # Flag for interrupting TTS synthesis
        self._initialize_model()
    
    def _initialize_model(self) -> None:
        """Initialize the Kokoro TTS model using fastRTC like V4."""
        try:
            logger.info("🧠 Profiling: Starting Kokoro TTS model load (via fastRTC)...")
            overall_start_time = time.monotonic()
            
            if USE_FASTRTC_TTS:
                # Use real fastRTC TTS model like V4
                logger.info("🧠 Profiling: Calling get_tts_model('kokoro')...")
                model_load_start_time = time.monotonic()
                self.tts_model = get_tts_model("kokoro")
                model_load_duration = time.monotonic() - model_load_start_time
                logger.info(f"🧠 Profiling: get_tts_model('kokoro') took {model_load_duration:.2f}s")
                logger.info("✅ Using real fastRTC Kokoro TTS implementation")
                
                # Check and configure GPU acceleration
                self._configure_gpu_acceleration()
                
                # Check available voices like V4
                try:
                    if hasattr(self.tts_model, 'model') and hasattr(self.tts_model.model, 'voices'):
                        available_voices = getattr(self.tts_model.model, 'voices', [])
                        if available_voices:
                            logger.info(f"Kokoro TTS: Available voice names (first few): {list(available_voices)[:5]}")
                        else:
                            logger.info("Kokoro TTS: Could not list specific voice names from model.")
                    else:
                        logger.info("Kokoro TTS: Voice listing not directly available via tts_model.model.voices.")
                except Exception as e:
                    logger.debug(f"Could not check voice information: {e}")
            else:
                # Fallback to stub implementation
                self.tts_model = KokoroONNX()
                logger.warning("⚠️ Using Kokoro ONNX stub implementation (fastRTC not available)")
            overall_duration = time.monotonic() - overall_start_time
            logger.info(f"✅ Kokoro TTS model loaded successfully! Total time: {overall_duration:.2f}s")
            self._set_available(True)
            # --- CRITICAL DEBUG: Log available voices after model load ---
            try:
                available_voices = []
                if hasattr(self.tts_model, "model") and hasattr(self.tts_model.model, "voices"):
                    available_voices = list(getattr(self.tts_model.model, "voices", []))
                elif hasattr(self.tts_model, "voices"):
                    available_voices = list(getattr(self.tts_model, "voices", []))
                logger.critical(f"[TTS CRITICAL] Available voices in TTS model after load: {available_voices[:10]}")
            except Exception as e:
                logger.critical(f"[TTS CRITICAL] Could not list available voices from TTS model: {e}")
            
            
        except ImportError as e:
            logger.error(f"❌ Kokoro TTS not available: {e}")
            logger.error("Please install fastRTC or kokoro-onnx")
            self._set_available(False)
        except Exception as e:
            logger.error(f"❌ Failed to load Kokoro TTS model: {e}")
            self._set_available(False)
    
    def _configure_gpu_acceleration(self) -> None:
        """Configure GPU acceleration for Kokoro TTS if available."""
        try:
            import onnxruntime as ort
            
            # Check available providers
            available_providers = ort.get_available_providers()
            logger.info(f"🔧 Available ONNX providers: {available_providers}")
            
            # Check if GPU providers are available
            gpu_providers = [p for p in available_providers if 'CUDA' in p or 'Tensorrt' in p]
            
            if gpu_providers:
                logger.info(f"🚀 GPU acceleration available with providers: {gpu_providers}")
                
                # Set environment variable to force GPU usage in kokoro-onnx
                # This will be picked up by the Kokoro constructor
                preferred_provider = gpu_providers[0]  # Use the first available GPU provider
                os.environ['ONNX_PROVIDER'] = preferred_provider
                logger.info(f"🔧 Set ONNX_PROVIDER environment variable to: {preferred_provider}")
                
                # Try to configure the underlying Kokoro model for GPU if possible
                if hasattr(self.tts_model, 'model') and hasattr(self.tts_model.model, 'sess'):
                    # This is for kokoro-onnx models that expose the ONNX session
                    try:
                        current_providers = self.tts_model.model.sess.get_providers()
                        logger.info(f"🔧 Current model providers: {current_providers}")
                        
                        # If not using GPU, try to recreate the session
                        if not any('CUDA' in p or 'Tensorrt' in p for p in current_providers):
                            logger.info("🔧 Model not using GPU, attempting to recreate session...")
                            
                            # Get model path from the session
                            if hasattr(self.tts_model.model, 'config') and hasattr(self.tts_model.model.config, 'model_path'):
                                model_path = self.tts_model.model.config.model_path
                                logger.info(f"🔧 Recreating ONNX session with GPU providers for {model_path}")
                                
                                # Create new session with GPU providers prioritized
                                session_options = ort.SessionOptions()
                                session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                                providers = gpu_providers + ['CPUExecutionProvider']
                                
                                new_session = ort.InferenceSession(
                                    model_path, 
                                    sess_options=session_options,
                                    providers=providers
                                )
                                
                                # Replace the session if successful
                                self.tts_model.model.sess = new_session
                                logger.info(f"✅ Successfully configured Kokoro TTS with GPU acceleration: {new_session.get_providers()}")
                            else:
                                logger.info("🔧 Model path not accessible, GPU configuration via environment variable")
                        else:
                            logger.info(f"✅ Model already using GPU providers: {current_providers}")
                            
                    except Exception as gpu_config_error:
                        logger.warning(f"⚠️ Could not configure GPU acceleration: {gpu_config_error}")
                        logger.info("🔧 Falling back to environment variable configuration")
                else:
                    logger.info("🔧 Model session not directly accessible, GPU configuration via environment variable")
                    
            else:
                logger.warning("⚠️ No GPU providers available, using CPU execution")
                
        except ImportError:
            logger.warning("⚠️ ONNX Runtime not available for GPU configuration")
        except Exception as e:
            logger.warning(f"⚠️ Error configuring GPU acceleration: {e}")
    
    @time_tts("TTS Synthesis")
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
            # Log the text being synthesized
            logger.info(f"🤖 ASSISTANT: '{text}'")
            if os.getenv("DEBUG_TTS", "false").lower() == "true":
                logger.debug(f"🔤 TTS Text Input: '{text}' (length: {len(text)} chars, words: {len(text.split())})")
            
            # Prepare TTS options
            options_params = {"speed": 1.05}
            kokoro_tts_lang = KOKORO_TTS_LANG_MAP.get(language, 'en-us')
            options_params["lang"] = kokoro_tts_lang
            
            if voice:
                options_params["voice"] = voice
            
            tts_options = KokoroTTSOptions(**options_params)
            
            # Enhanced logging with debug details
            if os.getenv("DEBUG_TTS", "false").lower() == "true":
                logger.debug(f"🔧 TTS Options: voice='{voice}', lang='{kokoro_tts_lang}', speed={options_params['speed']}")
            else:
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
                # Check for interruption before processing each chunk
                if self.should_interrupt:
                    logger.info("🛑 TTS synthesis interrupted")
                    break
                    
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
    
    @time_tts("TTS Stream Synthesis")
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
            # Log the text being streamed
            logger.info(f"🤖 ASSISTANT: '{text}'")
            logger.info(f"🌊 TTS Stream Input: '{text}' (length: {len(text)} chars, words: {len(text.split())})")
            logger.info(f"🔧 TTS Model Type: {type(self.tts_model)}")
            
            # Prepare TTS options
            options_params = {"speed": 1.05}
            kokoro_tts_lang = KOKORO_TTS_LANG_MAP.get(language, 'en-us')
            options_params["lang"] = kokoro_tts_lang
            
            if voice:
                options_params["voice"] = voice
            
            tts_options = KokoroTTSOptions(**options_params)
            
            # Enhanced logging
            logger.info(f"🔧 Stream TTS Options: voice='{voice}', lang='{kokoro_tts_lang}', speed={options_params['speed']}")
            logger.info(f"🔊 Streaming synthesis with voice '{voice}', lang '{kokoro_tts_lang}'")
            
            chunk_count = 0
            total_samples = 0
            
            logger.info(f"🔄 Starting stream_tts_sync iteration...")
            for tts_output_item in self.tts_model.stream_tts_sync(text, tts_options):
                # Check for interruption before processing each chunk
                if self.should_interrupt:
                    logger.info("🛑 TTS streaming synthesis interrupted")
                    break
                    
                if isinstance(tts_output_item, tuple) and len(tts_output_item) == 2:
                    sample_rate, audio_array = tts_output_item
                    if isinstance(audio_array, np.ndarray) and audio_array.size > 0:
                        chunk_count += 1
                        total_samples += audio_array.size
                        
                        # Yield optimized chunks to prevent timeouts and audio artifacts
                        chunk_size = min(2048, audio_array.size)  # Increased from 1024 to 2048
                        for i in range(0, audio_array.size, chunk_size):
                            # Check for interruption before yielding each mini-chunk
                            if self.should_interrupt:
                                logger.info("🛑 TTS streaming synthesis interrupted during mini-chunk")
                                return
                            mini_chunk = audio_array[i:i+chunk_size]
                            if mini_chunk.size > 0:
                                yield (sample_rate, mini_chunk.astype(np.float32))
                                
                elif isinstance(tts_output_item, np.ndarray) and tts_output_item.size > 0:
                    chunk_count += 1
                    total_samples += tts_output_item.size
                    sample_rate = 24000  # Kokoro default
                    
                    chunk_size = min(2048, tts_output_item.size)  # Increased from 1024 to 2048
                    for i in range(0, tts_output_item.size, chunk_size):
                        # Check for interruption before yielding each mini-chunk
                        if self.should_interrupt:
                            logger.info("🛑 TTS streaming synthesis interrupted during mini-chunk")
                            return
                        mini_chunk = tts_output_item[i:i+chunk_size]
                        if mini_chunk.size > 0:
                            yield (sample_rate, mini_chunk.astype(np.float32))
            
            logger.info(f"🔄 Finished stream_tts_sync iteration. Processing {chunk_count} chunks, {total_samples} samples")
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
    
    @time_tts("TTS Async Stream Synthesis")
    async def stream_synthesis_async(self, text: str, voice: str, language: str) -> "AsyncGenerator[Tuple[int, np.ndarray], None]":
        """
        Async streaming synthesis for real-time conversation flow.
        
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
            
            # Enhanced debug logging for async streaming
            if os.getenv("DEBUG_TTS", "false").lower() == "true":
                logger.debug(f"🌊 TTS Async Stream Input: '{text}' (length: {len(text)} chars, words: {len(text.split())})")
                logger.debug(f"🔧 Async Stream TTS Options: voice='{voice}', lang='{kokoro_tts_lang}', speed={options_params['speed']}")
            else:
                logger.debug(f"🔊 Async streaming synthesis: '{text[:50]}...' with voice '{voice}', lang '{kokoro_tts_lang}'")
            
            # Run synthesis in executor to avoid blocking the event loop
            loop = asyncio.get_event_loop()
            
            # Create a generator function that can be run in executor
            def _sync_generator():
                chunk_count = 0
                total_samples = 0
                chunks = []
                
                for tts_output_item in self.tts_model.stream_tts_sync(text, tts_options):
                    # Check for interruption before processing each chunk
                    if self.should_interrupt:
                        logger.info("🛑 TTS async streaming synthesis interrupted")
                        break
                        
                    if isinstance(tts_output_item, tuple) and len(tts_output_item) == 2:
                        sample_rate, audio_array = tts_output_item
                        if isinstance(audio_array, np.ndarray) and audio_array.size > 0:
                            chunk_count += 1
                            total_samples += audio_array.size
                            
                            # Split into optimized chunks for smooth streaming without artifacts
                            chunk_size = min(1536, audio_array.size)  # Increased from 512 to 1536 for better audio quality
                            for i in range(0, audio_array.size, chunk_size):
                                # Check for interruption before processing each mini-chunk
                                if self.should_interrupt:
                                    logger.info("🛑 TTS async streaming synthesis interrupted during mini-chunk")
                                    break
                                mini_chunk = audio_array[i:i+chunk_size]
                                if mini_chunk.size > 0:
                                    chunks.append((sample_rate, mini_chunk.astype(np.float32)))
                                    
                    elif isinstance(tts_output_item, np.ndarray) and tts_output_item.size > 0:
                        chunk_count += 1
                        total_samples += tts_output_item.size
                        sample_rate = 24000  # Kokoro default
                        
                        chunk_size = min(1536, tts_output_item.size)  # Increased from 512 to 1536 for better audio quality
                        for i in range(0, tts_output_item.size, chunk_size):
                            # Check for interruption before processing each mini-chunk
                            if self.should_interrupt:
                                logger.info("🛑 TTS async streaming synthesis interrupted during mini-chunk")
                                break
                            mini_chunk = tts_output_item[i:i+chunk_size]
                            if mini_chunk.size > 0:
                                chunks.append((sample_rate, mini_chunk.astype(np.float32)))
                
                logger.debug(f"✅ Async synthesis completed. Chunks: {chunk_count}, Total mini-chunks: {len(chunks)}, Samples: {total_samples}")
                return chunks
            
            # Run the synthesis in executor
            chunks = await loop.run_in_executor(None, _sync_generator)
            
            # Yield chunks asynchronously
            for sample_rate, audio_chunk in chunks:
                # Check for interruption before yielding each chunk
                if self.should_interrupt:
                    logger.info("🛑 TTS async synthesis interrupted during yielding")
                    break
                yield (sample_rate, audio_chunk)
                # Small yield to allow other coroutines to run
                await asyncio.sleep(0)
                
        except Exception as e:
            logger.error(f"❌ Async streaming synthesis failed: {e}")
            raise TTSError(f"Async streaming synthesis failed: {e}") from e
    
    @time_tts("TTS Sentence Synthesis")
    async def synthesize_sentence_async(self, sentence: str, voice: str, language: str) -> "AsyncGenerator[Tuple[int, np.ndarray], None]":
        """
        Synthesize a single sentence asynchronously for sentence-level streaming.
        Optimized for real-time conversation where sentences arrive from LLM streaming.
        
        Args:
            sentence: Complete sentence to synthesize
            voice: Voice identifier
            language: Language code
            
        Yields:
            Tuple[int, np.ndarray]: (sample_rate, audio_chunk)
        """
        if not sentence.strip():
            return
            
        # Enhanced sentence debug logging
        if os.getenv("DEBUG_TTS", "false").lower() == "true":
            logger.debug(f"📝 TTS Sentence Input: '{sentence}' (length: {len(sentence)} chars, words: {len(sentence.split())}, voice: {voice}, lang: {language})")
        else:
            logger.debug(f"🔊 Synthesizing sentence: '{sentence}' (voice: {voice}, lang: {language})")
        
        async for sample_rate, audio_chunk in self.stream_synthesis_async(sentence, voice, language):
            yield (sample_rate, audio_chunk)

    def is_available(self) -> bool:
        """Check if the TTS engine is available and ready.
        
        Returns:
            bool: True if engine is ready, False otherwise
        """
        return super().is_available() and self.tts_model is not None
    
    def interrupt(self) -> None:
        """Interrupt ongoing TTS synthesis."""
        self.should_interrupt = True
        logger.info("🛑 TTS interruption requested")
    
    def reset_interrupt(self) -> None:
        """Reset the interruption flag for new synthesis."""
        self.should_interrupt = False