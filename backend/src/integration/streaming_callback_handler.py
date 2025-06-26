"""
Streaming Callback Handler Module

High-performance async streaming pipeline for STT→LLM→TTS.
Optimized for immediate response and fluid conversation UX.
"""

import asyncio
import time
import numpy as np
from typing import AsyncGenerator, Tuple, Any, Optional, TYPE_CHECKING
from fastrtc import AdditionalOutputs

if TYPE_CHECKING:
    from typing import AsyncGenerator

from ..core.interfaces import TranscriptionResult
from ..audio import STTEngine, KokoroTTSEngine, VoiceMapper
from ..utils.logging import get_logger
from ..config.audio_config import AUDIO_SAMPLE_RATE, SILENT_AUDIO_FRAME_TUPLE

logger = get_logger(__name__)

EMPTY_AUDIO_YIELD_OUTPUT = (SILENT_AUDIO_FRAME_TUPLE, AdditionalOutputs())


class StreamingPipeline:
    """
    Orchestrates streaming STT→LLM→TTS pipeline for real-time conversation.
    """
    
    def __init__(self, voice_assistant, stt_engine, tts_engine, voice_mapper):
        self.voice_assistant = voice_assistant
        self.stt_engine = stt_engine
        self.tts_engine = tts_engine
        self.voice_mapper = voice_mapper
        
        # Streaming state
        self.partial_transcript = ""
        self.confidence_threshold = 0.6
        self.min_words_for_processing = 2
        
    async def process_audio_stream(self, audio_array: np.ndarray, sample_rate: int) -> AsyncGenerator[Tuple[Tuple[int, np.ndarray], AdditionalOutputs], None]:
        """
        Process audio through full streaming pipeline.
        
        Args:
            audio_array: Audio data as numpy array
            sample_rate: Audio sample rate
            
        Yields:
            Audio chunks from TTS streaming
        """
        try:
            # Immediate STT processing - no buffering
            transcript_result = await self._stream_stt(audio_array)
            
            if not transcript_result or not transcript_result.text.strip():
                yield EMPTY_AUDIO_YIELD_OUTPUT
                return
                
            user_text = transcript_result.text.strip()
            
            # Check confidence before proceeding
            if hasattr(transcript_result, 'confidence') and transcript_result.confidence < self.confidence_threshold:
                # Store partial but don't process yet
                self.partial_transcript = user_text
                yield EMPTY_AUDIO_YIELD_OUTPUT
                return
                
            # Check minimum words threshold
            if len(user_text.split()) < self.min_words_for_processing:
                self.partial_transcript = user_text
                yield EMPTY_AUDIO_YIELD_OUTPUT
                return
                
            # Process complete utterance
            logger.info(f"🎤 Processing: '{user_text}'")
            
            # Stream LLM→TTS pipeline
            async for audio_chunk in self._stream_llm_to_tts(user_text):
                yield audio_chunk
                
        except Exception as e:
            logger.error(f"❌ Streaming pipeline error: {e}")
            yield EMPTY_AUDIO_YIELD_OUTPUT
    
    async def _stream_stt(self, audio_array: np.ndarray) -> Optional[TranscriptionResult]:
        """
        Stream STT processing with immediate results.
        
        Args:
            audio_array: Audio data
            
        Returns:
            Transcription result or None
        """
        try:
            # Use existing STT engine but with immediate processing
            result = await self.stt_engine._transcribe_audio(audio_array)
            return result
        except Exception as e:
            logger.error(f"❌ STT streaming error: {e}")
            return None
    
    async def _stream_llm_to_tts(self, user_text: str) -> AsyncGenerator[Tuple[Tuple[int, np.ndarray], AdditionalOutputs], None]:
        """
        Stream LLM response directly to TTS as tokens arrive.
        
        Args:
            user_text: User input text
            
        Yields:
            Audio chunks from streaming TTS
        """
        sentence_buffer = ""
        
        try:
            # Get streaming LLM response
            async for token in self.voice_assistant.llm_service.stream_response(user_text):
                sentence_buffer += token
                
                # Check for sentence completion
                if self._is_sentence_complete(sentence_buffer):
                    # Stream this sentence to TTS immediately
                    async for audio_chunk in self._stream_sentence_to_tts(sentence_buffer.strip()):
                        yield audio_chunk
                    
                    sentence_buffer = ""
            
            # Process any remaining content
            if sentence_buffer.strip():
                async for audio_chunk in self._stream_sentence_to_tts(sentence_buffer.strip()):
                    yield audio_chunk
                    
        except Exception as e:
            logger.error(f"❌ LLM→TTS streaming error: {e}")
            # Fallback error message
            async for audio_chunk in self._stream_sentence_to_tts("Sorry, I encountered an error."):
                yield audio_chunk
    
    def _is_sentence_complete(self, text: str) -> bool:
        """
        Check if text contains a complete sentence for TTS.
        
        Args:
            text: Text to check
            
        Returns:
            True if sentence is complete
        """
        if not text.strip():
            return False
            
        # Sentence endings
        sentence_endings = ['.', '!', '?', '...']
        
        # Check for clear sentence endings
        for ending in sentence_endings:
            if text.rstrip().endswith(ending):
                return True
        
        # For streaming, also consider natural pauses
        natural_pauses = [', ', ' and ', ' but ', ' so ']
        for pause in natural_pauses:
            if pause in text and len(text.split()) >= 8:  # Long enough phrase
                return True
                
        # Consider length-based completion for very long phrases
        if len(text.split()) >= 15:
            return True
            
        return False
    
    async def _stream_sentence_to_tts(self, sentence: str) -> AsyncGenerator[Tuple[Tuple[int, np.ndarray], AdditionalOutputs], None]:
        """
        Stream a sentence through TTS engine.
        
        Args:
            sentence: Complete sentence to synthesize
            
        Yields:
            Audio chunks from TTS
        """
        try:
            # Get current language and voice
            current_language = self.voice_assistant.current_language
            available_voices = self.voice_mapper.get_voices_for_language(current_language)
            voice_id = available_voices[0] if available_voices else None
            
            logger.debug(f"🔊 TTS streaming: '{sentence}' (lang: {current_language}, voice: {voice_id})")
            
            # Stream TTS synthesis
            async for sample_rate, audio_chunk in self.tts_engine.stream_synthesis_async(
                sentence, voice_id, current_language
            ):
                if isinstance(audio_chunk, np.ndarray) and audio_chunk.size > 0:
                    # Yield in smaller chunks for responsiveness
                    chunk_size = 1024
                    for i in range(0, audio_chunk.size, chunk_size):
                        mini_chunk = audio_chunk[i:i+chunk_size]
                        if mini_chunk.size > 0:
                            yield (sample_rate, mini_chunk), AdditionalOutputs()
                            
        except Exception as e:
            logger.error(f"❌ TTS sentence streaming error: {e}")
            yield EMPTY_AUDIO_YIELD_OUTPUT


class StreamingCallbackHandler:
    """
    FastRTC callback handler optimized for streaming STT→LLM→TTS pipeline.
    Replaces the original sync callback handler with async streaming.
    """
    
    def __init__(self, voice_assistant, stt_engine: STTEngine, tts_engine: KokoroTTSEngine, 
                 voice_mapper: VoiceMapper, event_loop=None):
        """
        Initialize streaming callback handler.
        
        Args:
            voice_assistant: Main voice assistant instance
            stt_engine: Speech-to-text engine
            tts_engine: Text-to-speech engine  
            voice_mapper: Voice mapping component
            event_loop: Async event loop
        """
        self.voice_assistant = voice_assistant
        self.stt_engine = stt_engine
        self.tts_engine = tts_engine
        self.voice_mapper = voice_mapper
        self.event_loop = event_loop
        
        # Create streaming pipeline
        self.pipeline = StreamingPipeline(
            voice_assistant, stt_engine, tts_engine, voice_mapper
        )
        
        # Performance tracking
        self.total_requests = 0
        self.stream_start_time = None
    
    def process_audio_stream(self, audio_data_tuple: tuple):
        """
        Main FastRTC callback - processes audio through streaming pipeline.
        
        Args:
            audio_data_tuple: Tuple containing audio data from FastRTC
            
        Yields:
            Audio chunks for streaming back to client
        """
        self.total_requests += 1
        request_start = time.time()
        
        try:
            # Parse audio data
            sample_rate, raw_audio_array = self._parse_audio_data(audio_data_tuple)
            
            if raw_audio_array is None or raw_audio_array.size == 0:
                yield EMPTY_AUDIO_YIELD_OUTPUT
                return
            
            # Preprocess audio 
            audio_array = self._preprocess_audio(raw_audio_array, sample_rate)
            
            if audio_array.size == 0:
                yield EMPTY_AUDIO_YIELD_OUTPUT
                return
            
            # Process through async streaming pipeline
            async_generator = self.pipeline.process_audio_stream(audio_array, sample_rate)
            
            # Run async generator in sync context for FastRTC
            if self.event_loop and not self.event_loop.is_closed():
                try:
                    # Create task to run the async generator
                    task = asyncio.run_coroutine_threadsafe(
                        self._async_generator_to_sync(async_generator), 
                        self.event_loop
                    )
                    
                    # Get results with timeout
                    results = task.result(timeout=30.0)
                    
                    # Yield all results
                    for result in results:
                        yield result
                        
                except asyncio.TimeoutError:
                    logger.error("❌ Streaming pipeline timeout")
                    yield EMPTY_AUDIO_YIELD_OUTPUT
                except Exception as e:
                    logger.error(f"❌ Event loop error: {e}")
                    yield EMPTY_AUDIO_YIELD_OUTPUT
            else:
                logger.warning("❌ No valid event loop for async processing")
                yield EMPTY_AUDIO_YIELD_OUTPUT
                
        except Exception as e:
            logger.error(f"❌ Streaming callback error: {e}")
            yield EMPTY_AUDIO_YIELD_OUTPUT
        finally:
            # Track performance
            processing_time = time.time() - request_start
            if processing_time > 1.0:  # Log slow requests
                logger.warning(f"⚠️ Slow streaming request: {processing_time:.3f}s")
    
    async def _async_generator_to_sync(self, async_gen):
        """
        Convert async generator to list for sync context.
        
        Args:
            async_gen: Async generator to convert
            
        Returns:
            List of yielded values
        """
        results = []
        async for item in async_gen:
            results.append(item)
        return results
    
    def _parse_audio_data(self, audio_data_tuple: tuple) -> Tuple[int, Optional[np.ndarray]]:
        """
        Parse and validate audio data from FastRTC.
        
        Args:
            audio_data_tuple: Raw audio data tuple
            
        Returns:
            Tuple of (sample_rate, audio_array)
        """
        try:
            if not isinstance(audio_data_tuple, tuple) or len(audio_data_tuple) != 2:
                logger.warning(f"⚠️ Invalid audio data format: {type(audio_data_tuple)}")
                return 16000, None
                
            sample_rate, raw_audio_array = audio_data_tuple
            
            # Validate sample rate
            if not isinstance(sample_rate, (int, float)) or sample_rate <= 0:
                logger.warning(f"⚠️ Invalid sample rate: {sample_rate}")
                sample_rate = 16000
            
            # Validate audio array
            if not isinstance(raw_audio_array, np.ndarray):
                try:
                    raw_audio_array = np.array(raw_audio_array, dtype=np.float32)
                except:
                    logger.warning("⚠️ Could not convert audio to numpy array")
                    return sample_rate, None
            
            return int(sample_rate), raw_audio_array
            
        except Exception as e:
            logger.error(f"❌ Audio parsing error: {e}")
            return 16000, None
    
    def _preprocess_audio(self, audio_array: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Preprocess audio for optimal STT performance.
        
        Args:
            audio_array: Raw audio array
            sample_rate: Audio sample rate
            
        Returns:
            Preprocessed audio array
        """
        try:
            # Handle multi-dimensional arrays
            if len(audio_array.shape) > 1:
                if audio_array.shape[0] == 1:
                    audio_array = audio_array[0]
                elif audio_array.shape[1] == 1:
                    audio_array = audio_array[:, 0]
                else:
                    audio_array = np.mean(audio_array, axis=1)  # Convert to mono
            
            # Ensure float32 format
            if audio_array.dtype == np.int16:
                audio_array = audio_array.astype(np.float32) / 32768.0
            elif audio_array.dtype != np.float32:
                audio_array = audio_array.astype(np.float32)
            
            # Resample to target sample rate if needed
            TARGET_SAMPLE_RATE = 16000
            if sample_rate != TARGET_SAMPLE_RATE and audio_array.size > 0:
                from scipy.signal import resample
                num_samples = int(len(audio_array) * TARGET_SAMPLE_RATE / sample_rate)
                audio_array = resample(audio_array, num_samples)
                logger.debug(f"🔄 Resampled {sample_rate}Hz → {TARGET_SAMPLE_RATE}Hz")
            
            return audio_array
            
        except Exception as e:
            logger.error(f"❌ Audio preprocessing error: {e}")
            return np.array([], dtype=np.float32)
    
    def get_stats(self) -> dict:
        """
        Get streaming callback handler statistics.
        
        Returns:
            Dictionary with performance stats
        """
        return {
            'total_requests': self.total_requests,
            'handler_type': 'streaming',
            'pipeline_active': self.pipeline is not None,
            'event_loop_running': (
                self.event_loop and 
                not self.event_loop.is_closed() and 
                self.event_loop.is_running()
            )
        }