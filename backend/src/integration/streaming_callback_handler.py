"""
Streaming Callback Handler Module

High-performance async streaming pipeline for STT→LLM→TTS.
Optimized for immediate response and fluid conversation UX.
"""

import asyncio
import os
import time
import numpy as np
from typing import AsyncGenerator, Tuple, Any, Optional, TYPE_CHECKING
from fastrtc import AdditionalOutputs

if TYPE_CHECKING:
    from typing import AsyncGenerator

from ..core.interfaces import TranscriptionResult
from ..audio import STTEngine, KokoroTTSEngine, VoiceMapper
from ..utils.logging import get_logger, time_streaming, create_streaming_timer
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
        import time
        start_time = time.time()
        
        try:
            # Step 1: STT Processing
            if os.getenv("DEBUG_STREAMING", "false").lower() == "true":
                logger.debug(f"🎯 Step 1: Starting STT processing (audio size: {audio_array.size} samples)")
            
            transcript_result = await self._stream_stt(audio_array)
            
            if not transcript_result or not transcript_result.text.strip():
                if os.getenv("DEBUG_STREAMING", "false").lower() == "true":
                    logger.debug("❌ Step 1: No transcript result - yielding empty")
                yield EMPTY_AUDIO_YIELD_OUTPUT
                return
                
            user_text = transcript_result.text.strip()
            
            # Step 2: Confidence and Quality Checks  
            if os.getenv("DEBUG_STREAMING", "false").lower() == "true":
                confidence = getattr(transcript_result, 'confidence', 'unknown')
                logger.debug(f"🔍 Step 2: Quality check - Text: '{user_text}' (confidence: {confidence}, words: {len(user_text.split())})")
            
            # Check confidence before proceeding
            if hasattr(transcript_result, 'confidence') and transcript_result.confidence < self.confidence_threshold:
                # Store partial but don't process yet
                self.partial_transcript = user_text
                if os.getenv("DEBUG_STREAMING", "false").lower() == "true":
                    logger.debug(f"⚠️ Step 2: Low confidence ({transcript_result.confidence}) - storing partial")
                yield EMPTY_AUDIO_YIELD_OUTPUT
                return
                
            # Check minimum words threshold
            if len(user_text.split()) < self.min_words_for_processing:
                self.partial_transcript = user_text
                if os.getenv("DEBUG_STREAMING", "false").lower() == "true":
                    logger.debug(f"⚠️ Step 2: Insufficient words ({len(user_text.split())}) - storing partial")
                yield EMPTY_AUDIO_YIELD_OUTPUT
                return
                
            # Step 3: Start LLM→TTS Pipeline
            if os.getenv("DEBUG_STREAMING", "false").lower() == "true":
                logger.debug(f"✅ Step 3: Starting LLM→TTS pipeline for: '{user_text}'")
            else:
                logger.info(f"🎤 USER: '{user_text}'")
            
            # Stream LLM→TTS pipeline
            async for audio_chunk in self._stream_llm_to_tts(user_text):
                yield audio_chunk
            
            # Log total processing time
            total_time = time.time() - start_time
            logger.debug(f"⏱️ Full Audio Stream Processing: {total_time:.3f}s")
                
        except Exception as e:
            logger.error(f"❌ Streaming pipeline error: {e}")
            yield EMPTY_AUDIO_YIELD_OUTPUT
    
    @time_streaming("STT Processing")
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
        import time
        start_time = time.time()
        sentence_buffer = ""
        token_count = 0
        sentence_count = 0
        
        try:
            if os.getenv("DEBUG_STREAMING", "false").lower() == "true":
                logger.debug(f"🧠 Step 4: Starting LLM streaming for: '{user_text}'")
            
            # Check for authentication/user identification
            logger.info(f"🔍 [STREAMING] Starting authentication check for text: '{user_text[:100]}...'")
            
            try:
                auth_result = self.voice_assistant.voice_print_manager.process_text(user_text)
                logger.info(f"🔍 [STREAMING] Authentication result: {auth_result}")
            except Exception as auth_error:
                logger.error(f"❌ [STREAMING] Authentication error: {auth_error}")
                import traceback
                logger.error(f"❌ [STREAMING] Auth traceback: {traceback.format_exc()}")
                auth_result = None
            
            if auth_result:
                action = auth_result.get('action')
                confirmation_message = None
                
                if action == 'user_identified':
                    # Successful login
                    identified_user_id = auth_result['user_id']
                    if identified_user_id != self.voice_assistant.user_id:
                        logger.info(f"🔄 [STREAMING] User authenticated: switching from '{self.voice_assistant.user_id}' to '{identified_user_id}'")
                        
                        # Extract username for display
                        username = identified_user_id.replace("user_", "")
                        
                        # Switch memory manager to new user
                        if self.voice_assistant.memory_manager.switch_user(identified_user_id):
                            logger.info(f"✅ [STREAMING] Memory manager switched to user: {identified_user_id}")
                            
                            # Refresh session with new authenticated user
                            self.voice_assistant.refresh_session_after_auth(identified_user_id, username)
                            
                            confirmation_message = f"Welcome back, {username}! I've loaded your personal memory profile and started a fresh session."
                        else:
                            logger.error(f"❌ [STREAMING] Failed to switch memory manager to user: {identified_user_id}")
                            confirmation_message = "I recognized you, but there was an issue accessing your personal profile."
                
                elif action == 'pin_request':
                    # User exists, need PIN
                    username = auth_result['username']
                    confirmation_message = f"Hello {username}! Please provide your 4-digit PIN to access your profile."
                    
                elif action == 'registration_success':
                    # New user registered and logged in
                    identified_user_id = auth_result['user_id']
                    username = identified_user_id.replace("user_", "")
                    
                    if self.voice_assistant.memory_manager.switch_user(identified_user_id):
                        logger.info(f"✅ [STREAMING] New user registered: {identified_user_id}")
                        
                        # Refresh session with new registered user
                        self.voice_assistant.refresh_session_after_auth(identified_user_id, username)
                        
                        confirmation_message = f"Welcome {username}! Your account has been created and I'll remember our conversations in your new session."
                    else:
                        logger.error(f"❌ [STREAMING] Failed to switch memory manager for new user: {identified_user_id}")
                        confirmation_message = "Your account was created, but there was an issue setting up your memory profile."
                
                elif action == 'login_cancelled':
                    # User cancelled login
                    username = auth_result.get('username', 'someone')
                    confirmation_message = f"No problem, {username}. I'll continue with the temporary session."
                
                elif action == 'suggest_registration':
                    # User identified but not registered
                    username = auth_result.get('username', 'unknown')
                    confirmation_message = auth_result.get('message', f"I don't know you yet, {username}. Would you like to register? Say 'register as {username} PIN 1234' with your chosen 4-digit PIN.")
                
                elif action == 'auth_failed':
                    # Authentication failed
                    reason = auth_result.get('reason', 'unknown')
                    if reason == 'invalid_pin':
                        confirmation_message = "Sorry, that PIN is incorrect. Please try again or say 'cancel' to stop."
                    elif reason == 'user_not_found':
                        username = auth_result.get('username', 'unknown')
                        confirmation_message = f"I don't have a user named '{username}'. Would you like to register? Say 'register as {username} PIN 1234' with your chosen 4-digit PIN."
                    elif reason == 'invalid_username':
                        confirmation_message = "That username isn't valid. Please choose a different name."
                    else:
                        confirmation_message = "Authentication failed. Please try again."
                
                # Stream the confirmation message if we have one
                if confirmation_message:
                    current_language = self.voice_assistant.current_language
                    available_voices = self.voice_mapper.get_voices_for_language(current_language)
                    voice_id = available_voices[0] if available_voices else None
                    
                    async for sample_rate, audio_chunk in self.tts_engine.stream_synthesis_async(
                        confirmation_message, voice_id, current_language
                    ):
                        if isinstance(audio_chunk, np.ndarray) and audio_chunk.size > 0:
                            yield (sample_rate, audio_chunk), AdditionalOutputs()
                    return
            
            # Get streaming LLM response
            async for token in self.voice_assistant.llm_service.stream_response(user_text):
                sentence_buffer += token
                token_count += 1
                
                # Debug token accumulation
                if os.getenv("DEBUG_STREAMING", "false").lower() == "true" and token_count % 10 == 0:
                    logger.debug(f"🔤 Token progress: {token_count} tokens, buffer: '{sentence_buffer[-50:]}...'")
                
                # Check for sentence completion
                if self._is_sentence_complete(sentence_buffer):
                    sentence_count += 1
                    sentence_to_synthesize = sentence_buffer.strip()
                    
                    if os.getenv("DEBUG_STREAMING", "false").lower() == "true":
                        logger.debug(f"📝 Step 5: Sentence #{sentence_count} complete: '{sentence_to_synthesize}' (after {token_count} tokens)")
                    
                    # Stream this sentence to TTS immediately
                    async for audio_chunk in self._stream_sentence_to_tts(sentence_to_synthesize):
                        yield audio_chunk
                    
                    sentence_buffer = ""
            
            # Process any remaining content
            if sentence_buffer.strip():
                sentence_count += 1
                remaining_text = sentence_buffer.strip()
                
                if os.getenv("DEBUG_STREAMING", "false").lower() == "true":
                    logger.debug(f"🔚 Step 6: Final sentence #{sentence_count}: '{remaining_text}' (total tokens: {token_count})")
                
                async for audio_chunk in self._stream_sentence_to_tts(remaining_text):
                    yield audio_chunk
            
            # Log total processing time
            total_time = time.time() - start_time
            logger.debug(f"⏱️ LLM to TTS Pipeline: {total_time:.3f}s")
                    
        except Exception as e:
            import traceback
            logger.error(f"❌ LLM→TTS streaming error: {e}")
            logger.error(f"❌ Full traceback: {traceback.format_exc()}")
            # Fallback error message
            async for audio_chunk in self._stream_sentence_to_tts("I encountered an error processing your request."):
                yield audio_chunk
    
    def _is_sentence_complete(self, text: str) -> bool:
        """
        Check if text contains a complete sentence for TTS.
        Uses improved logic to prevent word-breaking issues.
        
        Args:
            text: Text to check
            
        Returns:
            True if sentence is complete
        """
        if not text.strip():
            return False
            
        text_stripped = text.rstrip()
        words = text.split()
        
        # Sentence endings - highest priority
        sentence_endings = ['.', '!', '?', '...']
        for ending in sentence_endings:
            if text_stripped.endswith(ending):
                if os.getenv("DEBUG_STREAMING", "false").lower() == "true":
                    logger.debug(f"✂️ Sentence split: Found ending '{ending}' in: '{text_stripped}'")
                return True
        
        # For streaming, consider natural pauses but only at word boundaries
        # and with more conservative thresholds to prevent word-breaking
        if len(words) >= 10:  # Increased threshold from 8 to 10
            natural_pauses = [
                ', and ', ', but ', ', so ', ', however ', ', therefore ',
                ', because ', ', although ', ', while ', ', when ', ', if '
            ]
            for pause in natural_pauses:
                if pause in text and self._ends_at_word_boundary(text, pause):
                    if os.getenv("DEBUG_STREAMING", "false").lower() == "true":
                        logger.debug(f"✂️ Sentence split: Found natural pause '{pause}' at word boundary in: '{text}'")
                    return True
        
        # More conservative length-based completion
        # Only trigger for very long phrases and ensure we're at a word boundary
        if len(words) >= 20:  # Increased threshold from 15 to 20
            # Check if we can find a good breaking point
            if self._find_safe_break_point(text):
                if os.getenv("DEBUG_STREAMING", "false").lower() == "true":
                    logger.debug(f"✂️ Sentence split: Length-based break at {len(words)} words: '{text}'")
                return True
        
        return False
    
    def _ends_at_word_boundary(self, text: str, pause: str) -> bool:
        """
        Check if text after a pause ends at a safe word boundary.
        
        Args:
            text: Full text to check
            pause: Pause pattern to look for
            
        Returns:
            True if safe to break after the pause
        """
        pause_index = text.rfind(pause)
        if pause_index == -1:
            return False
        
        # Get text after the pause
        after_pause = text[pause_index + len(pause):].strip()
        
        # Don't break if there's only a partial word after the pause
        if not after_pause or len(after_pause.split()) < 3:
            return False
        
        # Don't break if the last word looks incomplete (no vowel or very short)
        last_word = after_pause.split()[-1].lower()
        if len(last_word) < 3 or not any(c in last_word for c in 'aeiou'):
            return False
        
        return True
    
    def _find_safe_break_point(self, text: str) -> bool:
        """
        Find a safe point to break long text without cutting words.
        
        Args:
            text: Text to find break point in
            
        Returns:
            True if a safe break point exists
        """
        words = text.split()
        if len(words) < 15:
            return False
        
        # Look for safe breaking points in the latter half of the text
        start_search = len(words) // 2
        
        for i in range(start_search, len(words) - 2):  # Leave at least 2 words after break
            word = words[i].lower()
            
            # Safe break points: complete words that commonly end clauses
            safe_endings = ['that', 'which', 'where', 'when', 'because', 'since', 'while']
            if word.rstrip('.,;:') in safe_endings:
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
        import time
        start_time = time.time()
        
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
                    # Yield in optimized chunks for smooth playback without artifacts
                    chunk_size = 2048  # Increased from 1024 to 2048 for better audio quality
                    for i in range(0, audio_chunk.size, chunk_size):
                        mini_chunk = audio_chunk[i:i+chunk_size]
                        if mini_chunk.size > 0:
                            yield (sample_rate, mini_chunk), AdditionalOutputs()
            
            # Log total processing time
            total_time = time.time() - start_time
            logger.debug(f"⏱️ Sentence to TTS: {total_time:.3f}s")
                            
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