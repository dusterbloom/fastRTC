"""
Stream Callback Handler Module

Handles real-time audio stream processing for the FastRTC voice assistant.
This is the core component that processes incoming audio, performs STT,
generates responses, and streams TTS audio back to the client.
"""

import time
import numpy as np
from datetime import datetime, timezone
from typing import Tuple, Generator, Any, Optional
from fastrtc import AdditionalOutputs
import threading

from ..core.interfaces import TranscriptionResult # Added import
from ..audio import STTEngine, KokoroTTSEngine, VoiceMapper
from ..audio.engines.tts.kokoro_tts import KokoroTTSOptions
from ..utils.async_utils import run_coro_from_sync_thread_with_timeout
from ..utils.logging import get_logger
from ..utils.sota_adaptive_vad import SimpleSOTAAdaptiveVAD, VADConfig # Added import
from ..config.language_config import LANGUAGE_NAMES, LANGUAGE_ABBREVIATIONS, KOKORO_TTS_LANG_MAP
from ..config.audio_config import AUDIO_SAMPLE_RATE, SILENT_AUDIO_FRAME_TUPLE

logger = get_logger(__name__)

# Audio constants
EMPTY_AUDIO_YIELD_OUTPUT = (SILENT_AUDIO_FRAME_TUPLE, AdditionalOutputs())


class StreamCallbackHandler:
    """
    Handles real-time audio stream processing for voice assistant interactions.
    
    This class processes incoming audio streams, performs speech-to-text conversion,
    generates intelligent responses, and streams text-to-speech audio back to the client.
    """
    
    def __init__(
        self,
        voice_assistant,
        stt_engine: STTEngine,
        tts_engine: KokoroTTSEngine,
        voice_mapper: VoiceMapper,
        event_loop=None,
    ):
        """
        Initialize the stream callback handler.
        
        Args:
            voice_assistant: The main voice assistant instance
            stt_engine: Speech-to-text engine
            tts_engine: Text-to-speech engine
            voice_mapper: Voice mapping component
            event_loop: Async event loop for coroutine execution
        """
        self.voice_assistant = voice_assistant
        self.stt_engine = stt_engine
        self.tts_engine = tts_engine
        self.voice_mapper = voice_mapper
        self.event_loop = event_loop
        
        # Sentence buffering for complete sentence detection
        self.sentence_buffer = ""
        self.buffer_timestamp = None
        self.buffer_timeout = 3.0  # seconds
        self.buffer_lock = threading.Lock()
        self.max_buffer_length = 500  # characters
        
        # Language names mapping
        self.lang_names = LANGUAGE_NAMES
        self.lang_abbreviations = LANGUAGE_ABBREVIATIONS

        # --- DEBUG: Log available voices from TTS model, if possible ---
        try:
            tts_model = getattr(self.tts_engine, "tts_model", None)
            available_voices = []
            if tts_model is not None:
                # Try to get voices from tts_model.model.voices or tts_model.voices
                if hasattr(tts_model, "model") and hasattr(tts_model.model, "voices"):
                    available_voices = list(getattr(tts_model.model, "voices", []))
                elif hasattr(tts_model, "voices"):
                    available_voices = list(getattr(tts_model, "voices", []))
        except Exception as e:
            
            logger.error(f"Error retrieving TTS model voices: {e}")
        
    def process_audio_stream(self, audio_data_tuple: tuple) -> Generator[Tuple[Tuple[int, np.ndarray], AdditionalOutputs], None, None]:
        """
        Main callback function for processing audio streams in real-time.
        
        This function:
        1. Processes incoming audio data
        2. Performs speech-to-text conversion
        3. Detects the spoken language
        4. Generates intelligent responses
        5. Converts responses to speech
        6. Streams audio back to the client
        
        Args:
            audio_data_tuple: Tuple containing audio data from FastRTC
            
        Yields:
            Tuples of (audio_data, additional_outputs) for streaming back to client
       """
        # Interrupt ongoing TTS when user starts speaking
        self.tts_engine.interrupt()
        
        # DEBUG: Log every callback invocation with enhanced details
        current_time = time.time()
        logger.debug(f"🎤 CALLBACK INVOKED [{current_time:.3f}]: Received audio data: {type(audio_data_tuple)}")
        try:
            sample_rate, audio_array = audio_data_tuple
            audio_duration = len(audio_array) / sample_rate if sample_rate > 0 else 0
            audio_rms = np.sqrt(np.mean(audio_array**2)) if hasattr(audio_array, 'shape') and audio_array.size > 0 else 0
            logger.debug(f"🎤 CALLBACK: Sample rate: {sample_rate}Hz, Audio shape: {audio_array.shape if hasattr(audio_array, 'shape') else type(audio_array)}")
            logger.debug(f"🎤 CALLBACK: Duration: {audio_duration:.3f}s, RMS: {audio_rms:.6f}, Peak: {np.max(np.abs(audio_array)) if hasattr(audio_array, 'shape') and audio_array.size > 0 else 0:.6f}")
        except Exception as e:
            logger.error(f"🎤 CALLBACK ERROR: Failed to parse audio data: {e}")
            pass
        if not self.voice_assistant:
            logger.warning("🎤 process_audio_stream: Voice assistant not initialized. Yielding empty.")
            yield EMPTY_AUDIO_YIELD_OUTPUT
            return
        
        try:
            logger.info(f"🎤 process_audio_stream: Processing incoming audio data...")
            # --- COPY FROM start_original_backup.py ---
            # Process audio input (same as before)
            if isinstance(audio_data_tuple, tuple) and len(audio_data_tuple) == 2:
                sample_rate, raw_audio_array = audio_data_tuple

                # Audio preprocessing (keep your existing code)
                if isinstance(raw_audio_array, np.ndarray) and len(raw_audio_array.shape) > 1:
                    if raw_audio_array.shape[0] == 1:
                        audio_array = raw_audio_array[0]
                    elif raw_audio_array.shape[1] == 1:
                        audio_array = raw_audio_array[:, 0]
                    elif raw_audio_array.shape[1] > 1:
                        audio_array = np.mean(raw_audio_array, axis=1)
                    else:
                        audio_array = raw_audio_array.flatten()
                else:
                    audio_array = raw_audio_array if isinstance(raw_audio_array, np.ndarray) else np.array(raw_audio_array, dtype=np.float32)

                if isinstance(audio_array, np.ndarray):
                    if audio_array.dtype == np.int16:
                        audio_array = audio_array.astype(np.float32) / 32768.0
                    else:
                        audio_array = audio_array.astype(np.float32)


                # --- Resample to 16kHz mono if needed ---
                TARGET_SAMPLE_RATE = 16000
                if sample_rate != TARGET_SAMPLE_RATE and audio_array.size > 0:
                    from scipy.signal import resample
                    num_samples = int(len(audio_array) * TARGET_SAMPLE_RATE / sample_rate)
                    audio_array = resample(audio_array, num_samples)
                    
                    sample_rate = TARGET_SAMPLE_RATE
            else:
                audio_array = np.array([], dtype=np.float32)
                sample_rate = 16000

            # (Removed verbose debug prints for cleaner terminal output)
            
            if not isinstance(audio_array, np.ndarray) or audio_array.size == 0: # Modified condition
                # (Removed verbose debug prints for cleaner terminal output)
                yield SILENT_AUDIO_FRAME_TUPLE, AdditionalOutputs()
                return
            
          
            # Perform speech-to-text conversion
            user_text = self._process_speech_to_text(audio_array, sample_rate)
            
            # Check for buffered content even if current user_text is empty
            if not user_text.strip():
                # Check if we have buffered content that should be processed due to timeout
                buffered_content = self._check_buffer_timeout()
                if buffered_content.strip():
                    user_text = buffered_content
                else:
                    yield SILENT_AUDIO_FRAME_TUPLE, AdditionalOutputs()
                    return
            
            # Update statistics
            self.voice_assistant.voice_detection_successes += 1
            
            # Generate intelligent response
            assistant_response = self._generate_response(user_text)
            
            # Update conversation tracking
            self._update_conversation(user_text, assistant_response)
            
            # Convert response to speech and stream back
            yield from self._stream_tts_response(assistant_response)
            
        except Exception as e:
            logger.error(f"❌ CRITICAL Error in stream processing: {e}")
            import traceback
            traceback.print_exc()
            logger.error("🎤 process_audio_stream: Exception caught, attempting error recovery.") # New log
            
            # Graceful error recovery
            yield from self._handle_error_recovery()
    
    def _process_speech_to_text(self, audio_array: np.ndarray, sample_rate: int) -> str:
        """
        Process audio array to extract text.
        
        Args:
            audio_array: Audio data as numpy array
            sample_rate: Audio sample rate
            
        Returns:
            Transcribed text
        """
        # Perform transcription using the method that accepts audio_array and sample_rate
        # and is designed to handle Hugging Face pipeline parameters.
        # No need for audio_bytes conversion here as transcribe_with_sample_rate takes array.
        try:
            # Get UI-selected language and convert to Whisper format for STT
            from src.config.language_config import get_whisper_language
            ui_language_kokoro = self.voice_assistant.current_language
            ui_language_whisper = get_whisper_language(ui_language_kokoro)
            
            print(f"[STT] Using UI-selected language: Kokoro='{ui_language_kokoro}' -> Whisper='{ui_language_whisper}'")
            
            transcription_result = run_coro_from_sync_thread_with_timeout(
                self.stt_engine._transcribe_audio(audio_array, ui_language_whisper),
                timeout=8.0, # Consistent with start_original_backup.py
                event_loop=self.event_loop
            )
        except TimeoutError:
            logger.warning("STT transcription timed out.")
            transcription_result = None # Or some default TranscriptionResult
        except Exception as e:
            logger.error(f"STT transcription failed: {e}")
            transcription_result = None # Or some default TranscriptionResult

        # outputs is now a TranscriptionResult object
        current_text = transcription_result.text.strip() if transcription_result and hasattr(transcription_result, 'text') else ""
        print(f"🗣️  [STT DEBUG] Transcription result: {transcription_result}")
        print(f"🗣️  [STT DEBUG] Has text attribute: {hasattr(transcription_result, 'text') if transcription_result else False}")
        print(f"🗣️  [STT DEBUG] Raw text: '{transcription_result.text if transcription_result and hasattr(transcription_result, 'text') else 'NO TEXT'}'")
        print(f"🗣️  [STT DEBUG] Current text (stripped): '{current_text}' (length: {len(current_text)})")
        
        # Thread-safe buffer management
        with self.buffer_lock:
            # Add current text to sentence buffer
            if current_text:
                # Check buffer size limit
                if len(self.sentence_buffer) + len(current_text) > self.max_buffer_length:
                    logger.warning(f"Buffer size limit reached, clearing buffer (was {len(self.sentence_buffer)} chars)")
                    self.sentence_buffer = current_text  # Start fresh with current text
                else:
                    self.sentence_buffer += (" " + current_text) if self.sentence_buffer else current_text
                
                # Update timestamp when new content is added
                self.buffer_timestamp = time.time()
            
            # Extract complete sentences from buffer
            complete_sentences, remaining_fragment = self._extract_complete_sentences(self.sentence_buffer)
            
            # Update buffer with remaining fragment
            self.sentence_buffer = remaining_fragment
            
            # Clear timestamp if buffer is empty
            if not self.sentence_buffer:
                self.buffer_timestamp = None
        
        # Return complete sentences, or handle timeout case
        user_text = complete_sentences
        
        print(f"[STT DEBUG] Current chunk: '{current_text}'")
        print(f"[STT DEBUG] Buffer state: '{self.sentence_buffer}' (length: {len(self.sentence_buffer)})")
        print(f"[STT DEBUG] Buffer timestamp: {self.buffer_timestamp}")
        print(f"[STT DEBUG] Complete sentences: '{complete_sentences}'")
        print(f"[STT DEBUG] Final user_text: '{user_text}' (length: {len(user_text)})")
        if self.sentence_buffer and self.buffer_timestamp:
            buffer_age = time.time() - self.buffer_timestamp
            print(f"[STT DEBUG] Buffer age: {buffer_age:.2f}s (timeout: {self.buffer_timeout}s)")
        # (Removed verbose debug prints for cleaner terminal output)
        
        return user_text
    
    def _extract_complete_sentences(self, text: str) -> tuple[str, str]:
        """
        Extract complete sentences from text, maintaining incomplete fragments.
        Enhanced to handle conversational patterns and edge cases.
        
        Args:
            text: Input text that may contain complete and incomplete sentences
            
        Returns:
            Tuple of (complete_sentences, remaining_fragment)
        """
        if not text.strip():
            return "", ""
        
        # Enhanced sentence ending patterns for conversational speech
        sentence_endings = ['.', '!', '?', '...', ', right?', ', you know?', ', okay?']
        
        # Conversational completeness indicators
        conversational_endings = [
            ', right', ', okay', ', you know', ', yeah', ', sure', 
            ' though', ' then', ' so', ' well', ' actually'
        ]
        
        # Find the last occurrence of any strong sentence ending
        last_ending_pos = -1
        ending_found = None
        for ending in sentence_endings:
            pos = text.rfind(ending)
            if pos > last_ending_pos:
                last_ending_pos = pos
                ending_found = ending
        
        # If no strong ending found, check for conversational patterns
        if last_ending_pos == -1:
            # Check for conversational completeness indicators
            for ending in conversational_endings:
                if text.lower().endswith(ending.lower()):
                    # Treat as complete if it seems like a conversational turn
                    words = text.split()
                    if len(words) >= 3:  # Minimum words for complete thought
                        return text.strip(), ""
            
            # No clear ending, but check if text seems long enough to be complete
            words = text.split()
            if len(words) >= 8:  # Long enough phrase might be complete
                # Look for natural break points (commas, "and", "but", etc.)
                break_words = ['and', 'but', 'so', 'then', 'because', 'since', 'while']
                for i, word in enumerate(reversed(words[-4:])):
                    if word.lower() in break_words and len(words) - i >= 4:
                        # Found a natural break point near the end
                        break_point = len(words) - i
                        complete_part = ' '.join(words[:break_point]).strip()
                        remaining_part = ' '.join(words[break_point:]).strip()
                        return complete_part, remaining_part
            
            # Still no clear completion, treat as incomplete fragment
            return "", text.strip()
        
        # Found a strong sentence ending
        complete_part = text[:last_ending_pos + len(ending_found)].strip()
        remaining_part = text[last_ending_pos + len(ending_found):].strip()
        
        return complete_part, remaining_part
    
    def _check_buffer_timeout(self) -> str:
        """
        Check if buffered content should be processed due to timeout.
        
        Returns:
            Buffered content if timeout occurred, empty string otherwise
        """
        with self.buffer_lock:
            if not self.sentence_buffer or not self.buffer_timestamp:
                return ""
            
            # Check if buffer has timed out
            if time.time() - self.buffer_timestamp > self.buffer_timeout:
                logger.info(f"[STT TIMEOUT] Processing buffered content due to timeout: '{self.sentence_buffer}'")
                content = self.sentence_buffer
                self.sentence_buffer = ""
                self.buffer_timestamp = None
                return content
            
            return ""
    
    def clear_sentence_buffer(self):
        """
        Manually clear the sentence buffer and reset timestamp.
        Useful for resetting state between conversations.
        """
        with self.buffer_lock:
            if self.sentence_buffer:
                logger.info(f"[STT BUFFER] Manually clearing buffer: '{self.sentence_buffer}'")
            self.sentence_buffer = ""
            self.buffer_timestamp = None
    
    def get_buffer_status(self) -> dict:
        """
        Get current buffer status for debugging/monitoring.
        
        Returns:
            Dictionary with buffer information
        """
        with self.buffer_lock:
            return {
                'buffer_content': self.sentence_buffer,
                'buffer_length': len(self.sentence_buffer),
                'buffer_timestamp': self.buffer_timestamp,
                'buffer_age': time.time() - self.buffer_timestamp if self.buffer_timestamp else None,
                'timeout_seconds': self.buffer_timeout,
                'max_length': self.max_buffer_length
            }
    
    def _get_kokoro_language(self, whisper_lang: str) -> str:
        """
        Convert Whisper language code to Kokoro language code.
        
        Args:
            whisper_lang: Language code from Whisper
            
        Returns:
            Kokoro-compatible language code
        """
        # Map common Whisper language codes to our system
        whisper_to_kokoro = {
            'en': 'a',  # American English
            'it': 'i',  # Italian
            'es': 'e',  # Spanish
            'fr': 'f',  # French
            'pt': 'p',  # Portuguese
            'ja': 'j',  # Japanese
            'zh': 'z',  # Chinese
            'hi': 'h',  # Hindi
        }
        return whisper_to_kokoro.get(whisper_lang, 'a')  # Default to English
    
    
    def _generate_response(self, user_text: str) -> str:
        """
        Generate an intelligent response to the user's input.
        
        Args:
            user_text: The user's transcribed speech
            
        Returns:
            Generated response text
        """
        start_turn_time = time.monotonic()
        
        try:
            # Use async LLM service with timeout protection
            assistant_response_text = run_coro_from_sync_thread_with_timeout(
                self.voice_assistant.get_llm_response_smart(user_text),
                timeout=10.0,
                event_loop=self.event_loop
            )
        except TimeoutError:
            assistant_response_text = "Let me think about that and get back to you quickly."
        except Exception as e:
            logger.error(f"❌ Error generating response: {e}")
            assistant_response_text = "I encountered an issue. Could you try that again?"
        
        turn_processing_time = time.monotonic() - start_turn_time
        # (Removed verbose debug prints for cleaner terminal output)
        print(f"[LLM DEBUG] assistant_response_text: '{assistant_response_text}' (length: {len(assistant_response_text)})")

        return assistant_response_text

    
    def _update_conversation(self, user_text: str, assistant_response: str):
        """
        Update conversation tracking and statistics.
        
        Args:
            user_text: User's input text
            assistant_response: Assistant's response text
        """
        # Update conversation buffer
        self.voice_assistant.conversation_buffer.add_turn(
            user_text=user_text,
            assistant_text=assistant_response,
            language=self.voice_assistant.current_language
            # metadata can be added here if/when needed, e.g., metadata={'source': 'start_clean'}
        )
        
        self.voice_assistant.turn_count += 1
        
        # Display statistics every 5 turns to reduce overhead
        if self.voice_assistant.turn_count % 5 == 0:
            self._display_statistics()
    
    def _display_statistics(self):
        """Display current system statistics."""
        try:
            avg_resp = (np.mean(list(self.voice_assistant.total_response_time)) 
                       if self.voice_assistant.total_response_time else 0)
            
            mem_stats = self.voice_assistant.amem_memory.get_stats()
            audio_stats = self.voice_assistant.audio_processor.get_detection_stats()
            
            audio_q_str = (f" | AudioRMS: {audio_stats['avg_rms']:.3f}" 
                          if audio_stats and audio_stats['calibrated'] else "")
            
            lang_abbr = self.lang_abbreviations.get(
                self.voice_assistant.current_language, 'UNK'
            )
            voice_count = len(self.voice_mapper.get_voices_for_language(
                self.voice_assistant.current_language
            ))
            lang_str = f" | Lang: {lang_abbr}({voice_count}v)"
            
            print(
                f"📊 Turn {self.voice_assistant.turn_count}: "
                f"AvgResp={avg_resp:.2f}s | MemOps={mem_stats['mem_ops']} | "
                f"User='{mem_stats['user_name_cache']}'{audio_q_str}{lang_str}"
            )
            # (Removed verbose debug prints for cleaner terminal output)
        except Exception as e:
            logger.warning(f"⚠️ Error displaying statistics: {e}")
    
    def _stream_tts_response(self, response_text: str) -> Generator[Tuple[Tuple[int, np.ndarray], AdditionalOutputs], None, None]:
        """
        Convert response text to speech and stream audio chunks.
        
        Args:
            response_text: Text to convert to speech
            
        Yields:
            Audio chunks for streaming back to client
        """
        # Reset interruption flag before starting new TTS synthesis
        self.tts_engine.reset_interrupt()
        
        additional_outputs = AdditionalOutputs()

        # --- FIX: Always use Kokoro language code for voice selection and TTS options ---
        from src.config.language_config import WHISPER_TO_KOKORO_LANG, KOKORO_TTS_LANG_MAP

        # Map current_language to Kokoro code if needed (update in-place, as in original backup)
        from src.config.language_config import KOKORO_VOICE_MAP
        cur_lang = self.voice_assistant.current_language
        if isinstance(cur_lang, str) and len(cur_lang) == 1 and cur_lang in KOKORO_VOICE_MAP:
            kokoro_lang_code = cur_lang
        else:
            # Use the same conversion as the original backup
            kokoro_lang_code = self.voice_assistant.convert_to_kokoro_language(cur_lang)
            self.voice_assistant.current_language = kokoro_lang_code

        # LOG: Show current language and mapped Kokoro code
        print(f"[TTS] Preparing TTS for language '{cur_lang}' mapped to Kokoro code '{kokoro_lang_code}'")
        print(f"[TTS] Final current_language after mapping: '{self.voice_assistant.current_language}'")

        tts_voices_to_try = self.voice_mapper.get_voices_for_language(self.voice_assistant.current_language)
        print(f"[TTS] Available voices for language '{self.voice_assistant.current_language}': {tts_voices_to_try}")
        
        # Debug: Show voice mapping configuration
        from src.config.language_config import KOKORO_VOICE_MAP
        print(f"[TTS] Voice map for all languages: {KOKORO_VOICE_MAP}")
        expected_voices = KOKORO_VOICE_MAP.get(self.voice_assistant.current_language, [])
        print(f"[TTS] Expected voices for '{self.voice_assistant.current_language}': {expected_voices}")
        tts_voices_to_try.append(None)  # Fallback to default voice

        tts_success = False
        for voice_id in tts_voices_to_try:
            try:
                # Configure TTS options
                options_params = {"speed": 1.05}
                kokoro_tts_lang = KOKORO_TTS_LANG_MAP.get(
                    self.voice_assistant.current_language, 'a'
                )
                options_params["lang"] = kokoro_tts_lang
                
                if voice_id:
                    options_params["voice"] = voice_id

                print(f"[TTS] Attempting TTS with lang='{kokoro_tts_lang}', voice='{voice_id}', options={options_params}")
                
                tts_options = KokoroTTSOptions(**options_params)
                print(f"🔊 Trying TTS with voice '{voice_id}', lang '{kokoro_tts_lang}'")
                
                # Stream TTS audio using the same logic as the original backup
                chunk_count = 0
                total_samples = 0

                for current_sr, current_chunk_array in self.voice_assistant.stream_tts_synthesis(
                    response_text, voice_id, kokoro_lang_code
                ):
                    if isinstance(current_chunk_array, np.ndarray) and current_chunk_array.size > 0:
                        chunk_count += 1
                        total_samples += current_chunk_array.size
                        chunk_size = min(1024, current_chunk_array.size)
                        for i in range(0, current_chunk_array.size, chunk_size):
                            mini_chunk = current_chunk_array[i:i+chunk_size]
                            if mini_chunk.size > 0:
                                yield (current_sr, mini_chunk), additional_outputs

                tts_success = True
                print(
                    f"✅ TTS stream completed. Voice: {voice_id}, "
                    f"Chunks: {chunk_count}, Samples: {total_samples}"
                )
                print(f"[TTS] TTS stream completed. Voice: {voice_id}, Chunks: {chunk_count}, Samples: {total_samples}")
                
                # Success message with language confirmation
                if self.voice_assistant.current_language != 'a' and voice_id:
                    lang_name = self.lang_names.get(
                        self.voice_assistant.current_language, 'Unknown'
                    )
                break
                
            except Exception as e:
                logger.warning(f"❌ TTS failed with voice '{voice_id}': {e}")
                continue
        
        if not tts_success:
            logger.error("❌ All TTS attempts failed")
            yield SILENT_AUDIO_FRAME_TUPLE, additional_outputs
    
    def _yield_audio_chunks(
        self, 
        sample_rate: int, 
        audio_array: np.ndarray, 
        additional_outputs: AdditionalOutputs,
        chunk_size: int = 1024
    ) -> Generator[Tuple[Tuple[int, np.ndarray], AdditionalOutputs], None, None]:
        """
        Yield audio array in smaller chunks to prevent timeouts.
        
        Args:
            sample_rate: Audio sample rate
            audio_array: Audio data array
            additional_outputs: Additional outputs for FastRTC
            chunk_size: Size of each chunk
            
        Yields:
            Audio chunks with metadata
        """
        chunk_size = min(chunk_size, audio_array.size)
        for i in range(0, audio_array.size, chunk_size):
            mini_chunk = audio_array[i:i+chunk_size]
            if mini_chunk.size > 0:
                yield (sample_rate, mini_chunk), additional_outputs
    
    def _handle_error_recovery(self) -> Generator[Tuple[Tuple[int, np.ndarray], AdditionalOutputs], None, None]:
        """
        Handle error recovery by generating a fallback audio response.
        
        Yields:
            Error recovery audio chunks
        """
        try:
            error_msg = "Sorry, I encountered an error. Please try again."
            tts_options = KokoroTTSOptions(speed=1.0, lang="en-us")
            
            for tts_err_chunk in self.tts_engine.stream_tts_sync(error_msg, tts_options):
                if isinstance(tts_err_chunk, tuple) and len(tts_err_chunk) == 2:
                    sr_err, arr_err = tts_err_chunk
                    if isinstance(arr_err, np.ndarray) and arr_err.size > 0:
                        yield (sr_err, arr_err), AdditionalOutputs()
                elif isinstance(tts_err_chunk, np.ndarray) and tts_err_chunk.size > 0:
                    yield (AUDIO_SAMPLE_RATE, tts_err_chunk), AdditionalOutputs()
        except Exception:
            yield EMPTY_AUDIO_YIELD_OUTPUT
    
    def _audio_to_bytes(self, audio_tuple: Tuple[int, np.ndarray]) -> bytes:
        """
        Convert audio tuple to bytes for STT processing.
        
        Args:
            audio_tuple: Tuple of (sample_rate, audio_array)
            
        Returns:
            Audio data as bytes
        """
        sample_rate, audio_array = audio_tuple
        
        # Ensure audio is in the correct format
        if audio_array.dtype != np.float32:
            audio_array = audio_array.astype(np.float32)
        
        # Normalize audio if needed
        if np.max(np.abs(audio_array)) > 1.0:
            audio_array = audio_array / np.max(np.abs(audio_array))
        
        return audio_array.tobytes()