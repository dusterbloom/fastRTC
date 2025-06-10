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

from ..audio import HuggingFaceSTTEngine, KokoroTTSEngine, HybridLanguageDetector, VoiceMapper
from ..audio.engines.tts.kokoro_tts import KokoroTTSOptions
from ..utils.async_utils import run_coro_from_sync_thread_with_timeout
from ..utils.logging import get_logger
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
        stt_engine: HuggingFaceSTTEngine,
        tts_engine: KokoroTTSEngine,
        language_detector: HybridLanguageDetector,
        voice_mapper: VoiceMapper,
        event_loop=None
    ):
        """
        Initialize the stream callback handler.
        
        Args:
            voice_assistant: The main voice assistant instance
            stt_engine: Speech-to-text engine
            tts_engine: Text-to-speech engine
            language_detector: Language detection component
            voice_mapper: Voice mapping component
            event_loop: Async event loop for coroutine execution
        """
        self.voice_assistant = voice_assistant
        self.stt_engine = stt_engine
        self.tts_engine = tts_engine
        self.language_detector = language_detector
        self.voice_mapper = voice_mapper
        self.event_loop = event_loop
        
        # Language names mapping
        self.lang_names = LANGUAGE_NAMES
        self.lang_abbreviations = LANGUAGE_ABBREVIATIONS
        
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
        if not self.voice_assistant:
            yield EMPTY_AUDIO_YIELD_OUTPUT
            return
        
        try:
            # Process the incoming audio data
            sample_rate, audio_array = self.voice_assistant.process_audio_array(audio_data_tuple)
            
            if audio_array.size == 0:
                yield SILENT_AUDIO_FRAME_TUPLE, AdditionalOutputs()
                return
            
            # Perform speech-to-text conversion
            user_text, detected_language = self._process_speech_to_text(audio_array, sample_rate)
            
            if not user_text.strip():
                yield SILENT_AUDIO_FRAME_TUPLE, AdditionalOutputs()
                return
            
            # Update language if changed
            self._update_language(detected_language)
            
            # Update statistics
            self.voice_assistant.voice_detection_successes += 1
            logger.info(f"👤 User: {user_text}")
            
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
            
            # Graceful error recovery
            yield from self._handle_error_recovery()
    
    def _process_speech_to_text(self, audio_array: np.ndarray, sample_rate: int) -> Tuple[str, str]:
        """
        Process audio array to extract text and detect language.
        
        Args:
            audio_array: Audio data as numpy array
            sample_rate: Audio sample rate
            
        Returns:
            Tuple of (transcribed_text, detected_language)
        """
        # Convert audio to bytes for STT processing
        audio_bytes = self._audio_to_bytes((sample_rate, audio_array))
        
        # Perform transcription
        outputs = self.stt_engine.transcribe(
            audio_bytes,
            chunk_length_s=30,
            batch_size=1,
            generate_kwargs={'task': 'transcribe'},
            return_timestamps=False,
        )
        
        user_text = outputs["text"].strip()
        logger.info(f"📝 Transcribed: '{user_text}'")
        
        # Detect language using multiple methods
        detected_language = self._detect_language_multi_method(outputs, user_text)
        
        return user_text, detected_language
    
    def _detect_language_multi_method(self, stt_outputs: dict, user_text: str) -> str:
        """
        Detect language using multiple detection methods for accuracy.
        
        Args:
            stt_outputs: Outputs from STT engine
            user_text: Transcribed text
            
        Returns:
            Detected language code
        """
        # Method 1: Check if Whisper provided language info
        whisper_language = 'en'  # Default
        
        if hasattr(stt_outputs, 'get'):
            if 'language' in stt_outputs:
                whisper_language = stt_outputs['language']
                logger.info(f"🎤 Whisper detected language: {whisper_language}")
            elif 'chunks' in stt_outputs and stt_outputs['chunks']:
                for chunk in stt_outputs['chunks']:
                    if 'language' in chunk:
                        whisper_language = chunk['language']
                        logger.info(f"🎤 Whisper chunk language: {whisper_language}")
                        break
        
        # Method 2: Text-based detection (primary method)
        text_detected_lang, confidence = self.language_detector.detect_language(user_text)
        
        # Method 3: Combine detections with priority to text detection
        if text_detected_lang != 'a':  # If text detection found non-English
            detected_language = text_detected_lang
            logger.info(f"🔤 Text-based detection: {detected_language} (confidence: {confidence:.3f})")
        else:
            # Fall back to Whisper detection
            detected_language = self._get_kokoro_language(whisper_language)
            logger.info(f"🎤 Using Whisper detection: {whisper_language} -> {detected_language}")
        
        return detected_language
    
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
    
    def _update_language(self, detected_language: str):
        """
        Update the current language if it has changed.
        
        Args:
            detected_language: The detected language code
        """
        if detected_language != self.voice_assistant.current_language:
            self.voice_assistant.current_language = detected_language
            lang_name = self.lang_names.get(detected_language, 'Unknown')
            logger.info(f"🌍 Language switched to: {lang_name} ({detected_language})")
        else:
            lang_name = self.lang_names.get(detected_language, 'Unknown')
            logger.info(f"🌍 Language confirmed: {lang_name} ({detected_language})")
    
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
        logger.info(f"🤖 Assistant: {assistant_response_text}")
        logger.info(f"⏱️ Turn Processing Time: {turn_processing_time:.2f}s")
        
        return assistant_response_text
    
    def _update_conversation(self, user_text: str, assistant_response: str):
        """
        Update conversation tracking and statistics.
        
        Args:
            user_text: User's input text
            assistant_response: Assistant's response text
        """
        # Update conversation buffer
        self.voice_assistant.conversation_buffer.append({
            'user': user_text,
            'assistant': assistant_response,
            'timestamp': datetime.now(timezone.utc)
        })
        
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
            
            logger.info(
                f"📊 Turn {self.voice_assistant.turn_count}: "
                f"AvgResp={avg_resp:.2f}s | MemOps={mem_stats['mem_ops']} | "
                f"User='{mem_stats['user_name_cache']}'{audio_q_str}{lang_str}"
            )
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
        additional_outputs = AdditionalOutputs()
        tts_voices_to_try = self.voice_mapper.get_voices_for_language(
            self.voice_assistant.current_language
        )
        tts_voices_to_try.append(None)  # Fallback to default voice
        
        logger.info(
            f"🎤 TTS using language '{self.voice_assistant.current_language}' "
            f"with voices: {tts_voices_to_try[:3]}"
        )
        
        tts_success = False
        for voice_id in tts_voices_to_try:
            try:
                # Configure TTS options
                options_params = {"speed": 1.05}
                kokoro_tts_lang = KOKORO_TTS_LANG_MAP.get(
                    self.voice_assistant.current_language, 'en-us'
                )
                options_params["lang"] = kokoro_tts_lang
                
                if voice_id:
                    options_params["voice"] = voice_id
                
                tts_options = KokoroTTSOptions(**options_params)
                logger.info(f"🔊 Trying TTS with voice '{voice_id}', lang '{kokoro_tts_lang}'")
                
                # Stream TTS audio
                chunk_count = 0
                total_samples = 0
                
                for tts_output_item in self.tts_engine.stream_tts_sync(response_text, tts_options):
                    if isinstance(tts_output_item, tuple) and len(tts_output_item) == 2:
                        current_sr, current_chunk_array = tts_output_item
                        if isinstance(current_chunk_array, np.ndarray) and current_chunk_array.size > 0:
                            chunk_count += 1
                            total_samples += current_chunk_array.size
                            
                            # Yield smaller chunks to prevent timeouts
                            yield from self._yield_audio_chunks(
                                current_sr, current_chunk_array, additional_outputs
                            )
                    
                    elif isinstance(tts_output_item, np.ndarray) and tts_output_item.size > 0:
                        chunk_count += 1
                        total_samples += tts_output_item.size
                        
                        yield from self._yield_audio_chunks(
                            AUDIO_SAMPLE_RATE, tts_output_item, additional_outputs
                        )
                
                tts_success = True
                logger.info(
                    f"✅ TTS stream completed. Voice: {voice_id}, "
                    f"Chunks: {chunk_count}, Samples: {total_samples}"
                )
                
                # Success message with language confirmation
                if self.voice_assistant.current_language != 'a' and voice_id:
                    lang_name = self.lang_names.get(
                        self.voice_assistant.current_language, 'Unknown'
                    )
                    logger.info(f"✅ TTS SUCCESS using {lang_name} voice: {voice_id}")
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