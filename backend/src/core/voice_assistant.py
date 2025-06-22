"""
Voice Assistant Core Module

Main orchestrator for the FastRTC voice assistant system.
Integrates all components from previous phases using dependency injection.
"""

import os
import hashlib

import numpy as np
import aiohttp
from collections import deque
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, Any, List
from .interfaces import AudioData


from ..audio import (
    BluetoothAudioProcessor, STTEngine, KokoroTTSEngine,
    VoiceMapper
)
from ..memory import AMemMemoryManager, ResponseCache, ConversationBuffer
from ..services import LLMService, AsyncManager
from src.config.settings import load_config
from ..config.language_config import LANGUAGE_NAMES, WHISPER_TO_KOKORO_LANG, DEFAULT_LANGUAGE
from ..utils.logging import get_logger

logger = get_logger(__name__)

# Centralized configuration
config = load_config()

class VoiceAssistant:
    """
    Main voice assistant orchestrator that integrates all system components.
    
    This class implements the dependency injection pattern and coordinates:
    - Audio processing and language detection
    - Memory management and conversation tracking
    - LLM services and response generation
    - Async operations and resource management
    """
    
    def __init__(
        self,
        audio_processor: Optional[BluetoothAudioProcessor] = None,
        stt_engine: Optional[STTEngine] = None,
        tts_engine: Optional[KokoroTTSEngine] = None,
        voice_mapper: Optional[VoiceMapper] = None,
        memory_manager: Optional[AMemMemoryManager] = None,
        response_cache: Optional[ResponseCache] = None,
        conversation_buffer: Optional[ConversationBuffer] = None,
        llm_service: Optional[LLMService] = None,
        async_manager: Optional[AsyncManager] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the voice assistant with dependency injection.
        
        Args:
            audio_processor: Audio processing component
            stt_engine: Speech-to-text engine
            tts_engine: Text-to-speech engine
            language_detector: Language detection component
            voice_mapper: Voice mapping component
            memory_manager: Memory management system
            response_cache: Response caching system
            conversation_buffer: Conversation tracking buffer
            llm_service: LLM service for response generation
            async_manager: Async operations manager
            config: Application configuration (required)
        """
        print("[DEBUG] VoiceAssistant.__init__: Starting initialization")
        logger.info("🧠 Initializing VoiceAssistant with dependency injection...")

        # Store configuration
        print("[DEBUG] VoiceAssistant.__init__: Storing config")
        self.config = config

        # Session and user tracking (set early for memory manager)
        print("[DEBUG] VoiceAssistant.__init__: Setting user ID and session ID")
        self.user_id = "guest_user"  # Default until user identifies themselves
        self.session_id = f"session_{self.user_id}"  # Persistent per user
        self.user_identified = False

        # Initialize or use provided components
        print("[DEBUG] VoiceAssistant.__init__: Creating audio processor")
        self.audio_processor = audio_processor or BluetoothAudioProcessor()
        print("[DEBUG] VoiceAssistant.__init__: Creating STT engine")
        self.stt_engine = stt_engine or STTEngine()
        print("[DEBUG] VoiceAssistant.__init__: Creating TTS engine")
        self.tts_engine = tts_engine or KokoroTTSEngine()
        print("[DEBUG] VoiceAssistant.__init__: Creating voice mapper")
        self.voice_mapper = voice_mapper or VoiceMapper()
        print("[DEBUG] VoiceAssistant.__init__: Creating response cache")
        self.response_cache = response_cache or ResponseCache()
        print("[DEBUG] VoiceAssistant.__init__: Creating conversation buffer")
        self.conversation_buffer = conversation_buffer or ConversationBuffer()
        print("[DEBUG] VoiceAssistant.__init__: Creating LLM service")
        self.llm_service = llm_service or LLMService()
        print("[DEBUG] VoiceAssistant.__init__: Creating async manager")
        self.async_manager = async_manager or AsyncManager()

        # Initialize memory manager with Qdrant setup
        print("[DEBUG] VoiceAssistant.__init__: About to setup memory manager")
        self.memory_manager = memory_manager or self._setup_memory_manager()
        print("[DEBUG] VoiceAssistant.__init__: Memory manager setup complete")

        # Conversation state
        config_default_lang = self.config.audio.get("default_language", "en") if hasattr(self.config.audio, "get") else getattr(self.config.audio, "default_language", "en")
        self.current_language = self.convert_to_kokoro_language(config_default_lang)
        self.turn_count = 0
        self.voice_detection_successes = 0
        self.total_response_time = deque(maxlen=20)
        
        # Voice recognition state
        self.enrollment_mode = False
        self.enrollment_samples = []
        self.enrollment_target_name = None

        # Async resources
        self.http_session: Optional[aiohttp.ClientSession] = None

        # Cache configuration
        self.cache_ttl_seconds = 180

        logger.info(f"👤 User ID: {self.user_id}, Session: {self.session_id}")
        self._log_configuration()
    
    def _setup_memory_manager(self) -> AMemMemoryManager:
        """
        Set up the memory manager with user-specific configuration.
        
        Returns:
            Configured AMemMemoryManager instance
        """
        logger.info(f"🔧 Setting up memory manager for user: {self.user_id}")
        # Set up dummy OpenAI key for local use
        os.environ["OPENAI_API_KEY"] = "dummy-key-for-local-use"
        return AMemMemoryManager(self.user_id)
    
    def _log_configuration(self):
        """Log the current system configuration."""
        if config.llm.use_ollama:
            logger.info(f"🗣️ Conversational LLM: Ollama ({config.llm.ollama_model} via {config.llm.ollama_url})")
        else:
            logger.info(f"🗣️ Conversational LLM: LM Studio ({config.llm.lm_studio_model} via {config.llm.lm_studio_url})")
        
        logger.info(f"🧠 A-MEM System: {config.memory.llm_model} with {config.memory.embedder_model} embeddings")
        logger.info("🎤 STT System: Faster-Whisper (CTranslate2 optimized)")
    
    async def initialize_async(self):
        """Initialize async components for the voice assistant."""
        logger.info("🔧 Initializing async components for VoiceAssistant...")
        
        # Initialize HTTP session
        connector = aiohttp.TCPConnector(limit_per_host=5, ssl=False)
        timeout = aiohttp.ClientTimeout(total=20, connect=5, sock_read=15)
        self.http_session = aiohttp.ClientSession(connector=connector, timeout=timeout)
        logger.info("✅ aiohttp ClientSession created.")
        
        # CRITICAL FIX: Initialize LLM service with HTTP session
        logger.info("🔧 Initializing LLM service with HTTP session...")
        await self.llm_service.initialize(
            http_session=self.http_session,
            response_cache=self.response_cache,
            conversation_buffer=self.conversation_buffer,
            memory_manager=self.memory_manager
        )
        logger.info("✅ LLM service initialized with HTTP session.")
        
        # Initialize memory manager async components
        await self.memory_manager.start_background_processor()
        logger.info("✅ Memory manager background processor started.")
        
        # Initialize async manager
        await self.async_manager.initialize()
        logger.info("✅ Async manager initialized.")
        
        logger.info("✅ Async components initialized for VoiceAssistant.")
    
    async def cleanup_async(self):
        """Clean up async resources."""
        logger.info("🧹 Starting async cleanup for VoiceAssistant...")
        
        # Close HTTP session
        if self.http_session:
            await self.http_session.close()
            logger.info("✅ aiohttp ClientSession closed.")
        
        # Shutdown memory manager
        await self.memory_manager.shutdown()
        logger.info("✅ Memory manager shutdown complete.")
        
        # Cleanup async manager
        await self.async_manager.cleanup()
        logger.info("✅ Async manager cleanup complete.")
        
        logger.info("🧹 Async cleanup completed for VoiceAssistant.")
    
    def process_audio_array(self, audio_data):
        """
        Process incoming audio data through the audio processor.
        
        Args:
            audio_data: Raw audio data from FastRTC
            
        Returns:
            Processed audio tuple (sample_rate, audio_array)
        """
        # Convert audio_data to AudioData format for processing
        import numpy as np

        # Log input audio stats
        if isinstance(audio_data, tuple) and len(audio_data) == 2:
            sample_rate, audio_array = audio_data
            if isinstance(audio_array, np.ndarray) and audio_array.size > 0:
                logger.info(
                    f"[AUDIO DIAG] Input: shape={audio_array.shape}, dtype={audio_array.dtype}, "
                    f"min={audio_array.min():.6f}, max={audio_array.max():.6f}, "
                    f"mean={audio_array.mean():.6f}, rms={np.sqrt(np.mean(audio_array**2)):.6f}"
                )

            # --- Robust flattening and conversion for all common shapes ---
            if isinstance(audio_array, np.ndarray):
                # --- Match start_original_backup.py audio flattening logic ---
                # Squeeze to remove single-dimensional entries
                audio_array = np.squeeze(audio_array)
                # If still 2D, average across axis 0 or 1 to get mono
                if len(audio_array.shape) > 1:
                    if audio_array.shape[0] == 1:
                        audio_array = audio_array[0]
                    elif audio_array.shape[1] == 1:
                        audio_array = audio_array[:, 0]
                    elif audio_array.shape[1] > 1:
                        audio_array = np.mean(audio_array, axis=1)
                    else:
                        audio_array = audio_array.flatten()
                # Convert int16 to float32 in range [-1, 1]
                if isinstance(audio_array, np.ndarray):
                    if audio_array.dtype == np.int16:
                        audio_array = audio_array.astype(np.float32) / 32768.0
                    else:
                        audio_array = audio_array.astype(np.float32)
    
                    # --- Resample to 16kHz mono if needed ---
                    TARGET_SAMPLE_RATE = 16000
                    if sample_rate != TARGET_SAMPLE_RATE:
                        from scipy.signal import resample
                        num_samples = int(len(audio_array) * TARGET_SAMPLE_RATE / sample_rate)
                        audio_array = resample(audio_array, num_samples)
                        logger.warning(f"[AUDIO DIAG] Resampled from {sample_rate}Hz to {TARGET_SAMPLE_RATE}Hz, new shape={audio_array.shape}")
                        sample_rate = TARGET_SAMPLE_RATE

            audio_data_obj = AudioData(
                samples=audio_array,
                sample_rate=sample_rate,
                duration=len(audio_array) / sample_rate if sample_rate > 0 else 0.0
            )
        else:
            # Handle other formats if needed
            audio_data_obj = AudioData(
                samples=np.array([], dtype=np.float32),
                sample_rate=16000,
                duration=0.0
            )
        
        # Process through the audio processor
        processed_audio = self.audio_processor.process(audio_data_obj)

        # Log processed audio stats
        if hasattr(processed_audio, "samples") and isinstance(processed_audio.samples, np.ndarray) and processed_audio.samples.size > 0:
            logger.info(
                f"[AUDIO DIAG] Processed: shape={processed_audio.samples.shape}, dtype={processed_audio.samples.dtype}, "
                f"min={processed_audio.samples.min():.6f}, max={processed_audio.samples.max():.6f}, "
                f"mean={processed_audio.samples.mean():.6f}, rms={np.sqrt(np.mean(processed_audio.samples**2)):.6f}"
            )

        return processed_audio.sample_rate, processed_audio.samples
    
    async def process_audio_turn(self, user_text: str) -> str:
        """
        Process a complete audio turn: generate response and update memory.
        
        Args:
            user_text: Transcribed user input
            
        Returns:
            Generated assistant response
        """
        # Check cache first
        cached_response = self.get_cached_response(user_text) # In VoiceAssistant
        if cached_response:
            logger.info("📋 Using cached response")
            return cached_response
        
        # Generate new response using LLM service
        
        # === MODIFICATION START ===
        # Build context from BOTH A-MEM long-term memory AND current session history
        
        # 1. Get long-term context from A-MEM
        amem_context = ""
        if self.memory_manager: # Ensure memory_manager is initialized
            amem_context = self.memory_manager.get_user_context() # This should return "The user's name is Peppy..."
            logger.debug(f"A-MEM Context for LLM: {amem_context}")

        # 2. Get short-term context from current conversation buffer
        session_context = ""
        turns = self.conversation_buffer.get_turns() # Gets turns from *current session* buffer
        if turns:
            session_turns_text = "\n".join([f"User: {turn.user}\nAssistant: {turn.assistant}" for turn in turns[-3:]]) # Assuming turn is ConversationTurn object
            if session_turns_text:
                session_context = f"\n\nRecent exchanges in this session:\n{session_turns_text}"
            logger.debug(f"Session Context for LLM: {session_context}")

        # Combine contexts (A-MEM context should ideally be part of the system prompt in LLMService,
        # but for now, let's pass it as part of the general context.
        # Or, better, LLMService should be responsible for fetching and combining these if needed)
        # For this immediate fix, let's ensure amem_context is passed to LLMService
        
        # The `context` variable passed to `llm_service.get_response` in the original code
        # was only derived from the current session's conversation_buffer.
        # We need to ensure the `amem_context` (which contains "Peppy") is used.
        # The LLMService._get_llm_context_prompt ALREADY takes a `context` argument
        # which is intended for this A-MEM info.
        
        # So, the main `context` variable passed to `llm_service.get_response` should be `amem_context`.
        # The `LLMService._get_llm_context_prompt` will then append its own `recent_conv` (session buffer).
        
        final_context_for_llm_service = amem_context
        # === MODIFICATION END ===
        
        # Calls the LLMService instance's get_response
        # Pass the A-MEM derived context here.
        response = await self.llm_service.get_response(user_text, final_context_for_llm_service)
        
        # Update memory and cache
        await self.memory_manager.add_to_memory_smart(user_text, response) # AMemMemoryManager
        self.cache_response(user_text, response) # VoiceAssistant's cache
        
        return response
    
    async def get_llm_response_smart(self, user_text: str) -> str:
        """
        Get intelligent LLM response with memory integration.
        
        Args:
            user_text: User input text
            
        Returns:
            Generated response text
        """
        return await self.process_audio_turn(user_text)
    
    def detect_language_from_text(self, text: str) -> str:
        """
        Detect language from text using the language detector.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Detected language code (Kokoro format)
        """
        language, confidence = self.language_detector.detect_language(text)
        logger.debug(f"Language detection: {language} (confidence: {confidence:.3f})")
        
        # Check if language is already in Kokoro format (single letter)
        if len(language) == 1:
            # Already in Kokoro format, return as-is
            logger.debug(f"Language already in Kokoro format: {language}")
            return language
        else:
            # Convert to Kokoro language code
            kokoro_language = self.convert_to_kokoro_language(language)
            logger.debug(f"Language mapping: {language} → {kokoro_language}")
            return kokoro_language
    
    def convert_to_kokoro_language(self, language_code: str) -> str:
        """
        Convert standard language codes to Kokoro language codes.
        
        Args:
            language_code: Standard language code (e.g., 'en', 'it', 'es')
            
        Returns:
            Kokoro language code (e.g., 'a', 'i', 'e')
        """
        # Use the mapping from language_config
        kokoro_code = WHISPER_TO_KOKORO_LANG.get(language_code, DEFAULT_LANGUAGE)
        return kokoro_code
    
    def get_voices_for_language(self, language_code: str) -> List[str]:
        """
        Get available voices for a specific language.
        
        Args:
            language_code: Language code to get voices for
            
        Returns:
            List of available voice IDs
        """
        return self.voice_mapper.get_voices_for_language(language_code)
    
    def stream_tts_synthesis(self, text: str, voice: str, language: str):
        """
        Stream TTS synthesis results for FastRTC integration.
        
        Args:
            text: Text to synthesize
            voice: Voice identifier
            language: Language code (Kokoro format)
            
        Yields:
            Tuple[int, np.ndarray]: (sample_rate, audio_chunk)
        """
        if not self.tts_engine.is_available():
            logger.error("TTS engine is not available for streaming")
            return
        
        try:
            # Use the TTS engine's streaming method
            for sample_rate, audio_chunk in self.tts_engine.stream_synthesis(text, voice, language):
                yield sample_rate, audio_chunk
        except Exception as e:
            logger.error(f"TTS streaming failed: {e}")
            raise
    
    def get_cached_response(self, text: str) -> Optional[str]:
        """
        Get cached response for the given text.
        
        Args:
            text: Input text to check cache for
            
        Returns:
            Cached response if available, None otherwise
        """
        return self.response_cache.get(text)
    
    def cache_response(self, text: str, response: str):
        """
        Cache a response for future use.
        
        Args:
            text: Input text (cache key)
            response: Response to cache
        """
        self.response_cache.put(text, response)
    
    def get_system_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive system statistics.
        
        Returns:
            Dictionary containing system statistics
        """
        try:
            # Memory statistics
            mem_stats = self.memory_manager.get_stats()
            
            # Audio statistics
            audio_stats = self.audio_processor.get_detection_stats()
            
            # Response time statistics
            avg_response_time = (
                sum(self.total_response_time) / len(self.total_response_time)
                if self.total_response_time else 0
            )
            
            # Language statistics
            current_lang_name = LANGUAGE_NAMES.get(self.current_language, 'Unknown')
            available_voices = len(self.get_voices_for_language(self.current_language))
            
            return {
                'session_info': {
                    'user_id': self.user_id,
                    'session_id': self.session_id,
                    'turn_count': self.turn_count,
                    'voice_detections': self.voice_detection_successes
                },
                'performance': {
                    'avg_response_time': avg_response_time,
                    'total_turns': self.turn_count,
                    'cache_size': len(self.response_cache._cache) if hasattr(self.response_cache, '_cache') else 0
                },
                'language': {
                    'current_language': self.current_language,
                    'language_name': current_lang_name,
                    'available_voices': available_voices
                },
                'memory': mem_stats,
                'audio': audio_stats
            }
        except Exception as e:
            logger.error(f"❌ Error getting system stats: {e}")
            return {'error': str(e)}
    
    def reset_session(self):
        """Reset the current session while preserving user identity."""
        logger.info("🔄 Resetting voice assistant session...")
        
        # Keep the same session ID to preserve user identity and memory
        # Only reset conversation state, not user identity
        
        # Reset counters
        self.turn_count = 0
        self.voice_detection_successes = 0
        self.total_response_time.clear()
        
        # Clear conversation buffer
        self.conversation_buffer.clear()
        
        # Reset language to default
        self.current_language = DEFAULT_LANGUAGE
        
        logger.info(f"✅ Session reset complete. New session: {self.session_id}")
    
    def identify_user(self, name: str) -> bool:
        """
        Identify user and switch to their persistent session.
        
        Args:
            name: User's name from speech ("My name is John")
            
        Returns:
            True if user identification was successful
        """
        try:
            # Clean and normalize the name
            clean_name = self._normalize_username(name)
            if not clean_name:
                logger.warning(f"⚠️ Invalid username: {name}")
                return False
            
            # Create user ID from name
            new_user_id = f"user_{clean_name}"
            
            # Check if this is a new user
            if new_user_id != self.user_id:
                logger.info(f"👤 User identification: {name} -> {new_user_id}")
                
                # Store old session info for potential recovery
                old_user_id = self.user_id
                old_session_id = self.session_id
                
                # Switch to new user
                self.user_id = new_user_id
                self.session_id = f"session_{new_user_id}"
                self.user_identified = True
                
                # Reinitialize memory manager for new user
                self._switch_user_memory(old_user_id, new_user_id)
                
                logger.info(f"✅ Switched to user: {clean_name} (session: {self.session_id})")
                return True
            else:
                logger.info(f"👤 User already identified as: {clean_name}")
                self.user_identified = True
                return True
                
        except Exception as e:
            logger.error(f"❌ User identification failed: {e}")
            return False
    
    def _normalize_username(self, name: str) -> str:
        """
        Normalize username for consistent storage.
        
        Args:
            name: Raw name from speech
            
        Returns:
            Normalized username or empty string if invalid
        """
        if not name or not isinstance(name, str):
            return ""
        
        # Clean the name: lowercase, remove special chars, keep letters/numbers/spaces
        import re
        clean = re.sub(r'[^a-zA-Z0-9\s]', '', name.lower().strip())
        
        # Replace spaces with underscores, remove multiple spaces
        clean = re.sub(r'\s+', '_', clean)
        
        # Limit length and ensure it's not empty
        clean = clean[:20]  # Max 20 characters
        
        return clean if len(clean) >= 2 else ""
    
    def _switch_user_memory(self, old_user_id: str, new_user_id: str):
        """
        Switch memory manager to new user.
        
        Args:
            old_user_id: Previous user ID
            new_user_id: New user ID
        """
        try:
            logger.info(f"🧠 Switching memory from {old_user_id} to {new_user_id}")
            
            # Create new memory manager for the new user
            old_memory_manager = self.memory_manager
            self.memory_manager = self._setup_memory_manager()
            
            # Update memory manager's user ID
            if hasattr(self.memory_manager, 'user_id'):
                self.memory_manager.user_id = new_user_id
            
            logger.info(f"✅ Memory switched to user: {new_user_id}")
            
        except Exception as e:
            logger.error(f"❌ Failed to switch user memory: {e}")
            # Fallback: keep old memory manager
            logger.warning("⚠️ Keeping previous memory manager as fallback")
    
    def get_user_info(self) -> Dict[str, Any]:
        """
        Get current user information.
        
        Returns:
            Dictionary with user info
        """
        return {
            "user_id": self.user_id,
            "session_id": self.session_id,
            "identified": self.user_identified,
            "is_guest": self.user_id.startswith("guest_"),
            "display_name": self.user_id.replace("user_", "").replace("_", " ").title() if self.user_identified else "Guest"
        }
    
    def check_for_user_identification(self, user_text: str) -> bool:
        """
        Check if user text contains identification and process it.
        
        Args:
            user_text: User's speech text
            
        Returns:
            True if user was identified from the text
        """
        import re
        
        # Patterns for user identification
        name_patterns = [
            r"my name is (\w+(?:\s+\w+)?)",
            r"i am (\w+(?:\s+\w+)?)",
            r"i'm (\w+(?:\s+\w+)?)",
            r"call me (\w+(?:\s+\w+)?)",
            r"this is (\w+(?:\s+\w+)?)"
        ]
        
        for pattern in name_patterns:
            match = re.search(pattern, user_text.lower())
            if match:
                name = match.group(1).strip()
                if self.identify_user(name):
                    return True
        
        return False
    
    def _on_voice_identified(self, user_id: str, name: str, confidence: float):
        """
        Callback when user is identified via voice recognition (background).
        
        Args:
            user_id: Identified user ID
            name: User's name
            confidence: Recognition confidence
        """
        try:
            # Only switch if it's a different user
            if user_id != self.user_id:
                logger.info(f"🎙️ Voice identified (background): {name} (confidence: {confidence:.3f})")
                
                # Switch user context
                old_user_id = self.user_id
                self.user_id = user_id
                self.session_id = f"session_{user_id}"
                self.user_identified = True
                
                # Switch memory context
                self._switch_user_memory(old_user_id, user_id)
                
                logger.info(f"✅ Switched to user: {name} via voice recognition")
        
        except Exception as e:
            logger.error(f"❌ Voice identification callback failed: {e}")
    
    def process_audio_for_background_recognition(self, audio: np.ndarray, sample_rate: int = 16000):
        """
        Process audio for background voice recognition (non-blocking).
        
        Args:
            audio: Audio samples
            sample_rate: Audio sample rate
        """
        try:
            # Add audio to background processing queue
            self.voice_print_manager.add_audio_chunk(audio, sample_rate)
        except Exception as e:
            logger.debug(f"Background voice processing error: {e}")
    
    def _check_enrollment_commands(self, user_text: str) -> Optional[str]:
        """
        Check for voice enrollment commands in user text.
        
        Args:
            user_text: User's speech text
            
        Returns:
            Response message if enrollment command found, None otherwise
        """
        import re
        
        text_lower = user_text.lower()
        
        # Check for enrollment start commands
        enrollment_patterns = [
            r"learn my voice",
            r"teach you my voice", 
            r"enroll my voice",
            r"remember my voice",
            r"train my voice"
        ]
        
        for pattern in enrollment_patterns:
            if re.search(pattern, text_lower):
                # Extract name if provided
                name_match = re.search(r"(?:i am|my name is|call me|i'm)\s+(\w+(?:\s+\w+)?)", text_lower)
                if name_match:
                    name = name_match.group(1).strip()
                    return self.start_voice_enrollment(name)
                else:
                    return "I'd like to learn your voice! What's your name?"
        
        # Check for enrollment cancellation
        if re.search(r"cancel.*enrollment|stop.*enrollment|never mind", text_lower):
            if self.enrollment_mode:
                return self.cancel_enrollment()
        
        return None
    
    def process_audio_for_voice_recognition(self, audio: np.ndarray, sample_rate: int = 16000) -> Optional[str]:
        """
        Process audio for voice-based user identification.
        
        Args:
            audio: Audio samples
            sample_rate: Audio sample rate
            
        Returns:
            User ID if identified, None otherwise
        """
        try:
            # Skip if audio is too short
            if len(audio) < sample_rate * 0.5:  # Less than 0.5 seconds
                return None
            
            # Try to identify speaker
            result = self.voice_print_manager.identify_speaker(audio, sample_rate)
            
            if result:
                user_id, name, confidence = result
                
                # Switch to identified user if different
                if user_id != self.user_id:
                    logger.info(f"🎙️ Voice identified: {name} (confidence: {confidence:.3f})")
                    
                    # Update user identity
                    old_user_id = self.user_id
                    self.user_id = user_id
                    self.session_id = f"session_{user_id}"
                    self.user_identified = True
                    
                    # Switch memory context
                    self._switch_user_memory(old_user_id, user_id)
                    
                    return user_id
            
            return None
            
        except Exception as e:
            logger.debug(f"Voice recognition error: {e}")
            return None
    
    def start_voice_enrollment(self, name: str) -> str:
        """
        Start voice enrollment process for a user.
        
        Args:
            name: User's name
            
        Returns:
            Instructions for the user
        """
        try:
            self.enrollment_mode = True
            self.enrollment_samples = []
            self.enrollment_target_name = name
            
            logger.info(f"🎙️ Starting voice enrollment for: {name}")
            
            return (f"Starting voice enrollment for {name}. "
                   f"Please say a few sentences. I need at least 3 samples to learn your voice.")
                   
        except Exception as e:
            logger.error(f"❌ Failed to start enrollment: {e}")
            return "Sorry, I couldn't start voice enrollment."
    
    def process_enrollment_audio(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
        """
        Process audio during enrollment.
        
        Args:
            audio: Audio samples
            sample_rate: Audio sample rate
            
        Returns:
            Status message for the user
        """
        try:
            if not self.enrollment_mode:
                return "Not in enrollment mode."
            
            # Skip if audio is too short
            if len(audio) < sample_rate * 1.0:  # Less than 1 second
                return "Please say something longer."
            
            # Add sample to enrollment
            self.enrollment_samples.append((audio, sample_rate))
            
            remaining = max(0, 3 - len(self.enrollment_samples))
            
            if remaining > 0:
                return f"Sample {len(self.enrollment_samples)} recorded. Need {remaining} more samples."
            else:
                # Enough samples, complete enrollment
                return self._complete_enrollment()
                
        except Exception as e:
            logger.error(f"❌ Enrollment audio processing failed: {e}")
            return "Sorry, there was an error processing your voice sample."
    
    def _complete_enrollment(self) -> str:
        """Complete the voice enrollment process"""
        try:
            if not self.enrollment_target_name or not self.enrollment_samples:
                return "Enrollment data missing."
            
            # Enroll user with voice samples
            success = self.voice_print_manager.enroll_user(
                name=self.enrollment_target_name,
                audio_samples=self.enrollment_samples
            )
            
            if success:
                # Switch to newly enrolled user
                user_id = f"user_{self.enrollment_target_name.lower().replace(' ', '_')}"
                old_user_id = self.user_id
                
                self.user_id = user_id
                self.session_id = f"session_{user_id}"
                self.user_identified = True
                
                # Switch memory context
                self._switch_user_memory(old_user_id, user_id)
                
                # Clear enrollment state
                self.enrollment_mode = False
                self.enrollment_samples = []
                self.enrollment_target_name = None
                
                logger.info(f"✅ Voice enrollment completed for: {self.enrollment_target_name}")
                
                return (f"Perfect! I've learned your voice, {self.enrollment_target_name}. "
                       f"From now on, I'll recognize you automatically when you speak.")
            else:
                return "Sorry, voice enrollment failed. Please try again."
                
        except Exception as e:
            logger.error(f"❌ Enrollment completion failed: {e}")
            return "Sorry, there was an error completing your voice enrollment."
    
    def cancel_enrollment(self) -> str:
        """Cancel ongoing voice enrollment"""
        self.enrollment_mode = False
        self.enrollment_samples = []
        self.enrollment_target_name = None
        return "Voice enrollment cancelled."
    
    def get_voice_recognition_stats(self) -> Dict[str, Any]:
        """Get voice recognition system statistics"""
        try:
            stats = self.voice_print_manager.get_stats()
            enrolled_users = self.voice_print_manager.get_enrolled_users()
            
            return {
                "system_stats": stats,
                "enrolled_users": enrolled_users,
                "enrollment_mode": self.enrollment_mode,
                "enrollment_samples_count": len(self.enrollment_samples) if self.enrollment_mode else 0
            }
        except Exception as e:
            logger.error(f"❌ Failed to get voice recognition stats: {e}")
            return {"error": str(e)}
    
    def __repr__(self) -> str:
        """String representation of the voice assistant."""
        return (
            f"VoiceAssistant(user_id='{self.user_id}', "
            f"session_id='{self.session_id}', "
            f"language='{self.current_language}', "
            f"turns={self.turn_count})"
        )