"""
Voice Assistant Core Module

Main orchestrator for the FastRTC voice assistant system.
Integrates all components from previous phases using dependency injection.
"""

import os
import hashlib
import uuid

import numpy as np
import aiohttp
from collections import deque
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, Any, List
from .interfaces import AudioData
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance

from ..audio import (
    BluetoothAudioProcessor, HuggingFaceSTTEngine, KokoroTTSEngine,
    HybridLanguageDetector, VoiceMapper
)
from ..memory import AMemMemoryManager, ResponseCache, ConversationBuffer
from ..services import LLMService, AsyncManager
from ..config.settings import load_config # Corrected import path
from ..config.language_config import LANGUAGE_NAMES, WHISPER_TO_KOKORO_LANG
from ..utils.logging import get_logger

logger = get_logger(__name__)

# Centralized configuration
config_global = load_config() # Renamed to avoid conflict

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
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        audio_processor: Optional[BluetoothAudioProcessor] = None,
        stt_engine: Optional[HuggingFaceSTTEngine] = None,
        tts_engine: Optional[KokoroTTSEngine] = None,
        language_detector: Optional[HybridLanguageDetector] = None,
        voice_mapper: Optional[VoiceMapper] = None,
        memory_manager: Optional[AMemMemoryManager] = None,
        response_cache: Optional[ResponseCache] = None,
        conversation_buffer: Optional[ConversationBuffer] = None,
        llm_service: Optional[LLMService] = None,
        async_manager: Optional[AsyncManager] = None,
        app_config: Optional[Dict[str, Any]] = None # Renamed from config
    ):
        """
        Initialize the voice assistant with dependency injection.
        
        Args:
            user_id: Optional user identifier.
            session_id: Optional session identifier.
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
            app_config: Application configuration
        """
        logger.info("🧠 Initializing VoiceAssistant with dependency injection...")

        # Store configuration
        self.config = app_config if app_config is not None else config_global

        # Session and user tracking
        self.user_id = user_id if user_id else getattr(self.config, 'user_id', "default_user")
        self.session_id = session_id if session_id else f"session_{uuid.uuid4()}"

        # Initialize or use provided components
        self.audio_processor = audio_processor or BluetoothAudioProcessor()
        self.stt_engine = stt_engine or HuggingFaceSTTEngine()
        self.tts_engine = tts_engine or KokoroTTSEngine()
        self.language_detector = language_detector or HybridLanguageDetector()
        self.voice_mapper = voice_mapper or VoiceMapper()
        self.response_cache = response_cache or ResponseCache()
        self.conversation_buffer = conversation_buffer or ConversationBuffer()
        self.llm_service = llm_service or LLMService()
        self.async_manager = async_manager or AsyncManager()

        # Initialize memory manager with Qdrant setup
        self.memory_manager = memory_manager or self._setup_memory_manager()

        # Conversation state
        audio_config = getattr(self.config, 'audio', None)
        default_lang = "en"
        if audio_config:
            default_lang = getattr(audio_config, "default_language", "en")
        
        self.current_language = default_lang
        self.turn_count = 0
        self.voice_detection_successes = 0
        self.total_response_time = deque(maxlen=20)

        # Async resources
        self.http_session: Optional[aiohttp.ClientSession] = None

        # Cache configuration
        self.cache_ttl_seconds = 180 # Consider moving to config

        logger.info(f"👤 User ID: {self.user_id}, Session: {self.session_id}")
        self._log_configuration()
    
    def _setup_memory_manager(self) -> AMemMemoryManager:
        """
        Set up the AMemMemoryManager using centralized configuration.
        This uses ChromaDB for persistence, configured per user.
        
        Returns:
            Configured AMemMemoryManager instance
        """
        logger.info("🔧 Setting up AMemMemoryManager with ChromaDB persistence using centralized config...")

        memory_conf = getattr(self.config, 'memory', None)
        deployment_conf = getattr(self.config, 'deployment', None)

        # Defaults, to be overridden by config if available
        amem_embedder_model = 'all-MiniLM-L6-v2'
        amem_llm_model = "llama3.2:3b"
        amem_evo_threshold = 50
        # Default base path for user-specific ChromaDBs, from deployment config
        persist_dir_base = "./chroma_db_amem"
        amem_llm_backend = "ollama" # Default, as not in centralized MemoryConfig
        amem_api_key = None # Default, as not in centralized MemoryConfig

        if memory_conf:
            amem_embedder_model = getattr(memory_conf, 'embedder_model', amem_embedder_model)
            amem_llm_model = getattr(memory_conf, 'llm_model', amem_llm_model)
            amem_evo_threshold = getattr(memory_conf, 'evolution_threshold', amem_evo_threshold)
            # Note: amem_llm_backend and amem_api_key are not in MemoryConfig,
            # so they will use defaults or require direct setting if needed.
            logger.info(f"Loaded from MemoryConfig: embedder_model='{amem_embedder_model}', "
                        f"llm_model='{amem_llm_model}', evolution_threshold={amem_evo_threshold}")

        if deployment_conf:
            persist_dir_base = getattr(deployment_conf, 'chroma_db_path', persist_dir_base)
            logger.info(f"Loaded from DeploymentConfig: chroma_db_path='{persist_dir_base}'")
        
        # A-MEM might internally use libraries that check for OPENAI_API_KEY.
        # Set a dummy one if no specific API key for A-MEM's LLM is configured and OPENAI_API_KEY is not already set.
        # This is a general precaution.
        if "OPENAI_API_KEY" not in os.environ and not amem_api_key:
             os.environ["OPENAI_API_KEY"] = "dummy-key-for-amem-local-use"
             logger.info("Set dummy OPENAI_API_KEY for A-MEM local use as it was not present.")


        logger.info(f"AMemMemoryManager effective config: user_id='{self.user_id}', "
                    f"amem_model='{amem_embedder_model}', llm_backend='{amem_llm_backend}', "
                    f"llm_model='{amem_llm_model}', evo_threshold={amem_evo_threshold}, "
                    f"persist_directory_base='{persist_dir_base}', api_key provided: {amem_api_key is not None}")

        return AMemMemoryManager(
            user_id=self.user_id,
            amem_model=amem_embedder_model,
            llm_backend=amem_llm_backend, # Using default or passed-in if AMemMemoryManager is ever directly instantiated elsewhere
            llm_model=amem_llm_model,
            evo_threshold=amem_evo_threshold,
            persist_directory_base=persist_dir_base,
            api_key=amem_api_key # Using default or passed-in
        )
    
    def _log_configuration(self):
        """Log the current system configuration."""
        llm_config = getattr(self.config, 'llm', None)
        memory_config = getattr(self.config, 'memory', None)
        audio_config = getattr(self.config, 'audio', None)

        if llm_config:
            if getattr(llm_config, 'use_ollama', False):
                logger.info(f"🗣️ Conversational LLM: Ollama ({getattr(llm_config, 'ollama_model', 'N/A')} via {getattr(llm_config, 'ollama_url', 'N/A')})")
            else:
                logger.info(f"🗣️ Conversational LLM: LM Studio ({getattr(llm_config, 'lm_studio_model', 'N/A')} via {getattr(llm_config, 'lm_studio_url', 'N/A')})")
        else:
            logger.warning("LLM configuration not found.")

        if memory_config:
            logger.info(f"🧠 A-MEM System: {getattr(memory_config, 'llm_model', 'N/A')} with {getattr(memory_config, 'embedder_model', 'N/A')} embeddings")
        else:
            logger.warning("Memory configuration not found.")
        
        if audio_config:
            logger.info(f"🎤 STT System: Hugging Face Transformers (Model: {getattr(audio_config, 'hf_model_id', 'N/A')})")
        else:
            logger.warning("Audio configuration not found.")
    
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
        if isinstance(audio_data, tuple) and len(audio_data) == 2:
            sample_rate, audio_array = audio_data
            
            # Ensure audio is single channel (convert from multi-channel if needed)
            if isinstance(audio_array, np.ndarray) and len(audio_array.shape) > 1:
                if audio_array.shape[1] > 1:  # Multi-channel
                    audio_array = np.mean(audio_array, axis=1)  # Convert to mono
                elif audio_array.shape[0] > 1 and audio_array.shape[1] == 1:  # Single column
                    audio_array = audio_array.flatten()
            
            # Ensure float32 type
            if isinstance(audio_array, np.ndarray):
                audio_array = audio_array.astype(np.float32)
            
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
        """Reset the current session and start fresh."""
        logger.info("🔄 Resetting voice assistant session...")
        
        # Generate new session ID
        self.session_id = f"session_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
        
        # Reset counters
        self.turn_count = 0
        self.voice_detection_successes = 0
        self.total_response_time.clear()
        
        # Clear conversation buffer
        self.conversation_buffer.clear()
        
        # Reset language to default
        self.current_language = DEFAULT_LANGUAGE
        
        logger.info(f"✅ Session reset complete. New session: {self.session_id}")
    
    def __repr__(self) -> str:
        """String representation of the voice assistant."""
        return (
            f"VoiceAssistant(user_id='{self.user_id}', "
            f"session_id='{self.session_id}', "
            f"language='{self.current_language}', "
            f"turns={self.turn_count})"
        )