"""Pytest configuration and fixtures for FastRTC Voice Assistant tests.

This module provides comprehensive test fixtures for dependency injection,
mock objects, and test data generation as specified in the refactoring plan.
"""

import pytest
import numpy as np
import asyncio
from unittest.mock import Mock, AsyncMock, MagicMock
from typing import Dict, List, Any, Optional

# Import our modules
from src.core.interfaces import (
    AudioData, 
    TranscriptionResult, 
    STTEngine, 
    TTSEngine, 
    AudioProcessor,
    LanguageDetector,
    MemoryManager,
    LLMService
)
from src.config.settings import (
    AppConfig, 
    AudioConfig, 
    MemoryConfig, 
    LLMConfig, 
    TTSConfig
)
from src.utils.logging import setup_logging


# Test configuration
@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session", autouse=True)
def setup_test_logging():
    """Set up logging for tests."""
    setup_logging(
        log_level="DEBUG",
        console_output=True,
        colored_output=False,  # Disable colors for test output
        component_levels={
            "backend.tests": "DEBUG"
        }
    )


# Configuration fixtures
@pytest.fixture
def app_config():
    """Provide test application configuration."""
    return AppConfig(
        audio=AudioConfig(
            sample_rate=16000,
            chunk_duration=2.0,
            noise_threshold=0.15,
            minimal_silent_frame_duration_ms=20
        ),
        memory=MemoryConfig(
            llm_model="test-llama3.2:3b",
            embedder_model="test-nomic-embed-text",
            evolution_threshold=50,
            cache_ttl_seconds=180
        ),
        llm=LLMConfig(
            use_ollama=True,
            ollama_url="http://localhost:11434",
            ollama_model="test-llama3:8b-instruct-q4_K_M",
            lm_studio_url="http://localhost:1234/v1",
            lm_studio_model="test-mistral-nemo-instruct-2407"
        ),
        tts=TTSConfig(
            preferred_voice="af_heart",
            fallback_voices=["af_alloy", "af_bella"],
            speed=1.05,
            default_language="a"
        )
    )


@pytest.fixture
def audio_config():
    """Provide test audio configuration."""
    return AudioConfig()


@pytest.fixture
def memory_config():
    """Provide test memory configuration."""
    return MemoryConfig()


@pytest.fixture
def llm_config():
    """Provide test LLM configuration."""
    return LLMConfig()


@pytest.fixture
def tts_config():
    """Provide test TTS configuration."""
    return TTSConfig()


# Audio data fixtures
@pytest.fixture
def sample_audio():
    """Provide sample audio data for testing."""
    duration = 1.0
    sample_rate = 16000
    samples = np.random.random(sample_rate).astype(np.float32) * 0.1  # Low amplitude
    
    return AudioData(
        samples=samples,
        sample_rate=sample_rate,
        duration=duration
    )


@pytest.fixture
def long_audio():
    """Provide longer audio sample for testing."""
    duration = 5.0
    sample_rate = 16000
    samples = np.random.random(int(sample_rate * duration)).astype(np.float32) * 0.1
    
    return AudioData(
        samples=samples,
        sample_rate=sample_rate,
        duration=duration
    )


@pytest.fixture
def silent_audio():
    """Provide silent audio for testing."""
    duration = 1.0
    sample_rate = 16000
    samples = np.zeros(sample_rate, dtype=np.float32)
    
    return AudioData(
        samples=samples,
        sample_rate=sample_rate,
        duration=duration
    )


@pytest.fixture
def noisy_audio():
    """Provide noisy audio for testing."""
    duration = 2.0
    sample_rate = 16000
    # Generate noise with some signal
    signal = np.sin(2 * np.pi * 440 * np.linspace(0, duration, int(sample_rate * duration)))
    noise = np.random.random(int(sample_rate * duration)) * 0.1
    samples = (signal * 0.5 + noise).astype(np.float32)
    
    return AudioData(
        samples=samples,
        sample_rate=sample_rate,
        duration=duration
    )


@pytest.fixture
def multilingual_audio_samples():
    """Provide audio samples for different languages."""
    def create_audio(freq: float, duration: float = 1.0) -> AudioData:
        sample_rate = 16000
        t = np.linspace(0, duration, int(sample_rate * duration))
        samples = np.sin(2 * np.pi * freq * t).astype(np.float32) * 0.1
        return AudioData(samples=samples, sample_rate=sample_rate, duration=duration)
    
    return {
        "english": create_audio(440.0),    # A4
        "italian": create_audio(523.25),   # C5
        "spanish": create_audio(659.25),   # E5
        "french": create_audio(783.99),    # G5
        "portuguese": create_audio(880.0), # A5
    }


# Mock engine fixtures
@pytest.fixture
def mock_stt_engine():
    """Provide mock STT engine."""
    mock = AsyncMock(spec=STTEngine)
    mock.transcribe.return_value = TranscriptionResult(
        text="Hello, how are you?",
        language="en",
        confidence=0.95,
        chunks=[
            {"text": "Hello,", "start": 0.0, "end": 0.5},
            {"text": "how", "start": 0.6, "end": 0.8},
            {"text": "are", "start": 0.9, "end": 1.1},
            {"text": "you?", "start": 1.2, "end": 1.5}
        ]
    )
    mock.is_available.return_value = True
    return mock


@pytest.fixture
def mock_tts_engine():
    """Provide mock TTS engine."""
    mock = AsyncMock(spec=TTSEngine)
    
    # Create synthetic response audio
    sample_rate = 16000
    duration = 2.0
    samples = np.random.random(int(sample_rate * duration)).astype(np.float32) * 0.1
    
    mock.synthesize.return_value = AudioData(
        samples=samples,
        sample_rate=sample_rate,
        duration=duration
    )
    mock.get_available_voices.return_value = ["af_heart", "af_bella", "af_sarah"]
    mock.is_available.return_value = True
    return mock


@pytest.fixture
def mock_audio_processor():
    """Provide mock audio processor."""
    mock = Mock(spec=AudioProcessor)
    
    def process_audio(audio: AudioData) -> AudioData:
        # Simple mock processing - just return slightly modified audio
        processed_samples = audio.samples * 0.95  # Slight amplitude reduction
        return AudioData(
            samples=processed_samples,
            sample_rate=audio.sample_rate,
            duration=audio.duration
        )
    
    mock.process.side_effect = process_audio
    mock.is_available.return_value = True
    return mock


@pytest.fixture
def mock_language_detector():
    """Provide mock language detector."""
    mock = Mock(spec=LanguageDetector)
    
    # Default to English with high confidence
    mock.detect_language.return_value = ("a", 0.95)  # American English
    mock.is_available.return_value = True
    
    return mock


@pytest.fixture
def mock_memory_manager():
    """Provide mock memory manager."""
    mock = AsyncMock(spec=MemoryManager)
    
    # Mock memory responses
    mock.get_user_context.return_value = "User's name is John. Likes coffee and technology."
    mock.add_memory.return_value = "memory_id_123"
    mock.search_memories.return_value = "Found relevant memories about user preferences."
    mock.clear_memory.return_value = True
    mock.is_available.return_value = True
    
    return mock


@pytest.fixture
def mock_llm_service():
    """Provide mock LLM service."""
    mock = AsyncMock(spec=LLMService)
    
    # Mock LLM responses
    mock.get_response.return_value = "Hello! I'm doing well, thank you for asking. How can I help you today?"
    mock.health_check.return_value = True
    mock.is_available.return_value = True
    
    return mock


# Parametrized fixtures for testing multiple scenarios
@pytest.fixture(params=["en", "it", "es", "fr", "pt"])
def language_code(request):
    """Parametrized fixture for different language codes."""
    return request.param


@pytest.fixture(params=[0.1, 0.5, 1.0, 2.0, 5.0])
def audio_duration(request):
    """Parametrized fixture for different audio durations."""
    return request.param


@pytest.fixture(params=[8000, 16000, 22050, 44100])
def sample_rate(request):
    """Parametrized fixture for different sample rates."""
    return request.param


# Test data generators
@pytest.fixture
def create_test_audio():
    """Factory fixture for creating test audio data."""
    def _create_audio(
        duration: float = 1.0,
        sample_rate: int = 16000,
        frequency: float = 440.0,
        amplitude: float = 0.1,
        noise_level: float = 0.0
    ) -> AudioData:
        """Create test audio with specified parameters."""
        t = np.linspace(0, duration, int(sample_rate * duration))
        signal = np.sin(2 * np.pi * frequency * t) * amplitude
        
        if noise_level > 0:
            noise = np.random.random(len(signal)) * noise_level
            signal += noise
        
        return AudioData(
            samples=signal.astype(np.float32),
            sample_rate=sample_rate,
            duration=duration
        )
    
    return _create_audio


@pytest.fixture
def create_transcription_result():
    """Factory fixture for creating transcription results."""
    def _create_result(
        text: str = "Test transcription",
        language: str = "en",
        confidence: float = 0.95,
        include_chunks: bool = False
    ) -> TranscriptionResult:
        """Create test transcription result."""
        chunks = None
        if include_chunks:
            words = text.split()
            chunks = []
            start_time = 0.0
            for word in words:
                end_time = start_time + 0.5
                chunks.append({
                    "text": word,
                    "start": start_time,
                    "end": end_time
                })
                start_time = end_time + 0.1
        
        return TranscriptionResult(
            text=text,
            language=language,
            confidence=confidence,
            chunks=chunks
        )
    
    return _create_result


# Error simulation fixtures
@pytest.fixture
def failing_stt_engine():
    """Provide STT engine that fails for testing error handling."""
    mock = AsyncMock(spec=STTEngine)
    mock.transcribe.side_effect = Exception("STT service unavailable")
    mock.is_available.return_value = False
    return mock


@pytest.fixture
def failing_tts_engine():
    """Provide TTS engine that fails for testing error handling."""
    mock = AsyncMock(spec=TTSEngine)
    mock.synthesize.side_effect = Exception("TTS service unavailable")
    mock.get_available_voices.return_value = []
    mock.is_available.return_value = False
    return mock


@pytest.fixture
def failing_memory_manager():
    """Provide memory manager that fails for testing error handling."""
    mock = AsyncMock(spec=MemoryManager)
    mock.add_memory.side_effect = Exception("Memory service unavailable")
    mock.search_memories.side_effect = Exception("Memory search failed")
    mock.get_user_context.return_value = ""
    mock.is_available.return_value = False
    return mock


# Performance testing fixtures
@pytest.fixture
def performance_timer():
    """Provide performance timing utility."""
    import time
    
    class Timer:
        def __init__(self):
            self.start_time = None
            self.end_time = None
        
        def start(self):
            self.start_time = time.time()
        
        def stop(self):
            self.end_time = time.time()
        
        @property
        def elapsed(self) -> float:
            if self.start_time and self.end_time:
                return self.end_time - self.start_time
            return 0.0
        
        def __enter__(self):
            self.start()
            return self
        
        def __exit__(self, exc_type, exc_val, exc_tb):
            self.stop()
    
    return Timer


# Cleanup fixtures
@pytest.fixture(autouse=True)
def cleanup_after_test():
    """Automatic cleanup after each test."""
    yield
    # Cleanup code here if needed
    # For example, clearing any global state, temp files, etc.
    pass


# Marks for test categorization
pytest.mark.unit = pytest.mark.unit
pytest.mark.integration = pytest.mark.integration
pytest.mark.performance = pytest.mark.performance
pytest.mark.slow = pytest.mark.slow