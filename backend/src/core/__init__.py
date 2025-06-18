"""Core components for FastRTC Voice Assistant."""

from .interfaces import (
    STTEngine,
    TTSEngine,
    AudioProcessor,
    LanguageDetector,
    MemoryManager,
    LLMService,
    TranscriptionResult,
    AudioData,
)
from .exceptions import (
    VoiceAssistantError,
    AudioProcessingError,
    STTError,
    TTSError,
    MemoryError,
    LLMError,
)
# Note: VoiceAssistant and main imports removed to avoid circular imports
# Import these directly when needed: from src.core.voice_assistant import VoiceAssistant
# from .voice_assistant import VoiceAssistant
# from .main import VoiceAssistantApplication, create_application, main

__all__ = [
    # Interfaces
    "STTEngine",
    "TTSEngine",
    "AudioProcessor",
    "LanguageDetector",
    "MemoryManager",
    "LLMService",
    # Data classes
    "TranscriptionResult",
    "AudioData",
    # Exceptions
    "VoiceAssistantError",
    "AudioProcessingError",
    "STTError",
    "TTSError",
    "MemoryError",
    "LLMError",
    # Core components (removed to avoid circular imports)
    # "VoiceAssistant",
    # "VoiceAssistantApplication",
    # "create_application",
    # "main",
]