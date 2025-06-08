"""Configuration management for FastRTC Voice Assistant.

This module provides dataclass-based configuration management extracted from
the original monolithic implementation.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class AudioConfig:
    """Audio processing configuration."""
    sample_rate: int = 16000
    chunk_duration: float = 2.0
    noise_threshold: float = 0.15
    minimal_silent_frame_duration_ms: int = 20
    
    # Extracted from original configuration
    hf_model_id: str = field(
        default_factory=lambda: os.getenv("HF_MODEL_ID", "openai/whisper-large-v3")
    )


@dataclass
class MemoryConfig:
    """Memory system configuration for A-MEM integration."""
    # Extracted from original A-MEM settings
    llm_model: str = field(
        default_factory=lambda: os.getenv("AMEM_LLM_MODEL", "llama3.2:3b")
    )
    embedder_model: str = field(
        default_factory=lambda: os.getenv("AMEM_EMBEDDER_MODEL", "nomic-embed-text")
    )
    evolution_threshold: int = 50
    cache_ttl_seconds: int = 180


@dataclass
class LLMConfig:
    """LLM service configuration."""
    # Extracted from original configuration
    use_ollama: bool = True  # USE_OLLAMA_FOR_CONVERSATION
    ollama_url: str = field(
        default_factory=lambda: os.getenv("OLLAMA_URL", "http://localhost:11434")
    )
    ollama_model: str = field(
        default_factory=lambda: os.getenv("OLLAMA_CONVERSATIONAL_MODEL", "llama3:8b-instruct-q4_K_M")
    )
    
    # LM Studio settings (fallback when use_ollama is False)
    lm_studio_url: str = field(
        default_factory=lambda: os.getenv("LM_STUDIO_URL", "http://192.168.1.5:1234/v1")
    )
    lm_studio_model: str = field(
        default_factory=lambda: os.getenv("LM_STUDIO_MODEL", "mistral-nemo-instruct-2407")
    )


@dataclass
class TTSConfig:
    """Text-to-Speech configuration for Kokoro TTS."""
    # Extracted from original Kokoro configuration
    preferred_voice: str = "af_heart"  # KOKORO_PREFERRED_VOICE
    fallback_voices: List[str] = field(
        default_factory=lambda: ["af_alloy", "af_bella"]  # KOKORO_FALLBACK_VOICE_1, _2
    )
    speed: float = 1.05
    default_language: str = "a"  # DEFAULT_LANGUAGE (American English)


@dataclass
class AppConfig:
    """Main application configuration container."""
    audio: AudioConfig = field(default_factory=AudioConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)
    
    def __post_init__(self):
        """Post-initialization validation and setup."""
        # Ensure fallback voices are set
        if not self.tts.fallback_voices:
            self.tts.fallback_voices = ["af_alloy", "af_bella"]


def load_config() -> AppConfig:
    """Load application configuration from environment variables and defaults.
    
    Returns:
        AppConfig: Fully configured application settings
    """
    return AppConfig()


def get_default_config() -> AppConfig:
    """Get default configuration for testing and development.
    
    Returns:
        AppConfig: Default configuration with safe values
    """
    return AppConfig(
        audio=AudioConfig(),
        memory=MemoryConfig(),
        llm=LLMConfig(),
        tts=TTSConfig()
    )


# Global constants extracted from original implementation
DEFAULT_LANGUAGE = 'a'  # American English
USE_OLLAMA_FOR_CONVERSATION = True
OLLAMA_CONVERSATIONAL_MODEL = "llama3:8b-instruct-q4_K_M"
OLLAMA_URL = "http://localhost:11434"
LM_STUDIO_MODEL = "mistral-nemo-instruct-2407"
LM_STUDIO_URL = "http://192.168.1.5:1234/v1"
AMEM_LLM_MODEL = "llama3.2:3b"
AMEM_EMBEDDER_MODEL = "nomic-embed-text"
HF_MODEL_ID = "openai/whisper-large-v3"
DEFAULT_SPEECH_THRESHOLD = 0.15