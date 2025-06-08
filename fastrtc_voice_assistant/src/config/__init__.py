"""Configuration management for FastRTC Voice Assistant."""

from .settings import AppConfig, AudioConfig, MemoryConfig, LLMConfig, TTSConfig
from .language_config import WHISPER_TO_KOKORO_LANG, KOKORO_VOICE_MAP, KOKORO_TTS_LANG_MAP
from .audio_config import AUDIO_SAMPLE_RATE, MINIMAL_SILENT_FRAME_DURATION_MS

__all__ = [
    "AppConfig",
    "AudioConfig", 
    "MemoryConfig",
    "LLMConfig",
    "TTSConfig",
    "WHISPER_TO_KOKORO_LANG",
    "KOKORO_VOICE_MAP", 
    "KOKORO_TTS_LANG_MAP",
    "AUDIO_SAMPLE_RATE",
    "MINIMAL_SILENT_FRAME_DURATION_MS",
]