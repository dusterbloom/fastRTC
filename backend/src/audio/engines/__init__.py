"""Audio engines for FastRTC Voice Assistant.

This module provides speech-to-text and text-to-speech engines.
"""

from .stt.faster_whisper_stt import FasterWhisperSTT
# from .stt.huggingface_stt import HuggingFaceSTTEngine  # Disabled due to scipy/sklearn compatibility
from .tts.kokoro_tts import KokoroTTSEngine

__all__ = [
    # 'HuggingFaceSTTEngine',  # Disabled due to scipy/sklearn compatibility
    'FasterWhisperSTT',
    'KokoroTTSEngine'
]