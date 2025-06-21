"""Speech-to-Text engines for FastRTC Voice Assistant."""

import os
from .base import BaseSTTEngine

# Check STT backend configuration (defaults to existing engine)
STT_BACKEND = os.environ.get('STT_BACKEND', 'faster').lower()

if STT_BACKEND == "slower":
    from .huggingface_stt import HuggingFaceSTTEngine as STTEngine
else:
    from .faster_whisper_stt import FasterWhisperSTT as STTEngine

__all__ = [
    'STTEngine',
    'BaseSTTEngine'
]