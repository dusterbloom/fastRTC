"""Speech-to-Text engines for FastRTC Voice Assistant."""

from .huggingface_stt import HuggingFaceSTTEngine
from .base import BaseSTTEngine

__all__ = [
    'HuggingFaceSTTEngine',
    'BaseSTTEngine'
]