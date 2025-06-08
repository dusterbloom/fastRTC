"""Audio configuration constants and utilities.

This module contains audio-related configuration extracted from the original
monolithic implementation.
"""

import numpy as np
from typing import Tuple

# Audio configuration constants (extracted from original)
AUDIO_SAMPLE_RATE = 16000
MINIMAL_SILENT_FRAME_DURATION_MS = 20

# Calculated constants based on sample rate
MINIMAL_SILENT_SAMPLES = int(AUDIO_SAMPLE_RATE * (MINIMAL_SILENT_FRAME_DURATION_MS / 1000.0))

# Pre-computed silent audio arrays for efficiency
SILENT_AUDIO_CHUNK_ARRAY = np.zeros(MINIMAL_SILENT_SAMPLES, dtype=np.float32)
SILENT_AUDIO_FRAME_TUPLE = (AUDIO_SAMPLE_RATE, SILENT_AUDIO_CHUNK_ARRAY)

# Audio processing constants
DEFAULT_CHUNK_DURATION = 2.0  # seconds
DEFAULT_NOISE_THRESHOLD = 0.15
DEFAULT_AUDIO_FORMAT = np.float32


def create_silent_audio(duration_ms: int) -> np.ndarray:
    """Create a silent audio array of specified duration.
    
    Args:
        duration_ms: Duration in milliseconds
        
    Returns:
        np.ndarray: Silent audio array
    """
    samples = int(AUDIO_SAMPLE_RATE * (duration_ms / 1000.0))
    return np.zeros(samples, dtype=DEFAULT_AUDIO_FORMAT)


def get_silent_frame_tuple() -> Tuple[int, np.ndarray]:
    """Get the pre-computed silent audio frame tuple.
    
    Returns:
        Tuple[int, np.ndarray]: (sample_rate, silent_audio_array)
    """
    return SILENT_AUDIO_FRAME_TUPLE


def calculate_samples_from_duration(duration_seconds: float) -> int:
    """Calculate number of samples for given duration.
    
    Args:
        duration_seconds: Duration in seconds
        
    Returns:
        int: Number of samples
    """
    return int(AUDIO_SAMPLE_RATE * duration_seconds)


def calculate_duration_from_samples(samples: int) -> float:
    """Calculate duration from number of samples.
    
    Args:
        samples: Number of audio samples
        
    Returns:
        float: Duration in seconds
    """
    return samples / AUDIO_SAMPLE_RATE


def validate_audio_array(audio: np.ndarray) -> bool:
    """Validate audio array format and values.
    
    Args:
        audio: Audio array to validate
        
    Returns:
        bool: True if valid, False otherwise
    """
    if not isinstance(audio, np.ndarray):
        return False
    
    if audio.dtype != DEFAULT_AUDIO_FORMAT:
        return False
    
    if len(audio.shape) != 1:  # Should be 1D array
        return False
    
    # Check for reasonable amplitude range
    if np.max(np.abs(audio)) > 10.0:  # Reasonable upper bound
        return False
    
    return True