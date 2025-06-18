"""
Stub implementation of KokoroONNX for testing and fallback purposes.

This module provides a mock/stub implementation of the KokoroONNX class
that can be used when the actual kokoro-onnx package is not available.
"""

import numpy as np
from typing import Generator, Tuple, Any
from dataclasses import dataclass


@dataclass
class KokoroTTSOptions:
    """Options for Kokoro TTS synthesis."""
    speed: float = 1.0
    lang: str = "en-us"
    voice: str = None


class MockKokoroModel:
    """Mock model class to simulate the Kokoro model structure."""
    
    def __init__(self):
        self.voices = [
            "af", "af_bella", "af_sarah", "af_nicole",
            "am_adam", "am_michael", "am_edward", "am_lewis",
            "bf_emma", "bf_isabella", "bf_amy", "bf_jenny",
            "bm_george", "bm_lewis", "bm_daniel", "bm_william"
        ]


class KokoroONNX:
    """
    Stub implementation of KokoroONNX for testing purposes.
    
    This class provides the same interface as the real KokoroONNX class
    but generates synthetic audio data instead of actual TTS synthesis.
    """
    
    def __init__(self):
        """Initialize the mock Kokoro ONNX model."""
        self.model = MockKokoroModel()
        self.sample_rate = 24000
    
    def stream_tts_sync(self, text: str, options: Any) -> Generator[Tuple[int, np.ndarray], None, None]:
        """
        Mock streaming TTS synthesis that generates synthetic audio.
        
        Args:
            text: Text to synthesize
            options: TTS options (KokoroTTSOptions or similar)
            
        Yields:
            Tuple[int, np.ndarray]: (sample_rate, audio_chunk)
        """
        # Calculate approximate audio length based on text
        # Rough estimate: 150 words per minute, 5 chars per word average
        chars_per_second = (150 * 5) / 60  # ~12.5 chars/second
        estimated_duration = max(0.5, len(text) / chars_per_second)
        
        # Generate synthetic audio in chunks
        chunk_duration = 0.5  # 0.5 second chunks
        total_chunks = max(1, int(estimated_duration / chunk_duration))
        
        for chunk_idx in range(total_chunks):
            # Generate synthetic audio chunk (sine wave with some variation)
            chunk_samples = int(self.sample_rate * chunk_duration)
            
            # Create a simple synthetic speech-like signal
            t = np.linspace(0, chunk_duration, chunk_samples)
            
            # Base frequency varies with chunk to simulate speech patterns
            base_freq = 200 + (chunk_idx % 3) * 50  # 200-300 Hz range
            
            # Generate synthetic speech-like waveform
            signal = (
                0.3 * np.sin(2 * np.pi * base_freq * t) +
                0.2 * np.sin(2 * np.pi * base_freq * 1.5 * t) +
                0.1 * np.sin(2 * np.pi * base_freq * 2.0 * t)
            )
            
            # Add some envelope to make it more speech-like
            envelope = np.exp(-t * 2) * (1 - np.exp(-t * 10))
            signal *= envelope
            
            # Add some noise for realism
            noise = np.random.normal(0, 0.02, chunk_samples)
            signal += noise
            
            # Ensure proper data type and range
            audio_chunk = np.clip(signal, -1.0, 1.0).astype(np.float32)
            
            yield (self.sample_rate, audio_chunk)
    
    def synthesize(self, text: str, options: Any = None) -> Tuple[int, np.ndarray]:
        """
        Mock non-streaming synthesis.
        
        Args:
            text: Text to synthesize
            options: TTS options
            
        Returns:
            Tuple[int, np.ndarray]: (sample_rate, complete_audio)
        """
        # Collect all chunks from streaming synthesis
        chunks = []
        for sample_rate, chunk in self.stream_tts_sync(text, options):
            chunks.append(chunk)
        
        if chunks:
            # Concatenate all chunks
            complete_audio = np.concatenate(chunks)
            return (self.sample_rate, complete_audio)
        else:
            # Return minimal audio if no chunks generated
            minimal_audio = np.zeros(int(self.sample_rate * 0.1), dtype=np.float32)
            return (self.sample_rate, minimal_audio)