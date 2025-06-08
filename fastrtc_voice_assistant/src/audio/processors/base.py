"""Base audio processor implementation."""

from abc import ABC, abstractmethod
from typing import Dict, Any
import numpy as np
import time
from collections import deque

from ...core.interfaces import AudioProcessor, AudioData
from ...core.exceptions import AudioProcessingError


class BaseAudioProcessor(AudioProcessor):
    """Base implementation for audio processors with common functionality."""
    
    def __init__(self):
        """Initialize base audio processor."""
        self.stats = {
            'frames_processed': 0,
            'total_samples': 0,
            'processing_time': 0.0,
            'last_processed': None
        }
        self.processing_history = deque(maxlen=100)
    
    def process(self, audio: AudioData) -> AudioData:
        """Process audio data with timing and statistics tracking.
        
        Args:
            audio: Input audio data
            
        Returns:
            AudioData: Processed audio data
            
        Raises:
            AudioProcessingError: If processing fails
        """
        start_time = time.time()
        
        try:
            # Delegate to specific implementation
            processed_audio = self._process_audio(audio)
            
            # Update statistics
            processing_time = time.time() - start_time
            self._update_stats(audio, processing_time)
            
            return processed_audio
            
        except Exception as e:
            raise AudioProcessingError(f"Audio processing failed: {e}") from e
    
    @abstractmethod
    def _process_audio(self, audio: AudioData) -> AudioData:
        """Implement specific audio processing logic.
        
        Args:
            audio: Input audio data
            
        Returns:
            AudioData: Processed audio data
        """
        pass
    
    def _update_stats(self, audio: AudioData, processing_time: float) -> None:
        """Update processing statistics.
        
        Args:
            audio: Processed audio data
            processing_time: Time taken for processing
        """
        self.stats['frames_processed'] += 1
        self.stats['total_samples'] += len(audio.samples)
        self.stats['processing_time'] += processing_time
        self.stats['last_processed'] = time.time()
        
        # Store processing history
        self.processing_history.append({
            'timestamp': time.time(),
            'samples': len(audio.samples),
            'duration': audio.duration,
            'processing_time': processing_time
        })
    
    def get_stats(self) -> Dict[str, Any]:
        """Get processing statistics.
        
        Returns:
            Dict[str, Any]: Processing statistics
        """
        avg_processing_time = (
            self.stats['processing_time'] / self.stats['frames_processed']
            if self.stats['frames_processed'] > 0 else 0.0
        )
        
        return {
            **self.stats,
            'avg_processing_time': avg_processing_time,
            'samples_per_second': (
                self.stats['total_samples'] / self.stats['processing_time']
                if self.stats['processing_time'] > 0 else 0.0
            )
        }
    
    def reset_stats(self) -> None:
        """Reset processing statistics."""
        self.stats = {
            'frames_processed': 0,
            'total_samples': 0,
            'processing_time': 0.0,
            'last_processed': None
        }
        self.processing_history.clear()
    
    def is_available(self) -> bool:
        """Check if the audio processor is available and ready.
        
        Returns:
            bool: True if processor is ready, False otherwise
        """
        return True  # Base implementation is always available