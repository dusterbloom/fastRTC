"""Base STT engine implementation."""

from abc import ABC, abstractmethod
import asyncio
import time
from typing import Dict, Any

from ....core.interfaces import STTEngine, AudioData, TranscriptionResult
from ....core.exceptions import STTError
from ....utils.logging import get_logger

logger = get_logger(__name__)


class BaseSTTEngine(STTEngine):
    """Base implementation for STT engines with common functionality."""
    
    def __init__(self):
        """Initialize base STT engine."""
        self.stats = {
            'transcriptions': 0,
            'total_audio_duration': 0.0,
            'total_processing_time': 0.0,
            'errors': 0,
            'last_transcription': None
        }
        self._is_available = False
    
    async def transcribe(self, audio: AudioData) -> TranscriptionResult:
        """Transcribe audio to text with timing and error handling.
        
        Args:
            audio: Audio data to transcribe
            
        Returns:
            TranscriptionResult: Transcription with metadata
            
        Raises:
            STTError: If transcription fails
        """
        if not self.is_available():
            raise STTError("STT engine is not available")
        
        start_time = time.time()
        
        try:
            # Delegate to specific implementation
            result = await self._transcribe_audio(audio)
            
            # Update statistics
            processing_time = time.time() - start_time
            self._update_stats(audio, processing_time, success=True)
            
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            self._update_stats(audio, processing_time, success=False)
            raise STTError(f"Transcription failed: {e}") from e
    
    @abstractmethod
    async def _transcribe_audio(self, audio: AudioData) -> TranscriptionResult:
        """Implement specific transcription logic.
        
        Args:
            audio: Audio data to transcribe
            
        Returns:
            TranscriptionResult: Transcription result
        """
        pass
    
    def _update_stats(self, audio: AudioData, processing_time: float, success: bool) -> None:
        """Update transcription statistics.
        
        Args:
            audio: Transcribed audio data
            processing_time: Time taken for transcription
            success: Whether transcription was successful
        """
        if success:
            self.stats['transcriptions'] += 1
            self.stats['total_audio_duration'] += audio.duration
            self.stats['last_transcription'] = time.time()
        else:
            self.stats['errors'] += 1
        
        self.stats['total_processing_time'] += processing_time
    
    def get_stats(self) -> Dict[str, Any]:
        """Get transcription statistics.
        
        Returns:
            Dict[str, Any]: Transcription statistics
        """
        avg_processing_time = (
            self.stats['total_processing_time'] / (self.stats['transcriptions'] + self.stats['errors'])
            if (self.stats['transcriptions'] + self.stats['errors']) > 0 else 0.0
        )
        
        real_time_factor = (
            self.stats['total_processing_time'] / self.stats['total_audio_duration']
            if self.stats['total_audio_duration'] > 0 else 0.0
        )
        
        return {
            **self.stats,
            'avg_processing_time': avg_processing_time,
            'real_time_factor': real_time_factor,
            'success_rate': (
                self.stats['transcriptions'] / (self.stats['transcriptions'] + self.stats['errors'])
                if (self.stats['transcriptions'] + self.stats['errors']) > 0 else 0.0
            )
        }
    
    def reset_stats(self) -> None:
        """Reset transcription statistics."""
        self.stats = {
            'transcriptions': 0,
            'total_audio_duration': 0.0,
            'total_processing_time': 0.0,
            'errors': 0,
            'last_transcription': None
        }
    
    def is_available(self) -> bool:
        """Check if the STT engine is available and ready.
        
        Returns:
            bool: True if engine is ready, False otherwise
        """
        return self._is_available
    
    def _set_available(self, available: bool) -> None:
        """Set availability status.
        
        Args:
            available: Whether the engine is available
        """
        self._is_available = available
        if available:
            logger.info(f"{self.__class__.__name__} is now available")
        else:
            logger.warning(f"{self.__class__.__name__} is not available")