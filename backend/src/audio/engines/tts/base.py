"""Base TTS engine implementation."""

from abc import ABC, abstractmethod
import asyncio
import time
from typing import Dict, Any, List

from ....core.interfaces import TTSEngine, AudioData
from ....core.exceptions import TTSError
from ....utils.logging import get_logger

logger = get_logger(__name__)


class BaseTTSEngine(TTSEngine):
    """Base implementation for TTS engines with common functionality."""
    
    def __init__(self):
        """Initialize base TTS engine."""
        self.stats = {
            'syntheses': 0,
            'total_text_length': 0,
            'total_audio_duration': 0.0,
            'total_processing_time': 0.0,
            'errors': 0,
            'last_synthesis': None
        }
        self._is_available = False
    
    async def synthesize(self, text: str, voice: str, language: str) -> AudioData:
        """Synthesize text to speech with timing and error handling.
        
        Args:
            text: Text to synthesize
            voice: Voice identifier
            language: Language code
            
        Returns:
            AudioData: Synthesized audio
            
        Raises:
            TTSError: If synthesis fails
        """
        if not self.is_available():
            raise TTSError("TTS engine is not available")
        
        if not text or not text.strip():
            raise TTSError("Text cannot be empty")
        
        start_time = time.time()
        
        try:
            # Delegate to specific implementation
            result = await self._synthesize_text(text, voice, language)
            
            # Update statistics
            processing_time = time.time() - start_time
            self._update_stats(text, result, processing_time, success=True)
            
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            self._update_stats(text, None, processing_time, success=False)
            raise TTSError(f"Synthesis failed: {e}") from e
    
    @abstractmethod
    async def _synthesize_text(self, text: str, voice: str, language: str) -> AudioData:
        """Implement specific synthesis logic.
        
        Args:
            text: Text to synthesize
            voice: Voice identifier
            language: Language code
            
        Returns:
            AudioData: Synthesized audio
        """
        pass
    
    def _update_stats(self, text: str, audio: AudioData, processing_time: float, success: bool) -> None:
        """Update synthesis statistics.
        
        Args:
            text: Synthesized text
            audio: Generated audio data (None if failed)
            processing_time: Time taken for synthesis
            success: Whether synthesis was successful
        """
        if success and audio:
            self.stats['syntheses'] += 1
            self.stats['total_text_length'] += len(text)
            self.stats['total_audio_duration'] += audio.duration
            self.stats['last_synthesis'] = time.time()
        else:
            self.stats['errors'] += 1
        
        self.stats['total_processing_time'] += processing_time
    
    def get_stats(self) -> Dict[str, Any]:
        """Get synthesis statistics.
        
        Returns:
            Dict[str, Any]: Synthesis statistics
        """
        avg_processing_time = (
            self.stats['total_processing_time'] / (self.stats['syntheses'] + self.stats['errors'])
            if (self.stats['syntheses'] + self.stats['errors']) > 0 else 0.0
        )
        
        real_time_factor = (
            self.stats['total_processing_time'] / self.stats['total_audio_duration']
            if self.stats['total_audio_duration'] > 0 else 0.0
        )
        
        chars_per_second = (
            self.stats['total_text_length'] / self.stats['total_processing_time']
            if self.stats['total_processing_time'] > 0 else 0.0
        )
        
        return {
            **self.stats,
            'avg_processing_time': avg_processing_time,
            'real_time_factor': real_time_factor,
            'chars_per_second': chars_per_second,
            'success_rate': (
                self.stats['syntheses'] / (self.stats['syntheses'] + self.stats['errors'])
                if (self.stats['syntheses'] + self.stats['errors']) > 0 else 0.0
            )
        }
    
    def reset_stats(self) -> None:
        """Reset synthesis statistics."""
        self.stats = {
            'syntheses': 0,
            'total_text_length': 0,
            'total_audio_duration': 0.0,
            'total_processing_time': 0.0,
            'errors': 0,
            'last_synthesis': None
        }
    
    def is_available(self) -> bool:
        """Check if the TTS engine is available and ready.
        
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