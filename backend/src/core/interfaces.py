"""Abstract interfaces for FastRTC Voice Assistant components.

This module defines the abstract base classes and data structures that form
the contracts for all voice assistant components. These interfaces enable
dependency injection, testing, and modular architecture.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from dataclasses import dataclass


@dataclass
class TranscriptionResult:
    """Result from speech-to-text transcription.
    
    Attributes:
        text: Transcribed text
        language: Detected language code (optional)
        confidence: Confidence score 0.0-1.0 (optional)
        chunks: Detailed transcription chunks with timestamps (optional)
    """
    text: str
    language: Optional[str] = None
    confidence: Optional[float] = None
    chunks: Optional[List[Dict]] = None


@dataclass
class AudioData:
    """Container for audio data with metadata.
    
    Attributes:
        samples: Audio samples as numpy array
        sample_rate: Sample rate in Hz
        duration: Duration in seconds
    """
    samples: np.ndarray
    sample_rate: int
    duration: float
    
    def __post_init__(self):
        """Validate audio data consistency."""
        expected_samples = int(self.sample_rate * self.duration)
        actual_samples = len(self.samples)
        
        # Allow small discrepancies due to rounding
        if abs(expected_samples - actual_samples) > self.sample_rate * 0.1:  # 100ms tolerance
            raise ValueError(
                f"Audio data inconsistent: expected ~{expected_samples} samples "
                f"for {self.duration}s at {self.sample_rate}Hz, got {actual_samples}"
            )


class STTEngine(ABC):
    """Abstract base class for Speech-to-Text engines.
    
    Implementations should handle audio transcription with language detection
    and provide confidence scores when available.
    """
    
    @abstractmethod
    async def transcribe(self, audio: AudioData) -> TranscriptionResult:
        """Transcribe audio to text with language detection.
        
        Args:
            audio: Audio data to transcribe
            
        Returns:
            TranscriptionResult: Transcription with metadata
            
        Raises:
            STTError: If transcription fails
        """
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the STT engine is available and ready.
        
        Returns:
            bool: True if engine is ready, False otherwise
        """
        pass


class TTSEngine(ABC):
    """Abstract base class for Text-to-Speech engines.
    
    Implementations should support multiple voices and languages,
    with voice selection based on language preferences.
    """
    
    @abstractmethod
    async def synthesize(self, text: str, voice: str, language: str) -> AudioData:
        """Synthesize text to speech.
        
        Args:
            text: Text to synthesize
            voice: Voice identifier
            language: Language code
            
        Returns:
            AudioData: Synthesized audio
            
        Raises:
            TTSError: If synthesis fails
        """
        pass
    
    @abstractmethod
    def get_available_voices(self, language: str) -> List[str]:
        """Get available voices for a language.
        
        Args:
            language: Language code
            
        Returns:
            List[str]: Available voice identifiers
        """
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the TTS engine is available and ready.
        
        Returns:
            bool: True if engine is ready, False otherwise
        """
        pass


class AudioProcessor(ABC):
    """Abstract base class for audio processing.
    
    Implementations should handle noise reduction, normalization,
    and other audio preprocessing tasks.
    """
    
    @abstractmethod
    def process(self, audio: AudioData) -> AudioData:
        """Process audio data (noise reduction, normalization, etc.).
        
        Args:
            audio: Input audio data
            
        Returns:
            AudioData: Processed audio data
            
        Raises:
            AudioProcessingError: If processing fails
        """
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the audio processor is available and ready.
        
        Returns:
            bool: True if processor is ready, False otherwise
        """
        pass


class LanguageDetector(ABC):
    """Abstract base class for language detection.
    
    Implementations should detect language from text and provide
    confidence scores for the detection.
    """
    
    @abstractmethod
    def detect_language(self, text: str) -> Tuple[str, float]:
        """Detect language from text.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Tuple[str, float]: (language_code, confidence_score)
            
        Note:
            Language codes should be compatible with the TTS system.
            Confidence scores should be in range 0.0-1.0.
        """
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the language detector is available and ready.
        
        Returns:
            bool: True if detector is ready, False otherwise
        """
        pass


class MemoryManager(ABC):
    """Abstract base class for memory management.
    
    Implementations should handle conversation memory, user context,
    and integration with memory systems like A-MEM.
    """
    
    @abstractmethod
    async def add_memory(self, user_text: str, assistant_text: str) -> Optional[str]:
        """Add a conversation turn to memory.
        
        Args:
            user_text: User's input text
            assistant_text: Assistant's response text
            
        Returns:
            Optional[str]: Memory ID if successful, None otherwise
            
        Raises:
            MemoryError: If memory storage fails
        """
        pass
    
    @abstractmethod
    async def search_memories(self, query: str) -> str:
        """Search memories for relevant information.
        
        Args:
            query: Search query
            
        Returns:
            str: Relevant memory information
            
        Raises:
            MemoryError: If memory search fails
        """
        pass
    
    @abstractmethod
    def get_user_context(self) -> str:
        """Get current user context from memory.
        
        Returns:
            str: User context information
        """
        pass
    
    @abstractmethod
    async def clear_memory(self) -> bool:
        """Clear all memory for the current user.
        
        Returns:
            bool: True if successful, False otherwise
        """
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the memory manager is available and ready.
        
        Returns:
            bool: True if manager is ready, False otherwise
        """
        pass


class LLMService(ABC):
    """Abstract base class for Large Language Model services.
    
    Implementations should handle conversation with LLMs,
    supporting both Ollama and LM Studio backends.
    """
    
    @abstractmethod
    async def get_response(self, user_text: str, context: str) -> str:
        """Get LLM response to user input.
        
        Args:
            user_text: User's input text
            context: Conversation context from memory
            
        Returns:
            str: LLM response text
            
        Raises:
            LLMError: If LLM request fails
        """
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        """Check if the LLM service is healthy and responsive.
        
        Returns:
            bool: True if service is healthy, False otherwise
        """
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the LLM service is available and ready.
        
        Returns:
            bool: True if service is ready, False otherwise
        """
        pass


class AsyncLifecycleManager(ABC):
    """Abstract base class for managing async component lifecycles.
    
    Implementations should handle startup, shutdown, and health monitoring
    of async components.
    """
    
    @abstractmethod
    async def startup(self) -> bool:
        """Start all managed components.
        
        Returns:
            bool: True if all components started successfully
        """
        pass
    
    @abstractmethod
    async def shutdown(self) -> bool:
        """Shutdown all managed components gracefully.
        
        Returns:
            bool: True if all components shut down successfully
        """
        pass
    
    @abstractmethod
    async def health_check(self) -> Dict[str, bool]:
        """Check health of all managed components.
        
        Returns:
            Dict[str, bool]: Component name to health status mapping
        """
        pass