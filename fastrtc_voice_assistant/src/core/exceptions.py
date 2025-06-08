"""Custom exception hierarchy for FastRTC Voice Assistant.

This module defines a comprehensive exception hierarchy that provides
clear error boundaries and enables proper error handling throughout
the voice assistant system.
"""

from typing import Optional, Any, Dict


class VoiceAssistantError(Exception):
    """Base exception for all voice assistant errors.
    
    This is the root exception class that all other voice assistant
    exceptions inherit from. It provides common functionality for
    error tracking and debugging.
    """
    
    def __init__(
        self, 
        message: str, 
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        original_exception: Optional[Exception] = None
    ):
        """Initialize voice assistant error.
        
        Args:
            message: Human-readable error message
            error_code: Machine-readable error code (optional)
            details: Additional error details (optional)
            original_exception: Original exception that caused this error (optional)
        """
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        self.original_exception = original_exception
    
    def __str__(self) -> str:
        """String representation of the error."""
        parts = [self.message]
        
        if self.error_code:
            parts.append(f"[{self.error_code}]")
        
        if self.details:
            details_str = ", ".join(f"{k}={v}" for k, v in self.details.items())
            parts.append(f"({details_str})")
        
        return " ".join(parts)


class AudioProcessingError(VoiceAssistantError):
    """Exception raised during audio processing operations.
    
    This includes errors in audio format conversion, noise reduction,
    normalization, and other audio preprocessing tasks.
    """
    
    def __init__(
        self,
        message: str,
        audio_info: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """Initialize audio processing error.
        
        Args:
            message: Error message
            audio_info: Information about the audio that caused the error
            **kwargs: Additional arguments passed to parent class
        """
        details = kwargs.get('details', {})
        if audio_info:
            details.update(audio_info)
        kwargs['details'] = details
        
        super().__init__(message, **kwargs)


class STTError(VoiceAssistantError):
    """Exception raised during speech-to-text operations.
    
    This includes errors in audio transcription, model loading,
    and communication with STT services.
    """
    
    def __init__(
        self,
        message: str,
        model_info: Optional[str] = None,
        audio_duration: Optional[float] = None,
        **kwargs
    ):
        """Initialize STT error.
        
        Args:
            message: Error message
            model_info: Information about the STT model
            audio_duration: Duration of audio that failed to transcribe
            **kwargs: Additional arguments passed to parent class
        """
        details = kwargs.get('details', {})
        if model_info:
            details['model'] = model_info
        if audio_duration is not None:
            details['audio_duration'] = audio_duration
        kwargs['details'] = details
        
        super().__init__(message, **kwargs)


class TTSError(VoiceAssistantError):
    """Exception raised during text-to-speech operations.
    
    This includes errors in speech synthesis, voice selection,
    and communication with TTS services.
    """
    
    def __init__(
        self,
        message: str,
        voice: Optional[str] = None,
        language: Optional[str] = None,
        text_length: Optional[int] = None,
        **kwargs
    ):
        """Initialize TTS error.
        
        Args:
            message: Error message
            voice: Voice that was being used
            language: Language that was being synthesized
            text_length: Length of text that failed to synthesize
            **kwargs: Additional arguments passed to parent class
        """
        details = kwargs.get('details', {})
        if voice:
            details['voice'] = voice
        if language:
            details['language'] = language
        if text_length is not None:
            details['text_length'] = text_length
        kwargs['details'] = details
        
        super().__init__(message, **kwargs)


class MemoryError(VoiceAssistantError):
    """Exception raised during memory operations.
    
    This includes errors in memory storage, retrieval, search,
    and communication with memory services like A-MEM.
    """
    
    def __init__(
        self,
        message: str,
        operation: Optional[str] = None,
        user_id: Optional[str] = None,
        **kwargs
    ):
        """Initialize memory error.
        
        Args:
            message: Error message
            operation: Memory operation that failed (e.g., 'add', 'search', 'retrieve')
            user_id: User ID associated with the memory operation
            **kwargs: Additional arguments passed to parent class
        """
        details = kwargs.get('details', {})
        if operation:
            details['operation'] = operation
        if user_id:
            details['user_id'] = user_id
        kwargs['details'] = details
        
        super().__init__(message, **kwargs)


class LLMError(VoiceAssistantError):
    """Exception raised during LLM operations.
    
    This includes errors in LLM communication, response generation,
    and service availability for both Ollama and LM Studio.
    """
    
    def __init__(
        self,
        message: str,
        service: Optional[str] = None,
        model: Optional[str] = None,
        response_code: Optional[int] = None,
        **kwargs
    ):
        """Initialize LLM error.
        
        Args:
            message: Error message
            service: LLM service that failed (e.g., 'ollama', 'lm_studio')
            model: Model that was being used
            response_code: HTTP response code if applicable
            **kwargs: Additional arguments passed to parent class
        """
        details = kwargs.get('details', {})
        if service:
            details['service'] = service
        if model:
            details['model'] = model
        if response_code is not None:
            details['response_code'] = response_code
        kwargs['details'] = details
        
        super().__init__(message, **kwargs)


class LanguageDetectionError(VoiceAssistantError):
    """Exception raised during language detection operations.
    
    This includes errors in language detection from text,
    model loading, and confidence scoring.
    """
    
    def __init__(
        self,
        message: str,
        text_sample: Optional[str] = None,
        detector_type: Optional[str] = None,
        **kwargs
    ):
        """Initialize language detection error.
        
        Args:
            message: Error message
            text_sample: Sample of text that failed detection (truncated for privacy)
            detector_type: Type of detector that failed (e.g., 'mediapipe', 'keyword')
            **kwargs: Additional arguments passed to parent class
        """
        details = kwargs.get('details', {})
        if text_sample:
            # Truncate text sample for privacy and log size
            details['text_sample'] = text_sample[:50] + "..." if len(text_sample) > 50 else text_sample
        if detector_type:
            details['detector_type'] = detector_type
        kwargs['details'] = details
        
        super().__init__(message, **kwargs)


class ConfigurationError(VoiceAssistantError):
    """Exception raised for configuration-related errors.
    
    This includes errors in configuration loading, validation,
    and missing required settings.
    """
    
    def __init__(
        self,
        message: str,
        config_key: Optional[str] = None,
        config_value: Optional[Any] = None,
        **kwargs
    ):
        """Initialize configuration error.
        
        Args:
            message: Error message
            config_key: Configuration key that caused the error
            config_value: Configuration value that caused the error
            **kwargs: Additional arguments passed to parent class
        """
        details = kwargs.get('details', {})
        if config_key:
            details['config_key'] = config_key
        if config_value is not None:
            details['config_value'] = str(config_value)
        kwargs['details'] = details
        
        super().__init__(message, **kwargs)


class FastRTCError(VoiceAssistantError):
    """Exception raised during FastRTC integration operations.
    
    This includes errors in stream handling, callback processing,
    and FastRTC service communication.
    """
    
    def __init__(
        self,
        message: str,
        stream_id: Optional[str] = None,
        callback_type: Optional[str] = None,
        **kwargs
    ):
        """Initialize FastRTC error.
        
        Args:
            message: Error message
            stream_id: Stream ID that caused the error
            callback_type: Type of callback that failed
            **kwargs: Additional arguments passed to parent class
        """
        details = kwargs.get('details', {})
        if stream_id:
            details['stream_id'] = stream_id
        if callback_type:
            details['callback_type'] = callback_type
        kwargs['details'] = details
        
        super().__init__(message, **kwargs)


# Convenience functions for creating common errors

def create_audio_error(message: str, audio_data: Optional[Any] = None) -> AudioProcessingError:
    """Create an audio processing error with audio metadata.
    
    Args:
        message: Error message
        audio_data: Audio data that caused the error
        
    Returns:
        AudioProcessingError: Configured error instance
    """
    audio_info = {}
    if hasattr(audio_data, 'sample_rate'):
        audio_info['sample_rate'] = audio_data.sample_rate
    if hasattr(audio_data, 'duration'):
        audio_info['duration'] = audio_data.duration
    if hasattr(audio_data, 'samples') and hasattr(audio_data.samples, 'shape'):
        audio_info['shape'] = audio_data.samples.shape
    
    return AudioProcessingError(message, audio_info=audio_info)


def create_service_unavailable_error(service_name: str, details: Optional[str] = None) -> VoiceAssistantError:
    """Create a service unavailable error.
    
    Args:
        service_name: Name of the unavailable service
        details: Additional details about the unavailability
        
    Returns:
        VoiceAssistantError: Configured error instance
    """
    message = f"{service_name} service is unavailable"
    if details:
        message += f": {details}"
    
    return VoiceAssistantError(
        message,
        error_code="SERVICE_UNAVAILABLE",
        details={"service": service_name}
    )