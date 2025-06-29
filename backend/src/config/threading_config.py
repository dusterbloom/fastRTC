"""
Threading Pipeline Configuration

Configuration settings for the new threading-based pipeline.
Allows running both old and new systems in parallel during transition.
"""

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class ThreadingPipelineConfig:
    """Configuration for threading-based pipeline."""
    
    # Feature flags
    enabled: bool = True
    fallback_to_async: bool = True  # Fallback to async pipeline if threading fails
    
    # Performance settings
    max_queue_size: int = 100
    processing_timeout: float = 0.1
    generation_timeout: float = 30.0
    cleanup_interval: float = 10.0
    
    # STT worker settings
    stt_confidence_threshold: float = 0.6
    stt_min_audio_length: float = 0.5  # seconds
    
    # LLM worker settings  
    llm_min_sentence_length: int = 10  # characters
    llm_sentence_endings: tuple = ('.', '!', '?', '...')
    
    # TTS worker settings
    tts_chunk_size: int = 1024
    
    # Output settings
    output_timeout: float = 0.05
    
    @classmethod
    def from_env(cls) -> 'ThreadingPipelineConfig':
        """Create configuration from environment variables."""
        return cls(
            enabled=os.getenv('USE_THREADING_PIPELINE', 'false').lower() == 'true',
            fallback_to_async=os.getenv('THREADING_FALLBACK_TO_ASYNC', 'true').lower() == 'true',
            max_queue_size=int(os.getenv('THREADING_MAX_QUEUE_SIZE', '100')),
            processing_timeout=float(os.getenv('THREADING_PROCESSING_TIMEOUT', '0.1')),
            generation_timeout=float(os.getenv('THREADING_GENERATION_TIMEOUT', '30.0')),
            cleanup_interval=float(os.getenv('THREADING_CLEANUP_INTERVAL', '10.0')),
            stt_confidence_threshold=float(os.getenv('THREADING_STT_CONFIDENCE', '0.6')),
            stt_min_audio_length=float(os.getenv('THREADING_STT_MIN_AUDIO', '0.5')),
            llm_min_sentence_length=int(os.getenv('THREADING_LLM_MIN_SENTENCE', '10')),
            tts_chunk_size=int(os.getenv('THREADING_TTS_CHUNK_SIZE', '1024')),
            output_timeout=float(os.getenv('THREADING_OUTPUT_TIMEOUT', '0.05')),
        )


# Global configuration instance
threading_config = ThreadingPipelineConfig.from_env()


def is_threading_enabled() -> bool:
    """Check if threading pipeline is enabled."""
    return threading_config.enabled


def should_fallback_to_async() -> bool:
    """Check if should fallback to async pipeline on threading failure."""
    return threading_config.fallback_to_async