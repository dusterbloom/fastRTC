"""
Audio Pipeline Manager

Pure threading-based pipeline manager for real-time audio processing.
Replaces sync/async bridging with event-driven worker threads and queues.

Inspired by RealtimeVoiceChat architecture but adapted for fastRTC modularity.
"""

import time
import logging
import threading
from queue import Queue, Empty
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import numpy as np

from ..utils.logging import get_logger

logger = get_logger(__name__)


class GenerationStatus(Enum):
    """Status of a generation request."""
    PENDING = "pending"
    STT_PROCESSING = "stt_processing" 
    LLM_STREAMING = "llm_streaming"
    TTS_SYNTHESIZING = "tts_synthesizing"
    COMPLETED = "completed"
    INTERRUPTED = "interrupted"
    FAILED = "failed"


@dataclass
class AudioChunk:
    """Audio data chunk for processing."""
    generation_id: int
    sample_rate: int
    audio_data: np.ndarray
    timestamp: float
    is_final: bool = False


@dataclass 
class TranscriptionChunk:
    """STT transcription result."""
    generation_id: int
    text: str
    confidence: float
    is_partial: bool = False
    timestamp: float = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()


@dataclass
class LLMTokenChunk:
    """LLM token/sentence chunk."""
    generation_id: int
    text: str
    is_sentence_complete: bool = False
    timestamp: float = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()


@dataclass
class TTSAudioChunk:
    """TTS synthesized audio chunk."""
    generation_id: int
    sample_rate: int
    audio_data: np.ndarray
    timestamp: float = None
    is_final: bool = False
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()


class GenerationState:
    """Tracks state of a single generation request."""
    
    def __init__(self, generation_id: int):
        self.id = generation_id
        self.status = GenerationStatus.PENDING
        self.created_at = time.time()
        
        # Threading events for coordination
        self.stt_complete = threading.Event()
        self.llm_started = threading.Event()
        self.llm_streaming = threading.Event()
        self.tts_started = threading.Event()
        self.interrupted = threading.Event()
        self.completed = threading.Event()
        
        # Data storage
        self.input_text = ""
        self.response_text = ""
        self.error_message = ""
        
        # Metrics
        self.stt_start_time = None
        self.llm_start_time = None
        self.tts_start_time = None
        self.first_audio_time = None
        
    def mark_interrupted(self):
        """Mark this generation as interrupted."""
        self.status = GenerationStatus.INTERRUPTED
        self.interrupted.set()
        logger.info(f"🛑 Generation {self.id} marked as interrupted")
        
    def mark_completed(self):
        """Mark this generation as completed."""
        self.status = GenerationStatus.COMPLETED
        self.completed.set()
        logger.info(f"✅ Generation {self.id} completed")
        
    def mark_failed(self, error: str):
        """Mark this generation as failed."""
        self.status = GenerationStatus.FAILED
        self.error_message = error
        logger.error(f"❌ Generation {self.id} failed: {error}")


class AudioPipelineManager:
    """
    Central manager for threading-based audio processing pipeline.
    
    Coordinates worker threads using queues and events, eliminating 
    sync/async complexity while preserving modular design.
    """
    
    def __init__(
        self,
        max_queue_size: int = 100,
        generation_timeout: float = 30.0,
        cleanup_interval: float = 10.0
    ):
        # Configuration
        self.max_queue_size = max_queue_size
        self.generation_timeout = generation_timeout
        self.cleanup_interval = cleanup_interval
        
        # Threading control
        self.stop_event = threading.Event()
        self.running = False
        
        # Pipeline queues
        self.audio_input_queue = Queue(maxsize=max_queue_size)
        self.transcription_queue = Queue(maxsize=max_queue_size)
        self.llm_token_queue = Queue(maxsize=max_queue_size)
        self.tts_audio_queue = Queue(maxsize=max_queue_size)
        self.output_queue = Queue(maxsize=max_queue_size)
        
        # Generation tracking
        self.generation_counter = 0
        self.generation_states: Dict[int, GenerationState] = {}
        self.state_lock = threading.Lock()
        
        # Worker threads (will be set by workers)
        self.workers = []
        
        # Metrics
        self.total_generations = 0
        self.completed_generations = 0
        self.interrupted_generations = 0
        self.failed_generations = 0
        
        # Cleanup thread
        self.cleanup_thread = None
        
    def start(self):
        """Start the pipeline manager."""
        if self.running:
            logger.warning("Pipeline manager already running")
            return
            
        logger.info("🚀 Starting AudioPipelineManager")
        self.running = True
        self.stop_event.clear()
        
        # Start cleanup thread
        self.cleanup_thread = threading.Thread(
            target=self._cleanup_worker,
            name="PipelineCleanup",
            daemon=True
        )
        self.cleanup_thread.start()
        
        logger.info("✅ AudioPipelineManager started")
        
    def stop(self):
        """Stop the pipeline manager and all workers."""
        if not self.running:
            return
            
        logger.info("🛑 Stopping AudioPipelineManager")
        self.running = False
        self.stop_event.set()
        
        # Interrupt all active generations
        with self.state_lock:
            for state in self.generation_states.values():
                if state.status not in [GenerationStatus.COMPLETED, GenerationStatus.FAILED]:
                    state.mark_interrupted()
        
        # Wait for cleanup thread
        if self.cleanup_thread and self.cleanup_thread.is_alive():
            self.cleanup_thread.join(timeout=5.0)
            
        logger.info("✅ AudioPipelineManager stopped")
        
    def create_generation(self) -> int:
        """Create a new generation and return its ID."""
        with self.state_lock:
            self.generation_counter += 1
            generation_id = self.generation_counter
            self.generation_states[generation_id] = GenerationState(generation_id)
            self.total_generations += 1
            
        logger.debug(f"🆕 Created generation {generation_id}")
        return generation_id
        
    def get_generation_state(self, generation_id: int) -> Optional[GenerationState]:
        """Get generation state by ID."""
        with self.state_lock:
            return self.generation_states.get(generation_id)
            
    def interrupt_generation(self, generation_id: int) -> bool:
        """Interrupt a specific generation."""
        with self.state_lock:
            state = self.generation_states.get(generation_id)
            if state and state.status not in [GenerationStatus.COMPLETED, GenerationStatus.FAILED]:
                state.mark_interrupted()
                self.interrupted_generations += 1
                return True
        return False
        
    def interrupt_all_active(self):
        """Interrupt all active generations."""
        with self.state_lock:
            for state in self.generation_states.values():
                if state.status not in [GenerationStatus.COMPLETED, GenerationStatus.FAILED, GenerationStatus.INTERRUPTED]:
                    state.mark_interrupted()
                    self.interrupted_generations += 1
        logger.info("🛑 Interrupted all active generations")
        
    def put_audio_input(self, audio_chunk: AudioChunk, timeout: float = 1.0) -> bool:
        """Put audio chunk into input queue."""
        try:
            self.audio_input_queue.put(audio_chunk, timeout=timeout)
            return True
        except Exception as e:
            logger.error(f"Failed to queue audio input: {e}")
            return False
            
    def get_output_audio(self, timeout: float = 0.1) -> Optional[TTSAudioChunk]:
        """Get output audio chunk."""
        try:
            return self.output_queue.get(timeout=timeout)
        except Empty:
            return None
            
    def register_worker(self, worker):
        """Register a worker thread."""
        self.workers.append(worker)
        logger.debug(f"Registered worker: {worker.__class__.__name__}")
        
    def get_pipeline_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics."""
        with self.state_lock:
            active_generations = sum(
                1 for state in self.generation_states.values()
                if state.status not in [GenerationStatus.COMPLETED, GenerationStatus.FAILED, GenerationStatus.INTERRUPTED]
            )
            
        return {
            "running": self.running,
            "total_generations": self.total_generations,
            "completed_generations": self.completed_generations,
            "interrupted_generations": self.interrupted_generations,
            "failed_generations": self.failed_generations,
            "active_generations": active_generations,
            "queue_sizes": {
                "audio_input": self.audio_input_queue.qsize(),
                "transcription": self.transcription_queue.qsize(),
                "llm_tokens": self.llm_token_queue.qsize(),
                "tts_audio": self.tts_audio_queue.qsize(),
                "output": self.output_queue.qsize(),
            }
        }
        
    def _cleanup_worker(self):
        """Background thread to cleanup old generations."""
        while not self.stop_event.wait(self.cleanup_interval):
            try:
                self._cleanup_old_generations()
            except Exception as e:
                logger.error(f"Error in cleanup worker: {e}")
                
    def _cleanup_old_generations(self):
        """Remove old completed/failed generations."""
        cutoff_time = time.time() - self.generation_timeout
        removed_count = 0
        
        with self.state_lock:
            to_remove = []
            for gen_id, state in self.generation_states.items():
                if (state.status in [GenerationStatus.COMPLETED, GenerationStatus.FAILED, GenerationStatus.INTERRUPTED] 
                    and state.created_at < cutoff_time):
                    to_remove.append(gen_id)
                    
            for gen_id in to_remove:
                del self.generation_states[gen_id]
                removed_count += 1
                
        if removed_count > 0:
            logger.debug(f"🧹 Cleaned up {removed_count} old generations")