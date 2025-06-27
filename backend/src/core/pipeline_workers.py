"""
Pipeline Worker Threads

Base classes and implementations for threading-based pipeline workers.
Each worker handles a specific stage of the audio processing pipeline.
"""

import time
import logging
import threading
from abc import ABC, abstractmethod
from queue import Queue, Empty
from typing import Optional, Any

from .pipeline_manager import (
    AudioPipelineManager, GenerationState, GenerationStatus,
    AudioChunk, TranscriptionChunk, LLMTokenChunk, TTSAudioChunk
)
from ..utils.logging import get_logger

logger = get_logger(__name__)


class BasePipelineWorker(threading.Thread, ABC):
    """
    Base class for all pipeline worker threads.
    
    Provides common functionality for queue processing, error handling,
    and coordination with the pipeline manager.
    """
    
    def __init__(
        self,
        name: str,
        pipeline_manager: AudioPipelineManager,
        input_queue: Queue,
        output_queue: Optional[Queue] = None,
        processing_timeout: float = 0.1
    ):
        super().__init__(name=name, daemon=True)
        self.pipeline_manager = pipeline_manager
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.processing_timeout = processing_timeout
        
        # Worker control
        self.running = False
        self.stop_requested = False
        
        # Metrics
        self.processed_items = 0
        self.failed_items = 0
        self.total_processing_time = 0.0
        self.start_time = None
        
        # Register with pipeline manager
        self.pipeline_manager.register_worker(self)
        
    def start_worker(self):
        """Start the worker thread."""
        if self.running:
            logger.warning(f"Worker {self.name} already running")
            return
            
        logger.info(f"🚀 Starting worker: {self.name}")
        self.running = True
        self.stop_requested = False
        self.start_time = time.time()
        self.start()
        
    def stop_worker(self, timeout: float = 5.0):
        """Stop the worker thread."""
        if not self.running:
            return
            
        logger.info(f"🛑 Stopping worker: {self.name}")
        self.stop_requested = True
        
        # Wait for thread to finish
        if self.is_alive():
            self.join(timeout=timeout)
            if self.is_alive():
                logger.warning(f"Worker {self.name} did not stop within timeout")
                
        self.running = False
        
    def run(self):
        """Main worker thread loop."""
        logger.info(f"✅ Worker {self.name} started")
        
        try:
            while not self.stop_requested and not self.pipeline_manager.stop_event.is_set():
                try:
                    # Get item from input queue
                    item = self.input_queue.get(timeout=self.processing_timeout)
                    
                    # Check if generation is still active
                    if hasattr(item, 'generation_id'):
                        state = self.pipeline_manager.get_generation_state(item.generation_id)
                        if not state or state.interrupted.is_set():
                            logger.debug(f"Skipping processing for interrupted generation {item.generation_id}")
                            continue
                    
                    # Process the item
                    start_time = time.time()
                    try:
                        result = self.process_item(item)
                        processing_time = time.time() - start_time
                        
                        self.processed_items += 1
                        self.total_processing_time += processing_time
                        
                        # Send result to output queue if successful
                        if result is not None and self.output_queue is not None:
                            self.output_queue.put(result)
                            
                    except Exception as e:
                        self.failed_items += 1
                        self.handle_processing_error(item, e)
                        
                except Empty:
                    # No item available, continue
                    continue
                except Exception as e:
                    logger.error(f"Unexpected error in worker {self.name}: {e}")
                    
        except Exception as e:
            logger.error(f"Fatal error in worker {self.name}: {e}")
        finally:
            logger.info(f"🏁 Worker {self.name} stopped")
            
    @abstractmethod
    def process_item(self, item: Any) -> Optional[Any]:
        """
        Process a single item from the input queue.
        
        Args:
            item: The item to process
            
        Returns:
            The processed result, or None if no output should be sent
        """
        pass
        
    def handle_processing_error(self, item: Any, error: Exception):
        """
        Handle processing errors.
        
        Args:
            item: The item that failed to process
            error: The exception that occurred
        """
        logger.error(f"Processing error in {self.name}: {error}")
        
        # Mark generation as failed if applicable
        if hasattr(item, 'generation_id'):
            state = self.pipeline_manager.get_generation_state(item.generation_id)
            if state:
                state.mark_failed(str(error))
                
    def get_worker_stats(self) -> dict:
        """Get worker statistics."""
        runtime = time.time() - self.start_time if self.start_time else 0
        avg_processing_time = (
            self.total_processing_time / self.processed_items 
            if self.processed_items > 0 else 0
        )
        
        return {
            "name": self.name,
            "running": self.running,
            "runtime_seconds": runtime,
            "processed_items": self.processed_items,
            "failed_items": self.failed_items,
            "average_processing_time": avg_processing_time,
            "input_queue_size": self.input_queue.qsize(),
            "output_queue_size": self.output_queue.qsize() if self.output_queue else 0,
        }


class InterruptionManager:
    """
    Manages interruption coordination across the pipeline.
    
    Handles user interruptions, timeout-based interruptions,
    and cleanup of interrupted generations.
    """
    
    def __init__(self, pipeline_manager: AudioPipelineManager):
        self.pipeline_manager = pipeline_manager
        self.interruption_callbacks = []
        
    def register_interruption_callback(self, callback):
        """Register a callback to be called on interruption."""
        self.interruption_callbacks.append(callback)
        
    def handle_user_speech_detected(self):
        """Handle detection of new user speech - interrupt current generation."""
        logger.info("🎤 User speech detected - interrupting current generation")
        
        # Interrupt all active generations
        self.pipeline_manager.interrupt_all_active()
        
        # Call interruption callbacks
        for callback in self.interruption_callbacks:
            try:
                callback()
            except Exception as e:
                logger.error(f"Error in interruption callback: {e}")
                
    def handle_generation_timeout(self, generation_id: int):
        """Handle timeout for a specific generation."""
        logger.warning(f"⏰ Generation {generation_id} timed out")
        
        state = self.pipeline_manager.get_generation_state(generation_id)
        if state:
            state.mark_failed("Generation timeout")
            
    def cleanup_interrupted_queues(self):
        """Clean up queues from interrupted generations."""
        # This will be called periodically to remove stale queue items
        # Implementation depends on how we want to handle queue cleanup
        pass


class QueueMonitor:
    """
    Monitors queue sizes and performance metrics.
    
    Provides alerts when queues are backing up or workers are falling behind.
    """
    
    def __init__(self, pipeline_manager: AudioPipelineManager):
        self.pipeline_manager = pipeline_manager
        self.alert_thresholds = {
            "queue_size_warning": 50,
            "queue_size_critical": 80,
            "processing_time_warning": 1.0,  # seconds
        }
        
    def check_queue_health(self) -> dict:
        """Check health of all queues and return status."""
        stats = self.pipeline_manager.get_pipeline_stats()
        queue_sizes = stats["queue_sizes"]
        
        alerts = []
        
        for queue_name, size in queue_sizes.items():
            if size >= self.alert_thresholds["queue_size_critical"]:
                alerts.append(f"CRITICAL: {queue_name} queue size {size}")
            elif size >= self.alert_thresholds["queue_size_warning"]:
                alerts.append(f"WARNING: {queue_name} queue size {size}")
                
        return {
            "healthy": len(alerts) == 0,
            "alerts": alerts,
            "queue_sizes": queue_sizes,
        }
        
    def get_performance_summary(self) -> dict:
        """Get overall performance summary."""
        stats = self.pipeline_manager.get_pipeline_stats()
        
        return {
            "pipeline_stats": stats,
            "queue_health": self.check_queue_health(),
        }