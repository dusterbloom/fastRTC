"""
WhisperLive Streaming Worker

Specialized worker for WhisperLive real-time transcription in threading pipeline.
Optimized for streaming audio processing with minimal latency.
"""

import time
import asyncio
import threading
import numpy as np
from typing import Optional, Any, Dict
from queue import Queue, Empty

from .pipeline_workers import BasePipelineWorker
from .pipeline_manager import (
    AudioPipelineManager, AudioChunk, TranscriptionChunk, GenerationStatus
)
from ..audio.engines.stt.whisper_live_stt import WhisperLiveSTTEngine
from ..utils.logging import get_logger

logger = get_logger(__name__)


class WhisperLiveStreamingWorker(BasePipelineWorker):
    """
    Specialized worker for WhisperLive real-time streaming transcription.
    
    Optimized for continuous audio streaming with minimal buffering
    and real-time transcription delivery.
    """
    
    def __init__(
        self,
        pipeline_manager: AudioPipelineManager,
        server_host: str = "localhost",
        server_port: int = 9090,
        model: str = "small",
        language: str = "None",
        confidence_threshold: float = 0.6,
        min_audio_length: float = 1.0,  # Shorter for real-time
        processing_timeout: float = 0.05,  # Faster processing
        chunk_duration: float = 0.5,  # 500ms chunks for streaming
        use_vad: bool = False,
        auto_start_server: bool = True,
        server_backend: str = "faster_whisper"
    ):
        super().__init__(
            name="WhisperLiveWorker",
            pipeline_manager=pipeline_manager,
            input_queue=pipeline_manager.audio_input_queue,
            output_queue=pipeline_manager.transcription_queue,
            processing_timeout=processing_timeout
        )
        
        # WhisperLive configuration
        self.server_host = server_host
        self.server_port = server_port
        self.model = model
        self.language = language
        self.confidence_threshold = confidence_threshold
        self.min_audio_length = min_audio_length
        self.chunk_duration = chunk_duration
        self.use_vad = use_vad
        self.auto_start_server = auto_start_server
        self.server_backend = server_backend
        
        # WhisperLive engine
        self.whisper_live_engine: Optional[WhisperLiveSTTEngine] = None
        
        # Audio streaming management
        self.audio_buffers = {}  # generation_id -> audio buffer
        self.stream_threads = {}  # generation_id -> streaming thread
        self.active_streams = set()  # active generation IDs
        
        # Sentence buffering for partial results
        self.sentence_buffers = {}  # generation_id -> partial text
        self.buffer_timestamps = {}  # generation_id -> last update time
        self.buffer_timeout = 2.0  # seconds
        self.max_buffer_length = 300  # characters
        
        # Performance tracking
        self.stream_count = 0
        self.total_latency = 0.0
        
        logger.info(f"🎤 WhisperLive streaming worker initialized (server: {server_host}:{server_port})")
    
    async def initialize(self):
        """Initialize WhisperLive engine and start worker."""
        try:
            # Initialize WhisperLive engine
            self.whisper_live_engine = WhisperLiveSTTEngine(
                server_host=self.server_host,
                server_port=self.server_port,
                model=self.model,
                language=self.language,
                use_vad=self.use_vad,
                auto_start_server=self.auto_start_server,
                server_backend=self.server_backend
            )
            
            # Initialize the engine
            if not await self.whisper_live_engine.initialize():
                raise RuntimeError("Failed to initialize WhisperLive engine")
            
            logger.info("✅ WhisperLive streaming worker initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize WhisperLive worker: {e}")
            return False
    
    async def process_audio_chunk(self, audio_chunk: AudioChunk) -> bool:
        """
        Process audio chunk with WhisperLive streaming.
        
        Args:
            audio_chunk: Audio chunk to process
            
        Returns:
            bool: True if processing started successfully
        """
        generation_id = audio_chunk.generation_id
        state = self.pipeline_manager.get_generation_state(generation_id)
        
        if not state:
            logger.warning(f"No state found for generation {generation_id}")
            return False
        
        try:
            logger.info(f"🎤 WhisperLive worker processing audio chunk for generation {generation_id}")
            
            # Initialize generation state
            if state.stt_start_time is None:
                state.stt_start_time = time.time()
                state.status = GenerationStatus.STT_PROCESSING
            
            # Add to audio buffer
            if generation_id not in self.audio_buffers:
                self.audio_buffers[generation_id] = []
                self.sentence_buffers[generation_id] = ""
                self.buffer_timestamps[generation_id] = time.time()
            
            self.audio_buffers[generation_id].append(audio_chunk.audio_data)
            
            # Check if we have enough audio for processing
            total_duration = len(self.audio_buffers[generation_id]) * self.chunk_duration
            
            if total_duration >= self.min_audio_length:
                # Start streaming transcription if not already active
                if generation_id not in self.active_streams:
                    await self._start_streaming_transcription(generation_id)
                else:
                    # Send new audio chunk to existing stream
                    await self._send_audio_to_stream(generation_id, audio_chunk.audio_data)
            
            return True
            
        except Exception as e:
            logger.error(f"WhisperLive processing error for generation {generation_id}: {e}")
            state.mark_failed(f"WhisperLive error: {e}")
            return False
    
    async def _start_streaming_transcription(self, generation_id: str):
        """Start streaming transcription for a generation."""
        try:
            self.active_streams.add(generation_id)
            
            # Combine buffered audio
            audio_buffer = self.audio_buffers[generation_id]
            combined_audio = np.concatenate(audio_buffer) if audio_buffer else np.array([])
            
            # Start streaming thread
            stream_thread = threading.Thread(
                target=self._streaming_transcription_thread,
                args=(generation_id, combined_audio),
                daemon=True
            )
            self.stream_threads[generation_id] = stream_thread
            stream_thread.start()
            
            logger.info(f"🎤 Started WhisperLive streaming for generation {generation_id}")
            
        except Exception as e:
            logger.error(f"Failed to start streaming transcription: {e}")
            self.active_streams.discard(generation_id)
    
    def _streaming_transcription_thread(self, generation_id: str, initial_audio: np.ndarray):
        """Thread for handling streaming transcription."""
        try:
            # Set up streaming transcription with WhisperLive
            # Note: This is a simplified implementation
            # In practice, you'd use WhisperLive's streaming capabilities
            
            # For now, process the audio chunk
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                result = loop.run_until_complete(
                    self.whisper_live_engine._transcribe_audio(initial_audio)
                )
                
                # Send result to pipeline
                if result and result.text.strip():
                    self._send_transcription_result(generation_id, result.text, is_final=True)
                
            finally:
                loop.close()
                
        except Exception as e:
            logger.error(f"Streaming transcription thread error: {e}")
        finally:
            # Clean up
            self.active_streams.discard(generation_id)
            self.stream_threads.pop(generation_id, None)
    
    async def _send_audio_to_stream(self, generation_id: str, audio_data: np.ndarray):
        """Send additional audio data to existing stream."""
        # In a full implementation, this would send audio to the streaming transcription
        # For now, we'll accumulate and re-process
        pass
    
    def _send_transcription_result(self, generation_id: str, text: str, is_final: bool = False):
        """Send transcription result to pipeline."""
        try:
            state = self.pipeline_manager.get_generation_state(generation_id)
            if not state:
                return
            
            # Update sentence buffer
            current_time = time.time()
            
            if is_final or self._should_flush_buffer(generation_id, text, current_time):
                # Send final transcription
                transcription_chunk = TranscriptionChunk(
                    generation_id=generation_id,
                    text=text,
                    confidence=0.9,  # WhisperLive doesn't provide confidence
                    is_final=is_final,
                    timestamp=current_time,
                    metadata={
                        'engine': 'whisper-live',
                        'model': self.model,
                        'backend': self.server_backend
                    }
                )
                
                # Add to output queue
                try:
                    self.output_queue.put_nowait(transcription_chunk)
                    logger.info(f"🎤 WhisperLive transcription sent: '{text[:50]}...'")
                except Exception as e:
                    logger.error(f"Failed to send transcription: {e}")
                
                # Mark STT as complete if final
                if is_final:
                    state.stt_complete.set()
                    state.transcription_text = text
                    logger.info(f"🎤 WhisperLive completed for generation {generation_id}")
                
                # Clear buffer
                self.sentence_buffers[generation_id] = ""
                self.buffer_timestamps[generation_id] = current_time
            else:
                # Update buffer
                self.sentence_buffers[generation_id] = text
                self.buffer_timestamps[generation_id] = current_time
                
        except Exception as e:
            logger.error(f"Error sending transcription result: {e}")
    
    def _should_flush_buffer(self, generation_id: str, text: str, current_time: float) -> bool:
        """Determine if sentence buffer should be flushed."""
        # Flush if buffer timeout exceeded
        last_update = self.buffer_timestamps.get(generation_id, 0)
        if current_time - last_update > self.buffer_timeout:
            return True
        
        # Flush if buffer too long
        if len(text) > self.max_buffer_length:
            return True
        
        # Flush if sentence appears complete (ends with punctuation)
        if text.strip().endswith(('.', '!', '?')):
            return True
        
        return False
    
    async def cleanup(self):
        """Clean up WhisperLive resources."""
        try:
            # Stop all active streams
            for generation_id in list(self.active_streams):
                self.active_streams.discard(generation_id)
            
            # Wait for streaming threads to finish
            for thread in self.stream_threads.values():
                if thread.is_alive():
                    thread.join(timeout=2.0)
            
            # Clean up WhisperLive engine
            if self.whisper_live_engine:
                await self.whisper_live_engine.cleanup()
                self.whisper_live_engine = None
            
            # Clear buffers
            self.audio_buffers.clear()
            self.sentence_buffers.clear()
            self.buffer_timestamps.clear()
            self.stream_threads.clear()
            
            logger.info("🧹 WhisperLive streaming worker cleaned up")
            
        except Exception as e:
            logger.error(f"Error during WhisperLive worker cleanup: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get worker statistics."""
        stats = super().get_stats()
        stats.update({
            'worker_type': 'whisper-live-streaming',
            'server_host': self.server_host,
            'server_port': self.server_port,
            'model': self.model,
            'backend': self.server_backend,
            'active_streams': len(self.active_streams),
            'stream_count': self.stream_count,
            'avg_latency': self.total_latency / max(self.stream_count, 1)
        })
        return stats
    
    def process_item(self, item: Any) -> Any:
        """Process item (not used in async worker)."""
        logger.warning("process_item called on WhisperLiveStreamingWorker - use async processing")
        return None