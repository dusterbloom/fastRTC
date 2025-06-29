"""
STT Streaming Worker

Threading-based worker for Speech-to-Text processing.
Uses async operations within thread event loop for FastRTC compatibility.
"""

import time
import asyncio
import numpy as np
from typing import Optional, Any

from .pipeline_workers import BasePipelineWorker
from .pipeline_manager import (
    AudioPipelineManager, AudioChunk, TranscriptionChunk, GenerationStatus
)
from ..audio import STTEngine
from ..utils.logging import get_logger

logger = get_logger(__name__)


class STTStreamingWorker(BasePipelineWorker):
    """
    Worker thread for streaming Speech-to-Text processing.
    
    Processes audio chunks and produces transcription results,
    integrating with existing STT engine infrastructure.
    """
    
    def __init__(
        self,
        pipeline_manager: AudioPipelineManager,
        stt_engine: STTEngine,
        confidence_threshold: float = 0.6,
        min_audio_length: float = 0.5,  # minimum seconds of audio
        processing_timeout: float = 0.1
    ):
        super().__init__(
            name="STTWorker",
            pipeline_manager=pipeline_manager,
            input_queue=pipeline_manager.audio_input_queue,
            output_queue=pipeline_manager.transcription_queue,
            processing_timeout=processing_timeout
        )
        
        self.stt_engine = stt_engine
        self.confidence_threshold = confidence_threshold
        self.min_audio_length = min_audio_length
        
        # Audio buffering for minimum length requirements
        self.audio_buffers = {}  # generation_id -> list of audio chunks
        
        logger.info(f"🎤 STTStreamingWorker initialized with confidence threshold {confidence_threshold}")
        
        # Event loop for async operations in this thread
        self.loop = None
        
    def run(self):
        """Main worker thread loop with async event loop."""
        # Create event loop for this thread
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        
        logger.info(f"✅ Worker {self.name} started with async event loop")
        
        try:
            # Run the async worker loop
            self.loop.run_until_complete(self._async_worker_loop())
        except Exception as e:
            logger.error(f"Fatal error in worker {self.name}: {e}")
        finally:
            if self.loop and not self.loop.is_closed():
                self.loop.close()
            logger.info(f"🏁 Worker {self.name} stopped")
            
    async def _async_worker_loop(self):
        """Async worker loop that processes items from queue."""
        while not self.stop_requested and not self.pipeline_manager.stop_event.is_set():
            try:
                # Get item from input queue (with timeout)
                try:
                    # Use asyncio timeout for queue get
                    item = await asyncio.wait_for(
                        asyncio.to_thread(self.input_queue.get, timeout=self.processing_timeout),
                        timeout=self.processing_timeout + 0.1
                    )
                except asyncio.TimeoutError:
                    continue
                except Exception:
                    continue
                
                # Check if generation is still active
                if hasattr(item, 'generation_id'):
                    state = self.pipeline_manager.get_generation_state(item.generation_id)
                    if not state or state.interrupted.is_set():
                        logger.debug(f"Skipping processing for interrupted generation {item.generation_id}")
                        continue
                
                # Process the item async
                start_time = time.time()
                try:
                    result = await self.process_item_async(item)
                    processing_time = time.time() - start_time
                    
                    self.processed_items += 1
                    self.total_processing_time += processing_time
                    
                    # Send result to output queue if successful
                    if result is not None and self.output_queue is not None:
                        await asyncio.to_thread(self.output_queue.put, result)
                        
                except Exception as e:
                    self.failed_items += 1
                    self.handle_processing_error(item, e)
                    
            except Exception as e:
                logger.error(f"Unexpected error in worker {self.name}: {e}")
                await asyncio.sleep(0.1)  # Brief pause before continuing
        
    async def process_item_async(self, audio_chunk: AudioChunk) -> Optional[TranscriptionChunk]:
        """
        Process audio chunk through STT engine asynchronously.
        
        Args:
            audio_chunk: Audio data to transcribe
            
        Returns:
            TranscriptionChunk with result, or None if no transcription ready
        """
        generation_id = audio_chunk.generation_id
        
        # DEBUG: Log audio chunk received
        print(f"🎤 STT WORKER: Received audio chunk for generation {generation_id}")
        print(f"🎤 STT WORKER: Audio shape: {audio_chunk.audio_data.shape}, dtype: {audio_chunk.audio_data.dtype}")
        logger.info(f"🎤 STT Worker received audio chunk for generation {generation_id}: "
                   f"{audio_chunk.audio_data.size} samples at {audio_chunk.sample_rate}Hz")
        
        # Update generation state
        state = self.pipeline_manager.get_generation_state(generation_id)
        if not state:
            logger.warning(f"No state found for generation {generation_id}")
            return None
            
        if state.stt_start_time is None:
            state.stt_start_time = time.time()
            state.status = GenerationStatus.STT_PROCESSING
            
        # Buffer audio chunks until we have minimum length
        if generation_id not in self.audio_buffers:
            self.audio_buffers[generation_id] = []
            
        self.audio_buffers[generation_id].append(audio_chunk)
        
        # Calculate total buffered audio length
        total_samples = sum(chunk.audio_data.size for chunk in self.audio_buffers[generation_id])
        total_duration = total_samples / audio_chunk.sample_rate
        
        # Check if we have enough audio or this is the final chunk
        if not audio_chunk.is_final and total_duration < self.min_audio_length:
            logger.debug(f"Buffering audio for generation {generation_id}: {total_duration:.2f}s")
            return None
            
        # Combine buffered audio
        combined_audio = self._combine_audio_chunks(self.audio_buffers[generation_id])
        
        # Clean up buffer
        del self.audio_buffers[generation_id]
        
        try:
            # Run STT processing async
            logger.debug(f"🎤 Processing {total_duration:.2f}s of audio for generation {generation_id}")
            
            start_time = time.time()
            transcription_result = await self._transcribe_audio_async(combined_audio, audio_chunk.sample_rate)
            processing_time = time.time() - start_time
            
            if not transcription_result or not transcription_result.text.strip():
                logger.debug(f"No transcription result for generation {generation_id}")
                return None
                
            # Check confidence threshold
            confidence = getattr(transcription_result, 'confidence', 1.0)
            if confidence < self.confidence_threshold:
                logger.debug(f"Low confidence transcription ({confidence:.2f}) for generation {generation_id}")
                return TranscriptionChunk(
                    generation_id=generation_id,
                    text=transcription_result.text,
                    confidence=confidence,
                    is_partial=True
                )
                
            # Mark STT as complete
            state.stt_complete.set()
            state.input_text = transcription_result.text
            
            logger.info(f"🎤 STT completed for generation {generation_id}: '{transcription_result.text}' "
                       f"(confidence: {confidence:.2f}, time: {processing_time:.3f}s)")
            
            return TranscriptionChunk(
                generation_id=generation_id,
                text=transcription_result.text,
                confidence=confidence,
                is_partial=False
            )
            
        except Exception as e:
            logger.error(f"STT processing error for generation {generation_id}: {e}")
            state.mark_failed(f"STT error: {e}")
            return None
            
    def _combine_audio_chunks(self, chunks: list) -> np.ndarray:
        """
        Combine multiple audio chunks into single array.
        
        Args:
            chunks: List of AudioChunk objects
            
        Returns:
            Combined audio data as numpy array
        """
        if not chunks:
            return np.array([], dtype=np.float32)
            
        if len(chunks) == 1:
            return chunks[0].audio_data
            
        # Concatenate all audio data
        audio_arrays = [chunk.audio_data for chunk in chunks]
        combined = np.concatenate(audio_arrays)
        
        logger.debug(f"Combined {len(chunks)} chunks into {combined.size} samples")
        return combined
        
    async def _transcribe_audio_async(self, audio_data: np.ndarray, sample_rate: int):
        """
        Asynchronously transcribe audio using existing STT engine.
        
        Args:
            audio_data: Audio samples
            sample_rate: Sample rate
            
        Returns:
            Transcription result
        """
        try:
            # Ensure audio is 1D mono for faster-whisper
            if audio_data.ndim > 1:
                print(f"🎤 STT: Converting {audio_data.shape} to 1D mono")
                # Take first channel if stereo, or flatten if needed
                audio_data = audio_data.flatten() if audio_data.shape[0] == 1 else audio_data[0]
                
            print(f"🎤 STT: Final audio shape: {audio_data.shape}, dtype: {audio_data.dtype}")
            
            # Convert int16 to float32 format expected by STT engine (like voice_assistant.py does)
            if audio_data.dtype == np.int16:
                print(f"🎤 STT: Converting int16 to float32 with /32768.0 normalization")
                audio_data = audio_data.astype(np.float32) / 32768.0
            elif audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)
                
            print(f"🎤 STT: Final audio range: [{np.min(audio_data):.3f}, {np.max(audio_data):.3f}]")
            
            # Use existing STT engine async methods
            print(f"🎤 STT: Starting transcription with {audio_data.shape} audio...")
            if hasattr(self.stt_engine, 'transcribe_audio_async'):
                # Use async method directly
                print(f"🎤 STT: Using transcribe_audio_async")
                result = await self.stt_engine.transcribe_audio_async(audio_data, sample_rate)
            elif hasattr(self.stt_engine, 'process_audio_async'):
                # Use async process method
                audio_bytes = audio_data.tobytes()
                result = await self.stt_engine.process_audio_async(audio_bytes, sample_rate)
            elif hasattr(self.stt_engine, 'process_audio'):
                # Fallback: run sync method in thread pool
                audio_bytes = audio_data.tobytes()
                result = await asyncio.to_thread(self.stt_engine.process_audio, audio_bytes, sample_rate)
            else:
                # Final fallback: check if transcribe method is async or sync
                if asyncio.iscoroutinefunction(self.stt_engine.transcribe):
                    # Async method - await directly
                    result = await self.stt_engine.transcribe(audio_data)
                else:
                    # Sync method - run in thread pool
                    result = await asyncio.to_thread(self.stt_engine.transcribe, audio_data)
                
            print(f"🎤 STT: Transcription completed! Result: {result}")
            return result
            
        except Exception as e:
            logger.error(f"STT transcription error: {e}")
            raise
            
    def cleanup_buffers(self, generation_id: int):
        """Clean up audio buffers for a specific generation."""
        if generation_id in self.audio_buffers:
            del self.audio_buffers[generation_id]
            logger.debug(f"Cleaned up audio buffer for generation {generation_id}")
            
    def handle_processing_error(self, item, error):
        """Handle STT processing errors."""
        super().handle_processing_error(item, error)
        
        # Clean up buffers for failed generation
        if hasattr(item, 'generation_id'):
            self.cleanup_buffers(item.generation_id)
            
    def get_worker_stats(self) -> dict:
        """Get STT worker statistics."""
        stats = super().get_worker_stats()
        stats.update({
            "confidence_threshold": self.confidence_threshold,
            "min_audio_length": self.min_audio_length,
            "buffered_generations": len(self.audio_buffers),
        })
        return stats
        
    def process_item(self, audio_chunk) -> Optional[Any]:
        """
        Synchronous wrapper for process_item_async.
        Required by BasePipelineWorker abstract method.
        
        Args:
            audio_chunk: Audio chunk data to process
            
        Returns:
            None - this worker uses async processing
        """
        # This method should not be called directly since STTStreamingWorker
        # overrides the worker loop to use async processing
        logger.warning("process_item called on STTStreamingWorker - this should use async processing")
        return None