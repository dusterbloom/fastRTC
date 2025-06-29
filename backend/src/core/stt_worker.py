"""
STT Streaming Worker

Threading-based worker for Speech-to-Text processing.
Uses async operations within thread event loop for FastRTC compatibility.
"""

import time
import asyncio
import numpy as np
from typing import Optional

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
    
    def process_item(self, item):
        """Sync wrapper for async processing (required by base class)."""
        # Run async processing in the current event loop
        if self.loop and self.loop.is_running():
            # If we're already in the event loop, create a task
            task = asyncio.create_task(self.process_item_async(item))
            return self.loop.run_until_complete(task)
        else:
            # If no event loop, run in new loop
            return asyncio.run(self.process_item_async(item))
        
    def run(self):
        """Main worker thread loop with async event loop."""
        # Create event loop for this thread
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        
        logger.info(f"✅ Worker {self.name} started with async event loop")
        logger.info(f"🔍 STT Worker input queue: {self.input_queue}")
        logger.info(f"🔍 STT Worker output queue: {self.output_queue}")
        
        try:
            # Run the async worker loop
            self.loop.run_until_complete(self._async_worker_loop())
        except Exception as e:
            logger.error(f"❌ Fatal error in worker {self.name}: {e}")
            import traceback
            logger.error(f"❌ Fatal error traceback: {traceback.format_exc()}")
        finally:
            if self.loop and not self.loop.is_closed():
                self.loop.close()
            logger.info(f"🏁 Worker {self.name} stopped")
            
    async def _async_worker_loop(self):
        """Async worker loop that processes items from queue."""
        logger.info(f"🔄 STT Worker async loop started")
        
        while not self.stop_requested and not self.pipeline_manager.stop_event.is_set():
            try:
                # Get item from input queue (with timeout)
                try:
                    # Use asyncio timeout for queue get
                    logger.debug(f"🔍 STT Worker waiting for queue item...")
                    item = await asyncio.wait_for(
                        asyncio.to_thread(self.input_queue.get, timeout=self.processing_timeout),
                        timeout=self.processing_timeout + 0.1
                    )
                    logger.info(f"🎤 STT Worker received item: {type(item)} for generation {getattr(item, 'generation_id', 'UNKNOWN')}")
                except asyncio.TimeoutError:
                    logger.debug(f"🔍 STT Worker queue timeout, continuing...")
                    continue
                except Exception as e:
                    logger.warning(f"🔍 STT Worker queue get error: {e}")
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
        
        # DEBUG: Log audio chunk received with detailed analysis
        logger.info(f"🎤 STT Worker received audio chunk for generation {generation_id}: "
                   f"{audio_chunk.audio_data.size} samples at {audio_chunk.sample_rate}Hz")
        
        # CRITICAL DEBUG: Analyze the audio data in detail
        audio_data = audio_chunk.audio_data
        print(f"🔍 [STT WORKER] Audio data analysis:")
        print(f"  - Type: {type(audio_data)}")
        print(f"  - Shape: {audio_data.shape}")
        print(f"  - Size: {audio_data.size}")
        print(f"  - Dtype: {audio_data.dtype}")
        print(f"  - Min: {audio_data.min():.6f}")
        print(f"  - Max: {audio_data.max():.6f}")
        print(f"  - Mean: {audio_data.mean():.6f}")
        print(f"  - RMS: {(audio_data ** 2).mean() ** 0.5:.6f}")
        print(f"  - Non-zero samples: {(audio_data != 0).sum()}")
        print(f"  - Sample rate: {audio_chunk.sample_rate}")
        print(f"  - Duration: {audio_data.size / audio_chunk.sample_rate:.3f}s")
        
        # Check if audio is empty or silent
        if audio_data.size == 0:
            print(f"❌ [STT WORKER] Audio data is EMPTY for generation {generation_id}")
            logger.error(f"❌ [STT WORKER] Audio data is EMPTY for generation {generation_id}")
            return None
            
        if (audio_data == 0).all():
            print(f"❌ [STT WORKER] Audio data is ALL ZEROS for generation {generation_id}")
            logger.error(f"❌ [STT WORKER] Audio data is ALL ZEROS for generation {generation_id}")
            return None
            
        # Check if audio has meaningful content
        audio_rms = (audio_data ** 2).mean() ** 0.5
        if audio_rms < 1e-6:
            print(f"❌ [STT WORKER] Audio data is too quiet (RMS: {audio_rms:.8f}) for generation {generation_id}")
            logger.error(f"❌ [STT WORKER] Audio data is too quiet (RMS: {audio_rms:.8f}) for generation {generation_id}")
            return None
        
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
            logger.info(f"🎤 Processing {total_duration:.2f}s of audio for generation {generation_id}")
            print(f"🔍 [STT WORKER] About to call _transcribe_audio_async for generation {generation_id}")
            print(f"  - Combined audio shape: {combined_audio.shape}")
            print(f"  - Combined audio size: {combined_audio.size}")
            print(f"  - Combined audio RMS: {(combined_audio ** 2).mean() ** 0.5:.6f}")
            
            start_time = time.time()
            transcription_result = await self._transcribe_audio_async(combined_audio, audio_chunk.sample_rate)
            processing_time = time.time() - start_time
            
            print(f"🔍 [STT WORKER] _transcribe_audio_async completed for generation {generation_id}")
            print(f"  - Processing time: {processing_time:.3f}s")
            print(f"  - Result type: {type(transcription_result)}")
            if transcription_result:
                print(f"  - Result text: '{getattr(transcription_result, 'text', 'NO_TEXT_ATTR')}'")
                print(f"  - Result confidence: {getattr(transcription_result, 'confidence', 'NO_CONFIDENCE_ATTR')}")
            
            if not transcription_result:
                logger.warning(f"❌ STT returned None for generation {generation_id}")
                return None
            
            if not transcription_result.text or not transcription_result.text.strip():
                logger.warning(f"❌ STT returned empty text for generation {generation_id}: '{transcription_result.text}'")
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
            logger.error(f"❌ STT processing error for generation {generation_id}: {e}")
            logger.error(f"❌ STT error type: {type(e).__name__}")
            logger.error(f"❌ STT error details: {str(e)}")
            import traceback
            logger.error(f"❌ STT traceback: {traceback.format_exc()}")
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
            print(f"🔍 [STT WORKER] _transcribe_audio_async called")
            print(f"  - Audio data shape: {audio_data.shape}")
            print(f"  - Audio data dtype: {audio_data.dtype}")
            print(f"  - Sample rate: {sample_rate}")
            
            # Convert audio data to bytes format expected by STT engine
            logger.debug(f"🔧 Converting audio data: dtype={audio_data.dtype}, shape={audio_data.shape}")
            
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)
                print(f"🔧 Converted audio to float32")
                
            # Normalize if needed
            max_val = np.max(np.abs(audio_data))
            if max_val > 1.0:
                audio_data = audio_data / max_val
                logger.debug(f"🔧 Normalized audio data by factor {max_val}")
                print(f"🔧 Normalized audio data by factor {max_val}")
            
            # Flatten 2D audio first (handle WebRTC format)
            if audio_data.ndim > 1:
                original_shape = audio_data.shape
                audio_data = audio_data.flatten()
                logger.debug(f"🔧 Flattened audio from {original_shape} to {audio_data.shape}")
                print(f"🔧 Flattened audio from {original_shape} to {audio_data.shape}")
            
            # Resample to 16kHz if needed (critical for STT accuracy)
            TARGET_SAMPLE_RATE = 16000
            if sample_rate != TARGET_SAMPLE_RATE and audio_data.size > 0:
                try:
                    from scipy.signal import resample
                    num_samples = int(len(audio_data) * TARGET_SAMPLE_RATE / sample_rate)
                    audio_data = resample(audio_data, num_samples)
                    sample_rate = TARGET_SAMPLE_RATE
                    logger.debug(f"🔄 Resampled audio to {TARGET_SAMPLE_RATE}Hz")
                    print(f"🔄 Resampled audio to {TARGET_SAMPLE_RATE}Hz")
                    
                    # Check audio after resampling
                    audio_rms_after = (audio_data ** 2).mean() ** 0.5
                    print(f"🔍 Audio RMS after resampling: {audio_rms_after:.8f}")
                    
                except ImportError:
                    logger.warning("⚠️ scipy not available for resampling, STT accuracy may be affected")
                except Exception as e:
                    logger.warning(f"⚠️ Resampling failed: {e}")
            
            # Update duration after resampling
            duration = len(audio_data) / sample_rate
            
            # Check available STT engine methods
            available_methods = []
            if hasattr(self.stt_engine, 'transcribe_audio_async'):
                available_methods.append('transcribe_audio_async')
            if hasattr(self.stt_engine, 'process_audio_async'):
                available_methods.append('process_audio_async')
            if hasattr(self.stt_engine, 'process_audio'):
                available_methods.append('process_audio')
            if hasattr(self.stt_engine, 'transcribe'):
                available_methods.append('transcribe')
            
            logger.debug(f"🔧 Available STT methods: {available_methods}")
            logger.debug(f"🔧 STT engine type: {type(self.stt_engine).__name__}")
            print(f"🔧 Available STT methods: {available_methods}")
            print(f"🔧 STT engine type: {type(self.stt_engine).__name__}")
            
            # Use the proper async interface for FasterWhisperGPUSTT
            if hasattr(self.stt_engine, '_transcribe_audio'):
                logger.debug("🔧 Using _transcribe_audio async method")
                print("🔧 Using _transcribe_audio async method")
                try:
                    # Call _transcribe_audio directly with numpy array (like working streaming implementation)
                    print(f"🔍 [STT WORKER] About to call _transcribe_audio with raw numpy array...")
                    result = await self.stt_engine._transcribe_audio(audio_data)
                    print(f"🔍 [STT WORKER] _transcribe_audio returned: {type(result)}")
                except Exception as e:
                    print(f"❌ [STT WORKER] Exception in _transcribe_audio: {e}")
                    logger.error(f"❌ [STT WORKER] Exception in _transcribe_audio: {e}")
                    import traceback
                    print(f"❌ [STT WORKER] Traceback: {traceback.format_exc()}")
                    raise
            elif hasattr(self.stt_engine, 'transcribe_audio_async'):
                logger.debug("🔧 Using transcribe_audio_async method")
                print("🔧 Using transcribe_audio_async method")
                result = await self.stt_engine.transcribe_audio_async(audio_data, sample_rate)
            elif hasattr(self.stt_engine, 'process_audio_async'):
                logger.debug("🔧 Using process_audio_async method")
                print("🔧 Using process_audio_async method")
                audio_bytes = audio_data.tobytes()
                result = await self.stt_engine.process_audio_async(audio_bytes, sample_rate)
            elif hasattr(self.stt_engine, 'process_audio'):
                logger.debug("🔧 Using process_audio method in thread pool")
                print("🔧 Using process_audio method in thread pool")
                audio_bytes = audio_data.tobytes()
                result = await asyncio.to_thread(self.stt_engine.process_audio, audio_bytes, sample_rate)
            else:
                logger.debug("🔧 Using transcribe method in thread pool")
                print("🔧 Using transcribe method in thread pool")
                result = await asyncio.to_thread(self.stt_engine.transcribe, audio_data)
            
            print(f"🔧 STT transcription completed")
            logger.debug(f"🔧 STT result type: {type(result)}")
            if result:
                logger.debug(f"🔧 STT result text: '{getattr(result, 'text', 'NO_TEXT_ATTR')}'")
                print(f"🔧 STT result text: '{getattr(result, 'text', 'NO_TEXT_ATTR')}'")
                
            return result
            
        except Exception as e:
            logger.error(f"❌ STT transcription error: {e}")
            logger.error(f"❌ STT error type: {type(e).__name__}")
            import traceback
            logger.error(f"❌ STT traceback: {traceback.format_exc()}")
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