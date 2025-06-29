"""
TTS Streaming Worker

Threading-based worker for Text-to-Speech synthesis.
Uses async operations within thread event loop for FastRTC compatibility.
"""

import time
import asyncio
import numpy as np
from typing import Optional, AsyncGenerator, Tuple, Any

from .pipeline_workers import BasePipelineWorker
from .pipeline_manager import (
    AudioPipelineManager, LLMTokenChunk, TTSAudioChunk, GenerationStatus
)
from ..audio.engines.tts.kokoro_tts import KokoroTTSEngine
from ..audio import VoiceMapper
from ..config.language_config import KOKORO_TTS_LANG_MAP
from ..utils.logging import get_logger

logger = get_logger(__name__)


class TTSStreamingWorker(BasePipelineWorker):
    """
    Worker thread for streaming Text-to-Speech synthesis.
    
    Processes LLM token chunks and produces streaming audio output,
    integrating with existing TTS engine and interruption system.
    """
    
    def __init__(
        self,
        pipeline_manager: AudioPipelineManager,
        tts_engine: KokoroTTSEngine,
        voice_mapper: VoiceMapper,
        voice_assistant,
        processing_timeout: float = 0.1,
        chunk_size: int = 1024
    ):
        super().__init__(
            name="TTSWorker",
            pipeline_manager=pipeline_manager,
            input_queue=pipeline_manager.llm_token_queue,
            output_queue=pipeline_manager.output_queue,
            processing_timeout=processing_timeout
        )
        
        self.tts_engine = tts_engine
        self.voice_mapper = voice_mapper
        self.voice_assistant = voice_assistant
        self.chunk_size = chunk_size
        
        # TTS state management
        self.active_generations = set()
        
        # Event loop for async operations in this thread
        self.loop = None
        
        logger.info("🔊 TTSStreamingWorker initialized")
        
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
                    await self.process_item_async(item)  # TTS streams directly to output queue
                    processing_time = time.time() - start_time
                    
                    self.processed_items += 1
                    self.total_processing_time += processing_time
                        
                except Exception as e:
                    self.failed_items += 1
                    self.handle_processing_error(item, e)
                    
            except Exception as e:
                logger.error(f"Unexpected error in worker {self.name}: {e}")
                await asyncio.sleep(0.1)  # Brief pause before continuing
        
    async def process_item_async(self, llm_chunk: LLMTokenChunk) -> None:
        """
        Process LLM token chunk through TTS engine.
        
        Args:
            llm_chunk: LLM response chunk to synthesize
            
        Returns:
            TTSAudioChunk with synthesized audio, or None if still processing
        """
        generation_id = llm_chunk.generation_id
        
        # Update generation state
        state = self.pipeline_manager.get_generation_state(generation_id)
        if not state:
            logger.warning(f"No state found for generation {generation_id}")
            return None
            
        if state.tts_start_time is None:
            state.tts_start_time = time.time()
            state.status = GenerationStatus.TTS_SYNTHESIZING
            state.tts_started.set()
            
        # Skip if only partial sentence and not complete
        if not llm_chunk.is_sentence_complete:
            logger.info(f"🔊 [TTS WORKER] Skipping partial sentence for generation {generation_id}: '{llm_chunk.text}'")
            return None
            
        text_to_synthesize = llm_chunk.text.strip()
        if not text_to_synthesize:
            logger.warning(f"🔊 [TTS WORKER] Empty text for generation {generation_id}")
            return None
            
        try:
            logger.info(f"🔊 Starting TTS synthesis for generation {generation_id}: '{text_to_synthesize}'")
            
            # Reset interruption flag for new synthesis
            self.tts_engine.reset_interrupt()
            self.active_generations.add(generation_id)
            
            # Stream TTS synthesis async - this now handles multiple chunks properly
            await self._stream_tts_synthesis_async(generation_id, text_to_synthesize, state)
            
        except Exception as e:
            logger.error(f"TTS processing error for generation {generation_id}: {e}")
            state.mark_failed(f"TTS error: {e}")
            self.active_generations.discard(generation_id)
            
    async def _stream_tts_synthesis_async(self, generation_id: int, text: str, state) -> None:
        """
        Stream TTS synthesis using existing TTS engine, pushing chunks directly to output queue.
        
        Args:
            generation_id: Generation ID
            text: Text to synthesize
            state: Generation state
        """
        try:
            # Get voice and language settings
            current_language = getattr(self.voice_assistant, 'current_language', 'a')
            voices = self.voice_mapper.get_voices_for_language(current_language)
            voice_id = voices[0] if voices else None
            
            logger.info(f"🔊 [TTS WORKER] Starting synthesis for generation {generation_id}")
            logger.info(f"🔊 [TTS WORKER] Text: '{text}' (length: {len(text)})")
            logger.info(f"🔊 [TTS WORKER] Voice: '{voice_id}', Language: '{current_language}'")
            logger.info(f"🔊 [TTS WORKER] TTS Engine type: {type(self.tts_engine)}")
            
            # Stream TTS synthesis using existing engine async
            total_samples = 0
            chunk_count = 0
            total_audio_chunks = 0
            
            # Use async TTS synthesis
            if hasattr(self.tts_engine, 'stream_synthesis_async'):
                logger.info(f"🔊 [TTS WORKER] Using stream_synthesis_async")
                # Use async streaming method
                synthesis_stream = self.tts_engine.stream_synthesis_async(text, voice_id, current_language)
            elif hasattr(self.tts_engine, 'stream_synthesis'):
                logger.info(f"🔊 [TTS WORKER] Using stream_synthesis (wrapped)")
                # Wrap sync generator in async generator
                synthesis_stream = self._wrap_sync_generator(
                    self.tts_engine.stream_synthesis(text, voice_id, current_language)
                )
            else:
                logger.info(f"🔊 [TTS WORKER] Using fallback synthesis")
                # Fallback to basic synthesis
                synthesis_stream = self._fallback_synthesis_async(text, voice_id, current_language)
            
            logger.info(f"🔊 [TTS WORKER] Starting synthesis stream iteration...")
            async for sample_rate, audio_chunk in synthesis_stream:
                # Check for interruption - both generation state and TTS engine flag
                if state.interrupted.is_set() or self.tts_engine.should_interrupt:
                    logger.info(f"🛑 TTS synthesis interrupted for generation {generation_id}")
                    break
                    
                if isinstance(audio_chunk, np.ndarray) and audio_chunk.size > 0:
                    chunk_count += 1
                    total_samples += audio_chunk.size
                    
                    # Record first audio time
                    if state.first_audio_time is None:
                        state.first_audio_time = time.time()
                        
                    # Split into smaller chunks for streaming and push directly to output queue
                    for i in range(0, audio_chunk.size, self.chunk_size):
                        mini_chunk = audio_chunk[i:i+self.chunk_size]
                        if mini_chunk.size > 0:
                            tts_chunk = TTSAudioChunk(
                                generation_id=generation_id,
                                sample_rate=sample_rate,
                                audio_data=mini_chunk.astype(np.float32),
                                is_final=False  # We'll mark the last chunk as final later
                            )
                            
                            # Push chunk directly to output queue async
                            await asyncio.to_thread(self.output_queue.put, tts_chunk)
                            total_audio_chunks += 1
                            
                            logger.debug(f"🔊 Pushed audio chunk {total_audio_chunks} for generation {generation_id}: "
                                       f"{mini_chunk.size} samples")
                            
            logger.info(f"🔊 [TTS WORKER] Synthesis completed. Chunks: {chunk_count}, Samples: {total_samples}, Audio chunks: {total_audio_chunks}")
            
            # Send final chunk marker if we generated any audio
            if total_audio_chunks > 0:
                # Create a final empty chunk to signal completion
                final_chunk = TTSAudioChunk(
                    generation_id=generation_id,
                    sample_rate=sample_rate if 'sample_rate' in locals() else 24000,
                    audio_data=np.array([], dtype=np.float32),
                    is_final=True
                )
                await asyncio.to_thread(self.output_queue.put, final_chunk)
                total_audio_chunks += 1
                logger.info(f"🔊 [TTS WORKER] Sent final chunk marker")
            else:
                logger.warning(f"🔊 [TTS WORKER] No audio chunks generated for generation {generation_id}")
                
            # Mark as completed if not interrupted
            if not state.interrupted.is_set():
                state.mark_completed()
                
            self.active_generations.discard(generation_id)
            
            if total_audio_chunks > 0:
                logger.info(f"✅ TTS synthesis completed for generation {generation_id}: "
                           f"{total_audio_chunks} chunks pushed to output queue, {total_samples} samples")
            else:
                logger.warning(f"No audio generated for generation {generation_id}")
                
        except Exception as e:
            logger.error(f"Error in TTS synthesis: {e}")
            self.active_generations.discard(generation_id)
            raise
            
    async def _wrap_sync_generator(self, sync_generator) -> AsyncGenerator[Tuple[int, np.ndarray], None]:
        """
        Wrap a sync generator to make it async while preserving interruption checks.
        
        Args:
            sync_generator: Synchronous generator to wrap
            
        Yields:
            Tuples of (sample_rate, audio_chunk)
        """
        try:
            # Use asyncio.to_thread for each iteration to allow interruption checks
            iterator = iter(sync_generator)
            while True:
                try:
                    # Get next item from sync generator in thread pool
                    item = await asyncio.to_thread(next, iterator)
                    yield item
                    # Small sleep to allow other coroutines and interruption checks
                    await asyncio.sleep(0.001)
                except StopIteration:
                    # Generator exhausted
                    break
                except Exception as e:
                    logger.error(f"Error in sync generator iteration: {e}")
                    break
        except Exception as e:
            logger.error(f"Error wrapping sync generator: {e}")
            raise
                
    async def _fallback_synthesis_async(self, text: str, voice_id: str, language: str) -> AsyncGenerator[Tuple[int, np.ndarray], None]:
        """
        Fallback async synthesis method.
        
        Args:
            text: Text to synthesize
            voice_id: Voice ID to use
            language: Language code
            
        Yields:
            Tuples of (sample_rate, audio_chunk)
        """
        try:
            # Simple fallback - generate silence for testing
            sample_rate = 24000
            duration = len(text) * 0.1  # Rough estimate
            samples = int(sample_rate * duration)
            
            # Generate chunks of silence
            chunk_size = 1024
            for i in range(0, samples, chunk_size):
                chunk_samples = min(chunk_size, samples - i)
                audio_chunk = np.zeros(chunk_samples, dtype=np.float32)
                yield sample_rate, audio_chunk
                await asyncio.sleep(0.01)  # Small delay to simulate processing
                
        except Exception as e:
            logger.error(f"Error in fallback synthesis: {e}")
            raise
            
    def interrupt_generation(self, generation_id: int):
        """
        Interrupt TTS synthesis for a specific generation.
        
        Args:
            generation_id: Generation to interrupt
        """
        if generation_id in self.active_generations:
            logger.info(f"🛑 Interrupting TTS for generation {generation_id}")
            
            # Set interruption flag in TTS engine
            self.tts_engine.interrupt()
            
            # Update generation state
            state = self.pipeline_manager.get_generation_state(generation_id)
            if state:
                state.mark_interrupted()
                
            self.active_generations.discard(generation_id)
            
    def interrupt_all_active(self):
        """Interrupt all active TTS syntheses."""
        if self.active_generations:
            logger.info(f"🛑 Interrupting all active TTS syntheses: {self.active_generations}")
            
            # Set global interruption flag
            self.tts_engine.interrupt()
            
            # Mark all active generations as interrupted
            for generation_id in list(self.active_generations):
                state = self.pipeline_manager.get_generation_state(generation_id)
                if state:
                    state.mark_interrupted()
                    
            self.active_generations.clear()
            
    def handle_processing_error(self, item, error):
        """Handle TTS processing errors."""
        super().handle_processing_error(item, error)
        
        # Clean up active generation tracking
        if hasattr(item, 'generation_id'):
            self.active_generations.discard(item.generation_id)
            
    def process_item(self, llm_chunk) -> Optional[Any]:
        """
        Synchronous wrapper for process_item_async.
        Required by BasePipelineWorker abstract method.
        
        Args:
            llm_chunk: LLM chunk data to process
            
        Returns:
            None - this worker uses async processing
        """
        # This method should not be called directly since TTSStreamingWorker
        # overrides the worker loop to use async processing
        logger.warning("process_item called on TTSStreamingWorker - this should use async processing")
        return None
        
    def get_worker_stats(self) -> dict:
        """Get TTS worker statistics."""
        stats = super().get_worker_stats()
        stats.update({
            "chunk_size": self.chunk_size,
            "active_generations": len(self.active_generations),
            "tts_engine_available": self.tts_engine.is_available(),
        })
        return stats


class TTSOutputWorker(BasePipelineWorker):
    """
    Worker thread for managing TTS audio output to FastRTC.
    
    Handles the final stage of audio delivery back to the FastRTC callback.
    """
    
    def __init__(
        self,
        pipeline_manager: AudioPipelineManager,
        output_callback=None,
        processing_timeout: float = 0.05  # Faster for audio output
    ):
        super().__init__(
            name="TTSOutputWorker",
            pipeline_manager=pipeline_manager,
            input_queue=pipeline_manager.output_queue,
            output_queue=None,  # No output queue, we're the final stage
            processing_timeout=processing_timeout
        )
        
        self.output_callback = output_callback
        self.delivered_chunks = 0
        
        logger.info("📢 TTSOutputWorker initialized")
        
    def process_item(self, tts_chunk: TTSAudioChunk) -> None:
        """
        Process TTS audio chunk for output delivery.
        
        Args:
            tts_chunk: Audio chunk to deliver
        """
        try:
            # Check if generation is still active
            state = self.pipeline_manager.get_generation_state(tts_chunk.generation_id)
            if not state or state.interrupted.is_set():
                logger.debug(f"Skipping output for interrupted generation {tts_chunk.generation_id}")
                return
                
            # Deliver audio chunk via callback
            if self.output_callback:
                self.output_callback(tts_chunk)
                self.delivered_chunks += 1
                
                logger.debug(f"📢 Delivered audio chunk for generation {tts_chunk.generation_id}: "
                           f"{tts_chunk.audio_data.size} samples")
            else:
                logger.warning("No output callback configured")
                
        except Exception as e:
            logger.error(f"Error delivering audio output: {e}")
            
    def set_output_callback(self, callback):
        """Set the output callback function."""
        self.output_callback = callback
        logger.info("📢 Output callback configured")
        
    def get_worker_stats(self) -> dict:
        """Get output worker statistics."""
        stats = super().get_worker_stats()
        stats.update({
            "delivered_chunks": self.delivered_chunks,
            "has_callback": self.output_callback is not None,
        })
        return stats