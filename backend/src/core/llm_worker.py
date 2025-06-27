"""
LLM Streaming Worker

Threading-based worker for streaming LLM response generation.
Uses async operations within thread event loop for FastRTC compatibility.
"""

import time
import asyncio
import re
from typing import Optional, AsyncGenerator

from .pipeline_workers import BasePipelineWorker
from .pipeline_manager import (
    AudioPipelineManager, TranscriptionChunk, LLMTokenChunk, GenerationStatus
)
from ..utils.logging import get_logger

logger = get_logger(__name__)


class LLMStreamingWorker(BasePipelineWorker):
    """
    Worker thread for streaming LLM response generation.
    
    Processes transcription chunks and streams LLM tokens/sentences
    for real-time response generation.
    """
    
    def __init__(
        self,
        pipeline_manager: AudioPipelineManager,
        voice_assistant,
        min_sentence_length: int = 10,  # minimum chars for sentence
        sentence_endings: tuple = ('.', '!', '?', '...'),
        processing_timeout: float = 0.1
    ):
        super().__init__(
            name="LLMWorker", 
            pipeline_manager=pipeline_manager,
            input_queue=pipeline_manager.transcription_queue,
            output_queue=pipeline_manager.llm_token_queue,
            processing_timeout=processing_timeout
        )
        
        self.voice_assistant = voice_assistant
        self.min_sentence_length = min_sentence_length
        self.sentence_endings = sentence_endings
        
        # Token buffering for sentence completion
        self.token_buffers = {}  # generation_id -> accumulated tokens
        
        # Event loop for async operations in this thread
        self.loop = None
        
        logger.info("🧠 LLMStreamingWorker initialized")
    
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
        
    async def process_item_async(self, transcription: TranscriptionChunk) -> Optional[LLMTokenChunk]:
        """
        Process transcription through LLM and stream response tokens.
        
        Args:
            transcription: Transcription result to process
            
        Returns:
            LLMTokenChunk with response tokens, or None if still processing
        """
        generation_id = transcription.generation_id
        
        # Skip partial transcriptions
        if transcription.is_partial:
            logger.debug(f"Skipping partial transcription for generation {generation_id}")
            return None
            
        # Update generation state
        state = self.pipeline_manager.get_generation_state(generation_id)
        if not state:
            logger.warning(f"No state found for generation {generation_id}")
            return None
            
        if state.llm_start_time is None:
            state.llm_start_time = time.time()
            state.status = GenerationStatus.LLM_STREAMING
            state.llm_started.set()
            
        user_text = transcription.text.strip()
        if not user_text:
            logger.debug(f"Empty transcription for generation {generation_id}")
            return None
            
        try:
            logger.info(f"🧠 Starting LLM processing for generation {generation_id}: '{user_text}'")
            
            # Stream LLM response tokens async
            return await self._stream_llm_response_async(generation_id, user_text, state)
            
        except Exception as e:
            logger.error(f"LLM processing error for generation {generation_id}: {e}")
            state.mark_failed(f"LLM error: {e}")
            return None
            
    async def _stream_llm_response_async(self, generation_id: int, user_text: str, state) -> Optional[LLMTokenChunk]:
        """
        Stream LLM response and yield sentence chunks.
        
        Args:
            generation_id: Generation ID
            user_text: User input text
            state: Generation state
            
        Returns:
            LLMTokenChunk with sentence, or None if still accumulating
        """
        try:
            # Get LLM response stream async
            response_stream = self._get_llm_stream_async(user_text)
            
            # Initialize token buffer for this generation
            if generation_id not in self.token_buffers:
                self.token_buffers[generation_id] = ""
                
            # Process tokens from stream
            sentence_chunk = None
            async for token in response_stream:
                # Check for interruption
                if state.interrupted.is_set():
                    logger.info(f"LLM streaming interrupted for generation {generation_id}")
                    break
                    
                # Add token to buffer
                self.token_buffers[generation_id] += token
                
                # Check if we have a complete sentence
                sentence = self._extract_complete_sentence(generation_id)
                if sentence:
                    logger.debug(f"🧠 Complete sentence for generation {generation_id}: '{sentence}'")
                    sentence_chunk = LLMTokenChunk(
                        generation_id=generation_id,
                        text=sentence,
                        is_sentence_complete=True
                    )
                    break
                    
            # If no complete sentence but stream ended, send remaining buffer
            if not sentence_chunk and generation_id in self.token_buffers:
                remaining_text = self.token_buffers[generation_id].strip()
                if remaining_text:
                    logger.debug(f"🧠 Final chunk for generation {generation_id}: '{remaining_text}'")
                    sentence_chunk = LLMTokenChunk(
                        generation_id=generation_id,
                        text=remaining_text,
                        is_sentence_complete=True
                    )
                    
            # Clean up buffer
            if generation_id in self.token_buffers:
                del self.token_buffers[generation_id]
                
            # Update conversation state
            if sentence_chunk:
                state.response_text = sentence_chunk.text
                self._update_conversation(user_text, sentence_chunk.text)
                
            return sentence_chunk
            
        except Exception as e:
            logger.error(f"Error streaming LLM response: {e}")
            raise
            
    async def _get_llm_stream_async(self, user_text: str) -> AsyncGenerator[str, None]:
        """
        Get streaming LLM response using async voice assistant.
        
        Args:
            user_text: User input text
            
        Yields:
            Token strings from LLM response
        """
        try:
            # Use voice assistant async methods for full functionality
            if hasattr(self.voice_assistant, 'stream_llm_response_async'):
                # Use the voice assistant's async streaming method
                async for token in self.voice_assistant.stream_llm_response_async(user_text):
                    yield token
            elif hasattr(self.voice_assistant, 'get_llm_response_async'):
                # Fallback to non-streaming async method
                response = await self.voice_assistant.get_llm_response_async(user_text)
                words = response.split()
                for i, word in enumerate(words):
                    if i == 0:
                        yield word
                    else:
                        yield " " + word
            elif hasattr(self.voice_assistant, 'llm_service'):
                # Use the voice assistant's LLM service directly
                llm_service = self.voice_assistant.llm_service
                if hasattr(llm_service, 'stream_response'):
                    async for token in llm_service.stream_response(user_text, context=""):
                        yield token
                else:
                    # Fallback to non-streaming
                    response = await llm_service.get_response(user_text, context="")
                    words = response.split()
                    for i, word in enumerate(words):
                        if i == 0:
                            yield word
                        else:
                            yield " " + word
            else:
                # Final fallback for testing
                response = "I understand your request and will help you with that."
                words = response.split()
                for i, word in enumerate(words):
                    if i == 0:
                        yield word
                    else:
                        yield " " + word
                        await asyncio.sleep(0.01)  # Small delay to simulate streaming
                    
        except Exception as e:
            logger.error(f"Error getting LLM stream: {e}")
            raise
            
    def _extract_complete_sentence(self, generation_id: int) -> Optional[str]:
        """
        Extract complete sentence from token buffer.
        
        Args:
            generation_id: Generation ID
            
        Returns:
            Complete sentence if found, None otherwise
        """
        if generation_id not in self.token_buffers:
            return None
            
        buffer = self.token_buffers[generation_id]
        
        # Look for sentence endings
        for ending in self.sentence_endings:
            if ending in buffer:
                # Find the last occurrence of this ending
                end_pos = buffer.rfind(ending)
                if end_pos != -1:
                    # Extract sentence including the ending
                    sentence = buffer[:end_pos + len(ending)].strip()
                    
                    # Check minimum length
                    if len(sentence) >= self.min_sentence_length:
                        # Update buffer to remove extracted sentence
                        self.token_buffers[generation_id] = buffer[end_pos + len(ending):].strip()
                        return sentence
                        
        # Check for other natural break points if buffer is getting long
        if len(buffer) > 100:  # Arbitrary threshold
            # Look for commas, conjunctions, etc.
            break_patterns = [', ', ' and ', ' but ', ' so ', ' then ']
            for pattern in break_patterns:
                if pattern in buffer:
                    break_pos = buffer.rfind(pattern)
                    if break_pos > self.min_sentence_length:
                        sentence = buffer[:break_pos].strip()
                        self.token_buffers[generation_id] = buffer[break_pos:].strip()
                        return sentence
                        
        return None
        
    def _update_conversation(self, user_text: str, assistant_text: str):
        """
        Update conversation buffer with new exchange.
        
        Args:
            user_text: User input
            assistant_text: Assistant response
        """
        try:
            if hasattr(self.voice_assistant, 'conversation_buffer'):
                self.voice_assistant.conversation_buffer.add_turn(
                    user_text=user_text,
                    assistant_text=assistant_text,
                    language=getattr(self.voice_assistant, 'current_language', 'en')
                )
                
            # Update turn count
            if hasattr(self.voice_assistant, 'turn_count'):
                self.voice_assistant.turn_count += 1
                
        except Exception as e:
            logger.warning(f"Error updating conversation: {e}")
            
    def cleanup_buffers(self, generation_id: int):
        """Clean up token buffers for a specific generation."""
        if generation_id in self.token_buffers:
            del self.token_buffers[generation_id]
            logger.debug(f"Cleaned up token buffer for generation {generation_id}")
            
    def handle_processing_error(self, item, error):
        """Handle LLM processing errors."""
        super().handle_processing_error(item, error)
        
        # Clean up buffers for failed generation
        if hasattr(item, 'generation_id'):
            self.cleanup_buffers(item.generation_id)
            
    def get_worker_stats(self) -> dict:
        """Get LLM worker statistics."""
        stats = super().get_worker_stats()
        stats.update({
            "min_sentence_length": self.min_sentence_length,
            "sentence_endings": self.sentence_endings,
            "buffered_generations": len(self.token_buffers),
        })
        return stats