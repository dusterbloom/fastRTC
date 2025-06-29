"""
LLM Streaming Worker

Threading-based worker for streaming LLM response generation.
Uses async operations within thread event loop for FastRTC compatibility.
"""

import time
import asyncio
import re
from typing import Optional, AsyncGenerator, Any

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
        
        # Log that LLM worker is being used
        logger.info("🧠 [LLM_WORKER] LLM Worker initialized - authentication enabled")
        
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
        logger.info(f"🧠 [LLM_WORKER] process_item_async called with transcription: {transcription.text[:50]}...")
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
        
        # Check for authentication/user identification
        logger.info(f"🔍 [LLM_WORKER] Starting authentication check for text: '{user_text[:100]}...'")
        logger.info(f"🔍 [LLM_WORKER] Voice assistant type: {type(self.voice_assistant)}")
        logger.info(f"🔍 [LLM_WORKER] Has voice_print_manager: {hasattr(self.voice_assistant, 'voice_print_manager')}")
        
        auth_result = None
        try:
            if hasattr(self.voice_assistant, 'voice_print_manager') and self.voice_assistant.voice_print_manager:
                voice_manager = self.voice_assistant.voice_print_manager
                logger.info(f"🔍 [LLM_WORKER] Processing text for voice auth: '{user_text[:50]}...'")
                logger.info(f"🔍 [LLM_WORKER] Audio buffer available: {voice_manager.audio_buffer is not None}")
                logger.info(f"🔍 [LLM_WORKER] Enrollment mode: {voice_manager.enrollment_mode}")
                
                # If we're in enrollment mode and have audio, process the voice sample
                if voice_manager.enrollment_mode and hasattr(voice_manager, 'audio_buffer') and voice_manager.audio_buffer is not None:
                    logger.info(f"🎤 [LLM_WORKER] Processing voice sample for enrollment (size: {len(voice_manager.audio_buffer)})")
                    # Process the voice sample directly
                    auth_result = voice_manager.add_voice_sample(voice_manager.audio_buffer)
                    logger.info(f"🔍 [LLM_WORKER] Voice sample result: {auth_result}")
                else:
                    # Regular text processing for authentication/enrollment initiation
                    auth_result = voice_manager.process_text(user_text)
                    logger.info(f"🔍 [LLM_WORKER] Authentication result: {auth_result}")
            else:
                logger.info(f"🔍 [LLM_WORKER] voice_print_manager not available - proceeding with normal LLM processing")
                auth_result = None
        except Exception as auth_error:
            logger.error(f"❌ [LLM_WORKER] Authentication error: {auth_error}")
            import traceback
            logger.error(f"❌ [LLM_WORKER] Auth traceback: {traceback.format_exc()}")
            auth_result = None
        
        if auth_result:
            action = auth_result.get('action')
            confirmation_message = None
            
            if action == 'user_identified':
                # Successful login
                identified_user_id = auth_result['user_id']
                if identified_user_id != self.voice_assistant.user_id:
                    logger.info(f"🔄 [LLM_WORKER] User authenticated: switching from '{self.voice_assistant.user_id}' to '{identified_user_id}'")
                    
                    # Extract username for display
                    username = identified_user_id.replace("user_", "")
                    
                    # Switch memory manager to new user
                    if self.voice_assistant.memory_manager.switch_user(identified_user_id):
                        logger.info(f"✅ [LLM_WORKER] Memory manager switched to user: {identified_user_id}")
                        
                        # Refresh session with new authenticated user
                        self.voice_assistant.refresh_session_after_auth(identified_user_id, username)
                        
                        confirmation_message = f"Welcome back, {username}! I've loaded your personal memory profile and started a fresh session."
                    else:
                        logger.error(f"❌ [LLM_WORKER] Failed to switch memory manager to user: {identified_user_id}")
                        confirmation_message = "I recognized you, but there was an issue accessing your personal profile."
            
            elif action == 'pin_request':
                # User exists, need PIN
                username = auth_result['username']
                confirmation_message = f"Hello {username}! Please provide your 4-digit PIN to access your profile."
                
            elif action == 'username_request':
                # Echo register initiated, asking for username
                confirmation_message = auth_result.get('message', 'What would you like your username to be?')
                
            elif action == 'pin_request_registration':
                # Username provided, asking for PIN during registration
                username = auth_result.get('username', 'user')
                confirmation_message = auth_result.get('message', f"Great! Now please provide a 4-digit PIN for {username}.")
                
            elif action == 'registration_success':
                # New user registered and logged in
                identified_user_id = auth_result['user_id']
                username = identified_user_id.replace("user_", "")
                
                if self.voice_assistant.memory_manager.switch_user(identified_user_id):
                    logger.info(f"✅ [LLM_WORKER] New user registered: {identified_user_id}")
                    
                    # Refresh session with new registered user
                    self.voice_assistant.refresh_session_after_auth(identified_user_id, username)
                    
                    confirmation_message = f"Welcome {username}! Your account has been created and I'll remember our conversations in your new session."
                else:
                    logger.error(f"❌ [LLM_WORKER] Failed to switch memory manager for new user: {identified_user_id}")
                    confirmation_message = "Your account was created, but there was an issue setting up your memory profile."
            
            elif action == 'suggest_registration':
                # User identified but not registered
                username = auth_result.get('username', 'unknown')
                confirmation_message = auth_result.get('message', f"I don't know you yet, {username}. Would you like to register? Say 'register as {username} PIN 1234' with your chosen 4-digit PIN.")
            
            elif action == 'login_cancelled':
                # User cancelled login
                username = auth_result.get('username', 'someone')
                confirmation_message = f"No problem, {username}. I'll continue with the temporary session."
            
            elif action == 'registration_cancelled':
                # User cancelled registration
                username = auth_result.get('username', 'someone')
                confirmation_message = f"No problem, {username}. Registration cancelled. I'll continue with the temporary session."
            
            elif action == 'voice_enrollment_started':
                # Voice enrollment initiated
                user_id = auth_result.get('user_id', 'unknown')
                phrase = auth_result.get('phrase', 'The quick brown fox jumps over the lazy dog')
                sample_num = auth_result.get('sample', 1)
                total_samples = auth_result.get('total_samples', 2)
                confirmation_message = f"Great! I'll register your voice. Please say this phrase clearly: '{phrase}'. This is sample {sample_num} of {total_samples}."
            
            elif action == 'voice_sample_received':
                # Voice sample received, need more
                sample_num = auth_result.get('sample', 2)
                total_samples = auth_result.get('total_samples', 2)
                confirmation_message = f"Perfect! Now say the same phrase one more time. This is sample {sample_num} of {total_samples}."
            
            elif action == 'voice_enrollment_complete':
                # Voice enrollment completed successfully
                user_id = auth_result.get('user_id', 'unknown')  # Should be 'user_1234567890'
                username = user_id.replace('user_', '')  # Extract just the timestamp
                
                # Switch to the new voice user (user_id is already in correct format)
                if self.voice_assistant.memory_manager.switch_user(user_id):
                    logger.info(f"✅ [LLM_WORKER] Voice user registered and switched: {user_id}")
                    self.voice_assistant.refresh_session_after_auth(user_id, f"Voice User {username}")
                    confirmation_message = f"Excellent! Your voice is now registered. Just say 'it's me' anytime to authenticate instantly. Welcome to your personalized session!"
                else:
                    logger.error(f"❌ [LLM_WORKER] Failed to switch to voice user: {user_id}")
                    confirmation_message = "Your voice was registered, but there was an issue setting up your session."
            
            elif action == 'auth_failed':
                # Authentication failed
                reason = auth_result.get('reason', 'unknown')
                if reason == 'invalid_pin':
                    confirmation_message = "Sorry, that PIN is incorrect. Please try again or say 'cancel' to stop."
                elif reason == 'user_not_found':
                    username = auth_result.get('username', 'unknown')
                    confirmation_message = f"I don't have a user named '{username}'. Would you like to register? Say 'register as {username} PIN 1234' with your chosen 4-digit PIN."
                elif reason == 'invalid_username':
                    confirmation_message = "That username isn't valid. Please choose a different name."
                elif reason == 'voice_not_recognized':
                    confirmation_message = auth_result.get('message', "I don't recognize your voice. Say 'register my voice' to enroll.")
                elif reason == 'no_audio':
                    confirmation_message = auth_result.get('message', "Please try speaking again.")
                else:
                    confirmation_message = "Authentication failed. Please try again."
            
            # If we have a confirmation message, return it directly instead of processing through LLM
            if confirmation_message:
                logger.info(f"🔐 [LLM_WORKER] Returning authentication message: {confirmation_message}")
                # Create a simple token chunk with the confirmation message
                return LLMTokenChunk(
                    generation_id=generation_id,
                    tokens=[confirmation_message],
                    is_sentence_complete=True,
                    sentence_text=confirmation_message,
                    is_final=True
                )
            
        # Proceed with normal LLM processing (no auth required)
        logger.info(f"🧠 [LLM_WORKER] No authentication action required - proceeding with normal LLM processing")
        
        try:
            logger.info(f"🧠 [LLM_WORKER] Starting LLM processing for generation {generation_id}: '{user_text}'")
            
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
            
    def process_item(self, transcription_chunk) -> Optional[Any]:
        """
        Synchronous wrapper for process_item_async.
        Required by BasePipelineWorker abstract method.
        
        Args:
            transcription_chunk: Transcription data to process
            
        Returns:
            None - this worker uses async processing
        """
        # This method should not be called directly since LLMStreamingWorker
        # overrides the worker loop to use async processing
        logger.warning("process_item called on LLMStreamingWorker - this should use async processing")
        return None
        
    def get_worker_stats(self) -> dict:
        """Get LLM worker statistics."""
        stats = super().get_worker_stats()
        stats.update({
            "min_sentence_length": self.min_sentence_length,
            "sentence_endings": self.sentence_endings,
            "buffered_generations": len(self.token_buffers),
        })
        return stats