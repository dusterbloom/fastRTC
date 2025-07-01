"""
STT Streaming Worker

Threading-based worker for Speech-to-Text processing.
Uses async operations within thread event loop for FastRTC compatibility.
"""

import time
import asyncio
import multiprocessing
import numpy as np
from typing import Optional, Any

from .pipeline_workers import BasePipelineWorker
from .pipeline_manager import (
    AudioPipelineManager, AudioChunk, TranscriptionChunk, GenerationStatus
)
from .transcription_worker import start_transcription_worker
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
        min_audio_length: float = 2.0,  # minimum seconds of audio (matches async handler)
        processing_timeout: float = 0.1,
        event_loop=None,
        model_path: str = "Systran/faster-whisper-large-v3",
        device: str = "cuda",
        compute_type: str = "int8_float16"
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
        self.event_loop = event_loop
        
        # Process-based transcription setup (RealtimeSTT pattern)
        self.model_path = model_path
        self.device = device
        self.compute_type = compute_type
        
        # Thread communication
        self.transcription_thread = None
        self.transcription_queue = None
        self.transcription_result_queue = None
        self.transcription_ready = False
        
        # Audio buffering for minimum length requirements
        self.audio_buffers = {}  # generation_id -> list of audio chunks
        
        # Sentence buffering (like async handler)
        self.sentence_buffers = {}  # generation_id -> string buffer
        self.buffer_timestamps = {}  # generation_id -> timestamp
        self.buffer_timeout = 3.0  # seconds (same as async handler)
        self.max_buffer_length = 500  # characters (same as async handler)
        
        logger.info(f"🎤 STTStreamingWorker initialized with confidence threshold {confidence_threshold}")
        
        # Event loop for async operations in this thread
        self.loop = None
        
        # Start transcription thread
        self._start_transcription_process()
        
    def _start_transcription_process(self):
        """Initialize model in main thread, then start worker thread."""
        try:
            logger.info("🎤 Initializing faster-whisper model in main thread...")
            
            # Import and initialize faster-whisper in MAIN thread
            import faster_whisper
            import threading
            import queue
            
            # Keep CUDA settings
            device = self.device
            compute_type = self.compute_type
            
            if device == "cuda":
                try:
                    import torch
                    logger.info(f"🎤 DEBUG: torch imported successfully, version: {torch.__version__}")
                    logger.info(f"🎤 DEBUG: torch.cuda.is_available() = {torch.cuda.is_available()}")
                    if torch.cuda.is_available():
                        logger.info(f"🎤 DEBUG: CUDA device count: {torch.cuda.device_count()}")
                        logger.info(f"🎤 DEBUG: Current device: {torch.cuda.current_device()}")
                        logger.info("🎤 CUDA available, using GPU acceleration")
                    else:
                        logger.error("🎤 DEBUG: CUDA reported as not available by torch.cuda.is_available()")
                        raise RuntimeError("CUDA not available")
                except Exception as cuda_error:
                    logger.error(f"🎤 DEBUG: Exception during CUDA check: {cuda_error}")
                    raise RuntimeError(f"CUDA initialization failed: {cuda_error}")
            
            # Initialize model in MAIN thread with CUDA context
            logger.info(f"🎤 Loading model with device: {device}, compute_type: {compute_type}")
            self.transcription_model = faster_whisper.WhisperModel(
                model_size_or_path=self.model_path,
                device=device,
                compute_type=compute_type
            )
            
            # Add model lock for thread safety
            self.model_lock = threading.Lock()
            
            logger.info(f"✅ Model initialized with {device} in main thread")
            
            # Create communication queues
            self.transcription_queue = queue.Queue()
            self.transcription_result_queue = queue.Queue()
            self.transcription_shutdown = threading.Event()
            
            # Start transcription thread (model already initialized)
            self.transcription_thread = threading.Thread(
                target=self._transcription_worker_thread,
                daemon=True
            )
            self.transcription_thread.start()
            
            self.transcription_ready = True
            logger.info("✅ Transcription worker ready with CUDA acceleration")
                
        except Exception as e:
            logger.error(f"❌ Failed to start transcription worker: {e}")
            self.transcription_ready = False
            raise
            
    def _transcription_worker_thread(self):
        """Worker thread for transcription processing."""
        logger.info("🎤 Transcription worker thread started")
        
        while not self.transcription_shutdown.is_set():
            try:
                # Get transcription request from queue
                try:
                    audio_data, language, use_prompt = self.transcription_queue.get(timeout=0.1)
                except:
                    continue
                
                # Perform transcription
                try:
                    logger.debug(f"🎤 THREAD: Transcribing audio with language {language}")
                    
                    # Validate audio data
                    if audio_data is None or audio_data.size == 0:
                        logger.error("🎤 THREAD: Received empty audio data")
                        self.transcription_result_queue.put(('error', "Empty audio data"))
                        continue
                    
                    print(f"🎤 THREAD: Input audio shape: {audio_data.shape}, dtype: {audio_data.dtype}")
                    
                    # Convert to 1D mono audio if needed
                    if audio_data.ndim > 1:
                        if audio_data.shape[0] == 1:
                            # Single channel, squeeze to 1D
                            audio_data = audio_data.squeeze(0)
                        else:
                            # Multiple channels, take first channel
                            audio_data = audio_data[0]
                        print(f"🎤 THREAD: Converted to 1D: {audio_data.shape}")
                    
                    # Convert int16 to float32 if needed (do this BEFORE normalization)
                    if audio_data.dtype == np.int16:
                        print(f"🎤 THREAD: Converting int16 to float32...")
                        audio_data = audio_data.astype(np.float32) / 32768.0
                        print(f"🎤 THREAD: Converted to float32, range: [{np.min(audio_data):.3f}, {np.max(audio_data):.3f}]")
                    elif audio_data.dtype != np.float32:
                        audio_data = audio_data.astype(np.float32)
                        print(f"🎤 THREAD: Converted to float32: {audio_data.dtype}")
                    
                    # Check audio content
                    audio_energy = np.mean(np.abs(audio_data))
                    print(f"🎤 THREAD: Audio energy: {audio_energy:.6f}")
                    
                    # Apply light normalization only if needed (don't over-normalize)
                    peak = np.max(np.abs(audio_data))
                    if peak > 0.8:  # Only normalize if audio is too loud
                        audio_data = (audio_data / peak) * 0.8
                        print(f"🎤 THREAD: Normalized loud audio, original peak: {peak:.3f}")
                    elif peak < 0.01:  # Audio might be too quiet
                        print(f"🎤 THREAD: Warning: Audio might be too quiet, peak: {peak:.6f}")
                    
                    # Synchronous transcription (simple approach)
                    print(f"🎤 THREAD: Starting transcription of {audio_data.shape} audio...")
                    import time
                    start_time = time.time()
                    
                    print(f"🎤 THREAD: Calling transcribe method with lock...")
                    with self.model_lock:
                        segments, info = self.transcription_model.transcribe(
                            audio_data,
                            language="en",  # Force English to prevent language confusion
                            beam_size=1,    # Reduced for speed and thread safety
                            initial_prompt=None,
                            vad_filter=False,  # Disabled since FastRTC has SileroVAD
                            temperature=0.0,  # Deterministic
                            condition_on_previous_text=False,
                            no_speech_threshold=0.6,  # Default threshold to reject noise
                            logprob_threshold=-1.0,   # Default threshold
                            compression_ratio_threshold=2.4,  # Reject repetitive text
                            without_timestamps=True  # Faster processing
                        )
                    
                    print(f"🎤 THREAD: Transcribe method returned, processing segments...")
                    
                    # Extract text
                    segments_list = list(segments)
                    transcription = " ".join(segment.text for segment in segments_list).strip()
                    
                    # Validate transcription quality
                    if self._is_valid_transcription(transcription):
                        elapsed = time.time() - start_time
                        print(f"🎤 THREAD: Transcription completed in {elapsed:.2f}s: '{transcription}' ({len(segments_list)} segments)")
                        logger.debug(f"🎤 THREAD: Transcription completed: '{transcription}'")
                        
                        # Send result
                        self.transcription_result_queue.put(('success', (transcription, info)))
                    else:
                        elapsed = time.time() - start_time
                        print(f"🎤 THREAD: Rejected invalid transcription in {elapsed:.2f}s: '{transcription}'")
                        # Send empty result for invalid transcription
                        self.transcription_result_queue.put(('success', ("", info)))
                    
                except Exception as e:
                    logger.error(f"🎤 THREAD: Transcription error: {e}")
                    self.transcription_result_queue.put(('error', str(e)))
                    
            except Exception as e:
                logger.error(f"🎤 THREAD: General error: {e}")
        
        logger.info("🎤 Transcription worker thread stopped")
    
    def _is_valid_transcription(self, text: str) -> bool:
        """Validate transcription quality to reject hallucinations."""
        if not text or len(text.strip()) == 0:
            return False
        
        # Check for excessive repetition (like "Amma Amma Amma...")
        words = text.split()
        if len(words) > 3:
            # Count most frequent word
            word_counts = {}
            for word in words:
                word_counts[word] = word_counts.get(word, 0) + 1
            
            max_count = max(word_counts.values())
            repetition_ratio = max_count / len(words)
            
            # Reject if more than 80% repetition
            if repetition_ratio > 0.8:
                print(f"🎤 VALIDATION: Rejected repetitive text (ratio: {repetition_ratio:.2f})")
                return False
        
        # Check for common hallucination patterns
        hallucination_patterns = [
            "thank you", "thanks for watching", "subscribe", "like and subscribe",
            "amma", "mama", "papa", "uh", "um", "ah", "eh"
        ]
        
        text_lower = text.lower()
        for pattern in hallucination_patterns:
            if text_lower.count(pattern) > 2:
                print(f"🎤 VALIDATION: Rejected hallucination pattern: '{pattern}'")
                return False
        
        # Must have minimum meaningful length
        if len(text.strip()) < 3:
            print(f"🎤 VALIDATION: Rejected too short text: '{text}'")
            return False
        
        print(f"🎤 VALIDATION: Accepted valid transcription: '{text[:50]}...'")
        return True
    
    def _stop_transcription_process(self):
        """Stop the transcription worker thread."""
        if hasattr(self, 'transcription_shutdown'):
            logger.info("🛑 Stopping transcription worker thread...")
            self.transcription_shutdown.set()
            
            if hasattr(self, 'transcription_thread') and self.transcription_thread.is_alive():
                self.transcription_thread.join(timeout=5)
                
        self.transcription_ready = False
        logger.info("🛑 Transcription worker stopped")
        
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
            # Stop transcription thread
            self._stop_transcription_process()
            if self.loop and not self.loop.is_closed():
                self.loop.close()
            logger.info(f"🏁 Worker {self.name} stopped")
            
    async def _async_worker_loop(self):
        """Async worker loop that processes items from queue."""
        while not self.stop_requested and not self.pipeline_manager.stop_event.is_set():
            try:
                # Check for sentence buffer timeouts
                await self._check_all_buffer_timeouts()
                
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
            print(f"🎤 STT BUFFERING: Generation {generation_id} - {total_duration:.2f}s/{self.min_audio_length}s (need more audio)")
            logger.debug(f"Buffering audio for generation {generation_id}: {total_duration:.2f}s")
            return None
            
        print(f"🎤 STT PROCESSING: Generation {generation_id} - {total_duration:.2f}s audio ready for transcription")
            
        # Combine buffered audio
        combined_audio = self._combine_audio_chunks(self.audio_buffers[generation_id])
        
        # Clean up buffer
        del self.audio_buffers[generation_id]
        
        try:
            # Run STT processing async
            logger.debug(f"🎤 Processing {total_duration:.2f}s of audio for generation {generation_id}")
            
            start_time = time.time()
            
            # Check if we have a dedicated STT engine (like WhisperLive)
            if hasattr(self, 'stt_engine') and self.stt_engine is not None:
                print(f"🎤 STT WORKER: Using STT engine: {type(self.stt_engine).__name__}")
                transcription_result = await self._transcribe_with_engine(combined_audio, None)
            else:
                # Use process-based transcription (RealtimeSTT pattern)
                print(f"🎤 STT WORKER: Using process-based transcription like RealtimeSTT...")
                transcription_result = await asyncio.to_thread(
                    self._transcribe_with_process, combined_audio, None
                )
            processing_time = time.time() - start_time
            
            if not transcription_result or not transcription_result.text.strip():
                logger.debug(f"No transcription result for generation {generation_id}")
                # Check for buffered sentence timeout
                buffered_text = self._check_sentence_buffer_timeout(generation_id)
                if buffered_text:
                    return self._create_transcription_chunk(generation_id, buffered_text, 1.0, state)
                return None
            
            current_text = transcription_result.text.strip()
            confidence = getattr(transcription_result, 'confidence', 1.0)
            
            print(f"🎤 STT WORKER: Raw transcription: '{current_text}' (confidence: {confidence:.2f})")
            
            # Add to sentence buffer and check for complete sentences
            complete_sentences = self._add_to_sentence_buffer(generation_id, current_text)
            
            if complete_sentences:
                # Found complete sentences, return them
                print(f"🎤 STT WORKER: Complete sentences found: '{complete_sentences}'")
                return self._create_transcription_chunk(generation_id, complete_sentences, confidence, state)
            elif audio_chunk.is_final:
                # This is the final chunk, flush any remaining buffer
                buffered_text = self._flush_sentence_buffer(generation_id)
                if buffered_text:
                    print(f"🎤 STT WORKER: Final chunk - flushing buffer: '{buffered_text}'")
                    return self._create_transcription_chunk(generation_id, buffered_text, confidence, state)
            
            # No complete sentences yet, continue buffering
            print(f"🎤 STT WORKER: Buffering text chunk: '{current_text}'")
            return None
            
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
        
    def _transcribe_with_process(self, audio_data: np.ndarray, language: Optional[str] = None):
        """
        Transcribe audio using the separate transcription thread.
        
        Args:
            audio_data: Audio samples as numpy array
            language: Target language (None for auto-detect)
            
        Returns:
            Transcription result object
        """
        if not self.transcription_ready:
            logger.error("🎤 Transcription worker not ready")
            return self._create_empty_transcription_result()
            
        try:
            # Ensure audio is in proper format before sending to thread
            print(f"🎤 THREAD: Input audio for transcription: {audio_data.shape}, dtype: {audio_data.dtype}")
            
            # Send transcription request to thread (keep original format, let thread handle conversion)
            self.transcription_queue.put((audio_data, language, True))  # use_prompt=True
            
            # Wait for result with timeout
            try:
                status, result = self.transcription_result_queue.get(timeout=10.0)  # 10 second timeout
                
                if status == 'success':
                    transcription_text, info = result
                    print(f"🎤 THREAD: Received transcription: '{transcription_text}'")
                    
                    # Create result object compatible with existing code
                    return self._create_transcription_result(
                        text=transcription_text,
                        language=getattr(info, 'language', 'en'),
                        confidence=getattr(info, 'language_probability', 1.0)
                    )
                elif status == 'error':
                    logger.error(f"🎤 THREAD: Transcription error: {result}")
                    return self._create_empty_transcription_result()
            except:
                logger.error("🎤 THREAD: Transcription timeout")
                return self._create_empty_transcription_result()
                
        except Exception as e:
            logger.error(f"🎤 THREAD: Communication error: {e}")
            return self._create_empty_transcription_result()
            
    def _create_transcription_result(self, text: str, language: str, confidence: float):
        """Create a transcription result object."""
        class TranscriptionResult:
            def __init__(self, text: str, language: str, confidence: float):
                self.text = text
                self.language = language
                self.confidence = confidence
                
        return TranscriptionResult(text, language, confidence)
        
    def _create_empty_transcription_result(self):
        """Create an empty transcription result."""
        return self._create_transcription_result("", "en", 0.0)
        
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

    async def _transcribe_with_engine(self, audio_data: np.ndarray, language: Optional[str] = None):
        """
        Transcribe audio using the configured STT engine (like WhisperLiveThreadingSTT).
        
        Args:
            audio_data: Audio samples as numpy array
            language: Target language (None for auto-detect)
            
        Returns:
            Transcription result object
        """
        try:
            print(f"🎤 ENGINE: Starting transcription with {type(self.stt_engine).__name__}")
            print(f"🎤 ENGINE: Audio shape: {audio_data.shape}, dtype: {audio_data.dtype}")
            
            # Ensure audio is 1D mono
            if audio_data.ndim > 1:
                print(f"🎤 ENGINE: Converting {audio_data.shape} to 1D mono")
                audio_data = audio_data.flatten() if audio_data.shape[0] == 1 else audio_data[0]
                
            # Convert to the format expected by the STT engine
            if audio_data.dtype == np.int16:
                print(f"🎤 ENGINE: Converting int16 to float32")
                audio_data = audio_data.astype(np.float32) / 32768.0
            elif audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)
                
            print(f"🎤 ENGINE: Prepared audio - shape: {audio_data.shape}, range: [{np.min(audio_data):.3f}, {np.max(audio_data):.3f}]")
            
            # Call the STT engine's transcribe method
            if hasattr(self.stt_engine, 'transcribe_audio_async'):
                print(f"🎤 ENGINE: Using transcribe_audio_async method")
                result = await self.stt_engine.transcribe_audio_async(audio_data, 16000)
            elif asyncio.iscoroutinefunction(self.stt_engine.transcribe):
                print(f"🎤 ENGINE: Using async transcribe method") 
                result = await self.stt_engine.transcribe(audio_data)
            else:
                print(f"🎤 ENGINE: Using sync transcribe method in thread")
                result = await asyncio.to_thread(self.stt_engine.transcribe, audio_data)
                
            print(f"🎤 ENGINE: Transcription completed! Result: {result}")
            return result
            
        except Exception as e:
            print(f"🎤 ENGINE: Transcription error: {e}")
            logger.error(f"STT engine transcription error: {e}")
            raise
            
    def _create_transcription_chunk(self, generation_id: int, text: str, confidence: float, state) -> TranscriptionChunk:
        """Create a transcription chunk and update generation state."""
        # Mark STT as complete
        state.stt_complete.set()
        state.input_text = text
        
        logger.info(f"🎤 STT completed for generation {generation_id}: '{text}' "
                   f"(confidence: {confidence:.2f})")
        
        return TranscriptionChunk(
            generation_id=generation_id,
            text=text,
            confidence=confidence,
            is_partial=False
        )
    
    def _add_to_sentence_buffer(self, generation_id: int, current_text: str) -> str:
        """Add text to sentence buffer and return complete sentences if found."""
        if not current_text.strip():
            return ""
        
        # Initialize buffer if needed
        if generation_id not in self.sentence_buffers:
            self.sentence_buffers[generation_id] = ""
        
        # Add current text to buffer
        current_buffer = self.sentence_buffers[generation_id]
        
        # Check buffer size limit
        if len(current_buffer) + len(current_text) > self.max_buffer_length:
            logger.warning(f"Buffer size limit reached for generation {generation_id}, clearing buffer")
            self.sentence_buffers[generation_id] = current_text
        else:
            self.sentence_buffers[generation_id] += (" " + current_text) if current_buffer else current_text
        
        # Update timestamp
        self.buffer_timestamps[generation_id] = time.time()
        
        # Extract complete sentences
        complete_sentences, remaining_fragment = self._extract_complete_sentences(self.sentence_buffers[generation_id])
        
        # Update buffer with remaining fragment
        self.sentence_buffers[generation_id] = remaining_fragment
        
        # Clear timestamp if buffer is empty
        if not self.sentence_buffers[generation_id]:
            self.buffer_timestamps.pop(generation_id, None)
        
        return complete_sentences
    
    def _extract_complete_sentences(self, text: str) -> tuple:
        """Extract complete sentences from text (same logic as async handler)."""
        if not text.strip():
            return "", ""
        
        # Enhanced sentence ending patterns for conversational speech
        sentence_endings = ['.', '!', '?', '...', ', right?', ', you know?', ', okay?']
        
        # Conversational completeness indicators
        conversational_endings = [
            ', right', ', okay', ', you know', ', yeah', ', sure', 
            ' though', ' then', ' so', ' well', ' actually'
        ]
        
        # Find the last occurrence of any strong sentence ending
        last_ending_pos = -1
        ending_found = None
        for ending in sentence_endings:
            pos = text.rfind(ending)
            if pos > last_ending_pos:
                last_ending_pos = pos
                ending_found = ending
        
        # If no strong ending found, check for conversational patterns
        if last_ending_pos == -1:
            # Check for conversational completeness indicators
            for ending in conversational_endings:
                if text.lower().endswith(ending.lower()):
                    # Treat as complete if it seems like a conversational turn
                    words = text.split()
                    if len(words) >= 3:  # Minimum words for complete thought
                        return text.strip(), ""
            
            # No clear ending, but check if text seems long enough to be complete
            words = text.split()
            if len(words) >= 8:  # Long enough phrase might be complete
                # Look for natural break points (commas, "and", "but", etc.)
                break_words = ['and', 'but', 'so', 'then', 'because', 'since', 'while']
                for i, word in enumerate(reversed(words[-4:])):
                    if word.lower() in break_words and len(words) - i >= 4:
                        # Found a natural break point near the end
                        break_point = len(words) - i
                        complete_part = ' '.join(words[:break_point]).strip()
                        remaining_part = ' '.join(words[break_point:]).strip()
                        return complete_part, remaining_part
            
            # Still no clear completion, treat as incomplete fragment
            return "", text.strip()
        
        # Found a strong sentence ending
        complete_part = text[:last_ending_pos + len(ending_found)].strip()
        remaining_part = text[last_ending_pos + len(ending_found):].strip()
        
        return complete_part, remaining_part
    
    def _check_sentence_buffer_timeout(self, generation_id: int) -> str:
        """Check if sentence buffer should be flushed due to timeout."""
        if generation_id not in self.sentence_buffers or not self.sentence_buffers[generation_id]:
            return ""
        
        if generation_id not in self.buffer_timestamps:
            return ""
        
        # Check if buffer has timed out
        if time.time() - self.buffer_timestamps[generation_id] > self.buffer_timeout:
            logger.info(f"🎤 STT TIMEOUT: Processing buffered content for generation {generation_id}")
            return self._flush_sentence_buffer(generation_id)
        
        return ""
    
    def _flush_sentence_buffer(self, generation_id: int) -> str:
        """Flush sentence buffer and return content."""
        if generation_id not in self.sentence_buffers:
            return ""
        
        content = self.sentence_buffers[generation_id]
        self.sentence_buffers.pop(generation_id, None)
        self.buffer_timestamps.pop(generation_id, None)
        
        return content
    
    def cleanup_buffers(self, generation_id: int):
        """Clean up audio buffers for a specific generation."""
        if generation_id in self.audio_buffers:
            del self.audio_buffers[generation_id]
            logger.debug(f"Cleaned up audio buffer for generation {generation_id}")
        
        # Also clean up sentence buffers
        self.sentence_buffers.pop(generation_id, None)
        self.buffer_timestamps.pop(generation_id, None)
            
    def handle_processing_error(self, item, error):
        """Handle STT processing errors."""
        super().handle_processing_error(item, error)
        
        # Clean up buffers for failed generation
        if hasattr(item, 'generation_id'):
            self.cleanup_buffers(item.generation_id)
            
    async def _check_all_buffer_timeouts(self):
        """Check all sentence buffers for timeouts and flush if needed."""
        expired_generations = []
        
        for generation_id, timestamp in self.buffer_timestamps.items():
            if time.time() - timestamp > self.buffer_timeout:
                expired_generations.append(generation_id)
        
        # Process expired buffers
        for generation_id in expired_generations:
            buffered_text = self._flush_sentence_buffer(generation_id)
            if buffered_text:
                # Create transcription chunk from expired buffer
                state = self.pipeline_manager.get_generation_state(generation_id)
                if state and not state.interrupted.is_set():
                    result = self._create_transcription_chunk(generation_id, buffered_text, 1.0, state)
                    if result and self.output_queue:
                        await asyncio.to_thread(self.output_queue.put, result)
    
    def get_worker_stats(self) -> dict:
        """Get STT worker statistics."""
        stats = super().get_worker_stats()
        stats.update({
            "confidence_threshold": self.confidence_threshold,
            "min_audio_length": self.min_audio_length,
            "buffered_generations": len(self.audio_buffers),
            "sentence_buffers": len(self.sentence_buffers),
            "buffer_timeout": self.buffer_timeout,
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