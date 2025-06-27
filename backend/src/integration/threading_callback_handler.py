"""
Threading-Based Callback Handler

Pure threading implementation replacing sync/async callback handler.
Eliminates event loop complexity with queue-based worker coordination.
"""

import time
import numpy as np
import threading
from typing import Tuple, Generator, Any
from fastrtc import AdditionalOutputs

from ..core.pipeline_manager import (
    AudioPipelineManager, AudioChunk, TTSAudioChunk, 
    GenerationStatus
)
from ..core.pipeline_workers import InterruptionManager
from ..core.stt_worker import STTStreamingWorker  
from ..core.llm_worker import LLMStreamingWorker
from ..core.tts_worker import TTSStreamingWorker, TTSOutputWorker
from ..audio import STTEngine, KokoroTTSEngine, VoiceMapper
from ..config.audio_config import AUDIO_SAMPLE_RATE, SILENT_AUDIO_FRAME_TUPLE
from ..utils.logging import get_logger

logger = get_logger(__name__)

# Audio constants
EMPTY_AUDIO_YIELD_OUTPUT = (SILENT_AUDIO_FRAME_TUPLE, AdditionalOutputs())


class ThreadingCallbackHandler:
    """
    Pure threading-based callback handler for fastRTC audio processing.
    
    Replaces the sync/async bridging complexity with event-driven worker threads
    and queue-based communication. Preserves existing module interfaces while
    eliminating performance bottlenecks.
    """
    
    def __init__(
        self,
        voice_assistant,
        stt_engine: STTEngine,
        tts_engine: KokoroTTSEngine,
        voice_mapper: VoiceMapper,
        max_queue_size: int = 100
    ):
        """
        Initialize threading-based callback handler.
        
        Args:
            voice_assistant: Voice assistant instance
            stt_engine: Speech-to-text engine
            tts_engine: Text-to-speech engine  
            voice_mapper: Voice mapping component
            max_queue_size: Maximum queue size for pipeline
        """
        self.voice_assistant = voice_assistant
        self.stt_engine = stt_engine
        self.tts_engine = tts_engine
        self.voice_mapper = voice_mapper
        
        # Pipeline components
        self.pipeline_manager = AudioPipelineManager(max_queue_size=max_queue_size)
        self.interruption_manager = InterruptionManager(self.pipeline_manager)
        
        # Worker threads
        self.stt_worker = None
        self.llm_worker = None
        self.tts_worker = None
        self.output_worker = None
        
        # Audio output management
        self.output_chunks = []
        self.output_lock = threading.Lock()
        self.current_generation_id = None
        
        # Performance tracking
        self.total_callbacks = 0
        self.successful_callbacks = 0
        self.start_time = time.time()
        
        logger.info("🎛️ ThreadingCallbackHandler initialized")
        
    def start(self):
        """Start the threading pipeline."""
        if self.pipeline_manager.running:
            logger.warning("Threading pipeline already running")
            return
            
        logger.info("🚀 Starting threading pipeline")
        
        # Start pipeline manager
        self.pipeline_manager.start()
        
        # Initialize workers
        self.stt_worker = STTStreamingWorker(
            pipeline_manager=self.pipeline_manager,
            stt_engine=self.stt_engine
        )
        
        self.llm_worker = LLMStreamingWorker(
            pipeline_manager=self.pipeline_manager,
            voice_assistant=self.voice_assistant
        )
        
        self.tts_worker = TTSStreamingWorker(
            pipeline_manager=self.pipeline_manager,
            tts_engine=self.tts_engine,
            voice_mapper=self.voice_mapper,
            voice_assistant=self.voice_assistant
        )
        
        self.output_worker = TTSOutputWorker(
            pipeline_manager=self.pipeline_manager,
            output_callback=self._handle_audio_output
        )
        
        # Start all workers
        self.stt_worker.start_worker()
        self.llm_worker.start_worker()
        self.tts_worker.start_worker()
        self.output_worker.start_worker()
        
        # Setup interruption handling
        self.interruption_manager.register_interruption_callback(
            self._handle_interruption
        )
        
        logger.info("✅ Threading pipeline started")
        
    def stop(self):
        """Stop the threading pipeline."""
        if not self.pipeline_manager.running:
            return
            
        logger.info("🛑 Stopping threading pipeline")
        
        # Stop workers
        workers = [self.stt_worker, self.llm_worker, self.tts_worker, self.output_worker]
        for worker in workers:
            if worker:
                worker.stop_worker()
                
        # Stop pipeline manager
        self.pipeline_manager.stop()
        
        logger.info("✅ Threading pipeline stopped")
        
    def process_audio_stream(self, audio_data_tuple: tuple) -> Generator[Tuple[Tuple[int, np.ndarray], AdditionalOutputs], None, None]:
        """
        Main callback function for processing audio streams in real-time.
        
        This function:
        1. Preprocesses incoming audio data
        2. Creates a new generation request
        3. Feeds audio into the pipeline
        4. Yields output audio chunks as they become available
        
        Args:
            audio_data_tuple: Tuple containing audio data from FastRTC
            
        Yields:
            Tuples of (audio_data, additional_outputs) for streaming back to client
        """
        # Interrupt ongoing TTS when user starts speaking
        self.tts_engine.interrupt()
        
        self.total_callbacks += 1
        start_time = time.time()
        
        # Handle interruption first
        self.interruption_manager.handle_user_speech_detected()
        
        # DEBUG: Log callback invocation
        logger.debug(f"🎤 CALLBACK INVOKED [{time.time():.3f}]: Processing audio data")
        
        try:
            # Preprocess audio
            audio_array, sample_rate = self._preprocess_audio(audio_data_tuple)
            if audio_array is None:
                yield EMPTY_AUDIO_YIELD_OUTPUT
                return
                
            # Create new generation
            generation_id = self.pipeline_manager.create_generation()
            self.current_generation_id = generation_id
            
            logger.info(f"🎤 Created generation {generation_id} for audio processing")
            
            # Create audio chunk and feed to pipeline
            audio_chunk = AudioChunk(
                generation_id=generation_id,
                sample_rate=sample_rate,
                audio_data=audio_array,
                timestamp=time.time(),
                is_final=True  # For now, treat each callback as complete audio
            )
            
            # Put audio into pipeline
            logger.info(f"🎤 Threading handler putting audio chunk into pipeline for generation {generation_id}")
            if not self.pipeline_manager.put_audio_input(audio_chunk):
                logger.error(f"Failed to queue audio for generation {generation_id}")
                yield EMPTY_AUDIO_YIELD_OUTPUT
                return
            logger.info(f"✅ Audio chunk queued successfully for generation {generation_id}")
                
            # Yield audio chunks as they become available
            yielded_chunks = 0
            timeout_start = time.time()
            max_wait_time = 10.0  # Maximum time to wait for response
            
            while time.time() - timeout_start < max_wait_time:
                # Check if generation was interrupted
                state = self.pipeline_manager.get_generation_state(generation_id)
                if state and state.interrupted.is_set():
                    logger.info(f"Generation {generation_id} was interrupted")
                    break
                    
                # Get available output chunks
                output_chunk = self.pipeline_manager.get_output_audio(timeout=0.1)
                if output_chunk and output_chunk.generation_id == generation_id:
                    # Convert to FastRTC format and yield
                    audio_output = (output_chunk.sample_rate, output_chunk.audio_data)
                    yield (audio_output, AdditionalOutputs())
                    yielded_chunks += 1
                    
                    logger.debug(f"📢 Yielded chunk {yielded_chunks} for generation {generation_id}")
                    
                    # Check if this was the final chunk
                    if output_chunk.is_final:
                        logger.info(f"✅ Final chunk yielded for generation {generation_id}")
                        break
                        
                # Check if generation is complete
                if state and state.status == GenerationStatus.COMPLETED:
                    logger.info(f"✅ Generation {generation_id} completed")
                    break
                    
                # Check if generation failed
                if state and state.status == GenerationStatus.FAILED:
                    logger.error(f"❌ Generation {generation_id} failed: {state.error_message}")
                    break
                    
            # If no chunks were yielded, return empty
            if yielded_chunks == 0:
                logger.warning(f"No audio chunks yielded for generation {generation_id}")
                yield EMPTY_AUDIO_YIELD_OUTPUT
            else:
                self.successful_callbacks += 1
                
            # Log processing time
            processing_time = time.time() - start_time
            logger.debug(f"⏱️ Total callback processing time: {processing_time:.3f}s")
            
        except Exception as e:
            logger.error(f"❌ Error in audio stream processing: {e}")
            yield EMPTY_AUDIO_YIELD_OUTPUT
            
    def _preprocess_audio(self, audio_data_tuple: tuple) -> Tuple[np.ndarray, int]:
        """
        Extract audio data directly from FastRTC (1 channel 16000Hz PCM).
        
        Args:
            audio_data_tuple: Raw audio data from FastRTC
            
        Returns:
            Tuple of (audio_array, sample_rate)
        """
        try:
            # FastRTC sends (sample_rate, audio_array) for 1 channel 16000Hz PCM
            if not isinstance(audio_data_tuple, tuple) or len(audio_data_tuple) != 2:
                logger.error(f"Invalid audio data format: expected tuple of length 2, got {type(audio_data_tuple)}")
                return None, None
                
            sample_rate, audio_array = audio_data_tuple
            
            # Convert to numpy array if needed
            if not isinstance(audio_array, np.ndarray):
                audio_array = np.array(audio_array, dtype=np.float32)
            
            # Simple validation
            if audio_array.size == 0:
                logger.debug("Empty audio array received")
                return None, None
                
            logger.debug(f"🎤 Audio passthrough: {audio_array.shape}, {sample_rate}Hz")
            return audio_array, sample_rate
            
        except Exception as e:
            logger.error(f"Error extracting audio data: {e}")
            logger.error(f"Audio data tuple contents: {audio_data_tuple}")
            import traceback
            traceback.print_exc()
            return None, None
            
    def _handle_audio_output(self, tts_chunk: TTSAudioChunk):
        """
        Handle audio output from TTS worker.
        
        Args:
            tts_chunk: TTS audio chunk to handle
        """
        with self.output_lock:
            self.output_chunks.append(tts_chunk)
            
        logger.debug(f"📢 Received audio output for generation {tts_chunk.generation_id}")
        
    def _handle_interruption(self):
        """Handle interruption callback."""
        logger.info("🛑 Interruption callback triggered")
        
        # Interrupt TTS engine directly for immediate effect
        self.tts_engine.interrupt()
        
        # Interrupt TTS worker for cleanup
        if self.tts_worker:
            self.tts_worker.interrupt_all_active()
            
    def get_handler_stats(self) -> dict:
        """Get callback handler statistics."""
        runtime = time.time() - self.start_time
        success_rate = self.successful_callbacks / self.total_callbacks if self.total_callbacks > 0 else 0
        
        return {
            "running": self.pipeline_manager.running,
            "runtime_seconds": runtime,
            "total_callbacks": self.total_callbacks,
            "successful_callbacks": self.successful_callbacks,
            "success_rate": success_rate,
            "current_generation": self.current_generation_id,
            "pipeline_stats": self.pipeline_manager.get_pipeline_stats(),
            "worker_stats": {
                "stt": self.stt_worker.get_worker_stats() if self.stt_worker else None,
                "llm": self.llm_worker.get_worker_stats() if self.llm_worker else None,
                "tts": self.tts_worker.get_worker_stats() if self.tts_worker else None,
                "output": self.output_worker.get_worker_stats() if self.output_worker else None,
            }
        }