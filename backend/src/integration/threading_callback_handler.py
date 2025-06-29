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
        logger.info(f"🎤 THREADING CALLBACK INVOKED [{time.time():.3f}]: Processing audio data")
        print(f"🎤 THREADING CALLBACK INVOKED [{time.time():.3f}]: Processing audio data")
        print(f"🔍 Audio data type: {type(audio_data_tuple)}, length: {len(audio_data_tuple) if hasattr(audio_data_tuple, '__len__') else 'N/A'}")
        
        try:
            # Preprocess audio
            print(f"🔍 Raw audio_data_tuple: {type(audio_data_tuple)}, content: {audio_data_tuple if len(str(audio_data_tuple)) < 200 else str(audio_data_tuple)[:200] + '...'}")
            audio_array, sample_rate = self._preprocess_audio(audio_data_tuple)
            print(f"🔍 Processed audio_array: type={type(audio_array)}, shape={getattr(audio_array, 'shape', 'no shape')}, size={getattr(audio_array, 'size', len(audio_array) if audio_array is not None else 'None')}")
            if audio_array is None:
                print("❌ Audio array is None, yielding empty")
                yield EMPTY_AUDIO_YIELD_OUTPUT
                return
            
            # CRITICAL FIX: Set audio buffer for voice authentication
            print(f"🔍 Checking voice_print_manager: hasattr={hasattr(self.voice_assistant, 'voice_print_manager')}")
            if hasattr(self.voice_assistant, 'voice_print_manager'):
                print(f"🔍 voice_print_manager exists: {self.voice_assistant.voice_print_manager is not None}")
            
            if hasattr(self.voice_assistant, 'voice_print_manager') and self.voice_assistant.voice_print_manager:
                # Flatten audio array if needed for voice authentication
                if len(audio_array.shape) > 1:
                    flattened_audio = audio_array.flatten()
                    print(f"🔧 Flattened audio from {audio_array.shape} to {flattened_audio.shape}")
                else:
                    flattened_audio = audio_array
                
                # Check if audio contains actual speech (not just silence)
                audio_max = abs(flattened_audio).max() if len(flattened_audio) > 0 else 0
                audio_rms = (flattened_audio ** 2).mean() ** 0.5 if len(flattened_audio) > 0 else 0
                print(f"🔊 Audio analysis: max={audio_max:.6f}, rms={audio_rms:.6f}, non_zero_samples={(flattened_audio != 0).sum()}")
                
                self.voice_assistant.voice_print_manager.set_audio_buffer(flattened_audio)
                print(f"✅ [THREADING] Audio buffer set for voice authentication (size: {len(flattened_audio)})")
                print(f"🔍 [THREADING] Voice auth enabled: {self.voice_assistant.voice_print_manager.enable_voice_auth}")
                print(f"🔍 [THREADING] Voice manager available: {self.voice_assistant.voice_print_manager.voice_manager is not None}")
                logger.info(f"🎤 [THREADING] Audio buffer set for voice authentication (size: {len(flattened_audio)})")
                logger.info(f"🔍 [THREADING] Voice auth enabled: {self.voice_assistant.voice_print_manager.enable_voice_auth}")
                logger.info(f"🔍 [THREADING] Voice manager available: {self.voice_assistant.voice_print_manager.voice_manager is not None}")
            else:
                print(f"❌ [THREADING] voice_print_manager not available or disabled")
                logger.warning(f"❌ [THREADING] voice_print_manager not available or disabled")
                
            # Create new generation
            generation_id = self.pipeline_manager.create_generation()
            self.current_generation_id = generation_id
            
            print(f"🎯 Created generation {generation_id} for audio processing")
            logger.info(f"🎤 Created generation {generation_id} for audio processing")
            
            # CRITICAL DEBUG: Analyze audio before creating chunk
            print(f"🔍 [THREADING HANDLER] Audio analysis before creating chunk:")
            print(f"  - Type: {type(audio_array)}")
            print(f"  - Shape: {audio_array.shape}")
            print(f"  - Size: {audio_array.size}")
            print(f"  - Dtype: {audio_array.dtype}")
            print(f"  - Min: {audio_array.min():.6f}")
            print(f"  - Max: {audio_array.max():.6f}")
            print(f"  - Mean: {audio_array.mean():.6f}")
            print(f"  - RMS: {(audio_array ** 2).mean() ** 0.5:.6f}")
            print(f"  - Non-zero samples: {(audio_array != 0).sum()}")
            print(f"  - Sample rate: {sample_rate}")
            print(f"  - Duration: {audio_array.size / sample_rate:.3f}s")
            
            # Create audio chunk and feed to pipeline
            audio_chunk = AudioChunk(
                generation_id=generation_id,
                sample_rate=sample_rate,
                audio_data=audio_array,
                timestamp=time.time(),
                is_final=True  # For now, treat each callback as complete audio
            )
            
            # CRITICAL DEBUG: Verify chunk creation didn't corrupt data
            print(f"🔍 [THREADING HANDLER] Audio chunk verification after creation:")
            print(f"  - Chunk audio type: {type(audio_chunk.audio_data)}")
            print(f"  - Chunk audio shape: {audio_chunk.audio_data.shape}")
            print(f"  - Chunk audio size: {audio_chunk.audio_data.size}")
            print(f"  - Chunk audio dtype: {audio_chunk.audio_data.dtype}")
            print(f"  - Chunk audio min: {audio_chunk.audio_data.min():.6f}")
            print(f"  - Chunk audio max: {audio_chunk.audio_data.max():.6f}")
            print(f"  - Chunk audio RMS: {(audio_chunk.audio_data ** 2).mean() ** 0.5:.6f}")
            print(f"  - Chunk sample rate: {audio_chunk.sample_rate}")
            
            # Put audio into pipeline
            print(f"🎤 Threading handler putting audio chunk into pipeline for generation {generation_id}")
            logger.info(f"🎤 Threading handler putting audio chunk into pipeline for generation {generation_id}")
            if not self.pipeline_manager.put_audio_input(audio_chunk):
                print(f"❌ Failed to queue audio for generation {generation_id}")
                logger.error(f"Failed to queue audio for generation {generation_id}")
                yield EMPTY_AUDIO_YIELD_OUTPUT
                return
            print(f"✅ Audio chunk queued successfully for generation {generation_id}")
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
                
                # DEBUG: Log pipeline state every 2 seconds
                elapsed = time.time() - timeout_start
                if int(elapsed) % 2 == 0 and elapsed > 1:
                    logger.info(f"🔍 [DEBUG] Generation {generation_id} waiting {elapsed:.1f}s - State: {state.status if state else 'None'}")
                    if state:
                        logger.info(f"🔍 [DEBUG] STT: {'✅' if state.stt_complete.is_set() else '⏳'}, "
                                   f"LLM: {'✅' if state.llm_started.is_set() else '⏳'}, "
                                   f"TTS: {'✅' if state.tts_started.is_set() else '⏳'}")
                    
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
                elapsed = time.time() - timeout_start
                if elapsed >= max_wait_time:
                    logger.error(f"⏰ TIMEOUT: Generation {generation_id} timed out after {elapsed:.1f}s")
                    if state:
                        logger.error(f"⏰ Final state - STT: {'✅' if state.stt_complete.is_set() else '❌'}, "
                                   f"LLM: {'✅' if state.llm_started.is_set() else '❌'}, "
                                   f"TTS: {'✅' if state.tts_started.is_set() else '❌'}, "
                                   f"Status: {state.status}")
                else:
                    logger.warning(f"No audio chunks yielded for generation {generation_id} (waited {elapsed:.1f}s)")
                yield EMPTY_AUDIO_YIELD_OUTPUT
            else:
                self.successful_callbacks += 1
                
            # Log processing time
            processing_time = time.time() - start_time
            logger.debug(f"⏱️ Total callback processing time: {processing_time:.3f}s")
            
        except Exception as e:
            logger.error(f"❌ Error in audio stream processing: {e}")
            import traceback
            logger.error(f"❌ Full traceback: {traceback.format_exc()}")
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