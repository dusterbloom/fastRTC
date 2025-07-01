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
    AudioPipelineManager,
    AudioChunk,
    TTSAudioChunk,
    GenerationStatus,
)
from ..core.pipeline_workers import InterruptionManager
from ..core.stt_worker import STTStreamingWorker

# WhisperLive threading handled via threading STT engine
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
        max_queue_size: int = 100,
        event_loop=None,
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
        self.event_loop = event_loop

        # Pipeline components
        self.pipeline_manager = AudioPipelineManager(max_queue_size=max_queue_size)
        self.interruption_manager = InterruptionManager(self.pipeline_manager)

        # Worker threads
        self.stt_worker = None
        self.llm_worker = None
        self.tts_worker = None
        self.output_worker = None
        
        # WhisperLive TranscriptionClient (like voiceagent example)
        self.whisper_live_client = None
        self.whisper_live_active = False
        self.last_transcription = ""
        self.whisperlive_mode = False  # Control flag for WhisperLive mode
        
        # Lock for transcription updates
        self.transcription_lock = threading.Lock()

        # Current generation tracking
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

        # Skip STT worker initialization when using WhisperLive VAD
        # WhisperLive handles both VAD and STT, so FastRTC STT workers are not needed
        import os

        stt_backend = os.environ.get("STT_BACKEND", "faster").lower()
        use_whisper_live_vad = os.environ.get("WHISPER_LIVE_VAD", "true").lower() == "true"
        
        print(f"🎤 DEBUG: STT_BACKEND = '{stt_backend}'")
        print(f"🎤 DEBUG: WHISPER_LIVE_VAD = '{use_whisper_live_vad}'")
        
        if stt_backend in ["whisper_live", "whisper-live"] and use_whisper_live_vad:
            print("🎤 DEBUG: Using WhisperLive VAD - skipping FastRTC STT worker initialization")
            logger.info("🎤 WhisperLive VAD enabled - disabling FastRTC STT workers to avoid conflicts")
            
            # Set stt_worker to None since WhisperLive handles STT directly
            self.stt_worker = None
            
            # Initialize WhisperLive TranscriptionClient (like working voiceagent example)
            try:
                print("🎤 DEBUG: Initializing WhisperLive TranscriptionClient")
                from whisper_live.client import TranscriptionClient
                
                # Configuration from environment
                from dotenv import load_dotenv
                import os
                env_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", ".env.development")
                load_dotenv(env_path)
                
                self.whisper_live_host = os.environ.get("WHISPER_LIVE_HOST", "localhost")
                self.whisper_live_port = int(os.environ.get("WHISPER_LIVE_PORT", "9091"))
                
                print(f"🎤 DEBUG: WhisperLive config - Host: {self.whisper_live_host}, Port: {self.whisper_live_port}")
                
                # This will be initialized when we start streaming
                self.whisper_live_client = None
                self.whisper_live_active = False
                
                print("🎤 DEBUG: WhisperLive TranscriptionClient configuration ready")
                logger.info("✅ WhisperLive TranscriptionClient ready for microphone streaming")
                    
            except Exception as e:
                print(f"🎤 ERROR: Failed to initialize WhisperLive TranscriptionClient: {e}")
                logger.error(f"Failed to initialize WhisperLive TranscriptionClient: {e}")
                self.whisper_live_client = None
            
            print("🎤 DEBUG: FastRTC STT worker disabled - WhisperLive handles STT directly")
            
        else:
            print("🎤 DEBUG: WhisperLive VAD disabled - initializing FastRTC STT worker")
            logger.info("🎤 Using standard faster-whisper STT worker for threading pipeline")
            self.stt_worker = STTStreamingWorker(
                pipeline_manager=self.pipeline_manager,
                stt_engine=self.stt_engine,
                event_loop=self.event_loop,
            )

        self.llm_worker = LLMStreamingWorker(
            pipeline_manager=self.pipeline_manager, voice_assistant=self.voice_assistant
        )

        self.tts_worker = TTSStreamingWorker(
            pipeline_manager=self.pipeline_manager,
            tts_engine=self.tts_engine,
            voice_mapper=self.voice_mapper,
            voice_assistant=self.voice_assistant,
        )

        # Note: TTSOutputWorker not needed - TTS worker puts directly to output queue
        self.output_worker = None

        # Initialize and start workers
        if self.stt_worker is not None:
            print("🧵 DEBUG: Initializing STT worker...")
            if hasattr(self.stt_worker, "initialize"):
                # Skip async initialization in threading mode - use sync init instead
                print("🧵 DEBUG: Skipping async initialization for threading mode")
                pass

            print("🧵 DEBUG: Starting STT worker...")
            self.stt_worker.start_worker()
        else:
            print("🧵 DEBUG: STT worker disabled - WhisperLive handles STT directly")
            
        print("🧵 DEBUG: Starting LLM worker...")
        self.llm_worker.start_worker()
        print("🧵 DEBUG: Starting TTS worker...")
        self.tts_worker.start_worker()
        print("🧵 DEBUG: All workers started!")

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
        workers = [self.llm_worker, self.tts_worker]
        if self.stt_worker is not None:
            workers.insert(0, self.stt_worker)
            
        for worker in workers:
            if worker:
                # Clean up resources if needed
                if hasattr(worker, "cleanup"):
                    # Skip async cleanup in threading mode
                    print("🧵 DEBUG: Skipping async cleanup for threading mode")
                    pass
                worker.stop_worker()

        # Stop pipeline manager
        self.pipeline_manager.stop()
        
        # Clean up WhisperLive TranscriptionClient if it exists
        if self.whisper_live_client is not None:
            print("🧹 DEBUG: Cleaning up WhisperLive TranscriptionClient")
            try:
                self.whisper_live_client.close_all_clients()
                self.whisper_live_active = False
                logger.info("✅ WhisperLive TranscriptionClient cleaned up")
            except Exception as e:
                logger.error(f"Error cleaning up WhisperLive TranscriptionClient: {e}")
            self.whisper_live_client = None

        logger.info("✅ Threading pipeline stopped")

    def process_audio_stream(
        self, audio_data_tuple: tuple
    ) -> Generator[Tuple[Tuple[int, np.ndarray], AdditionalOutputs], None, None]:
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
        # DEBUG: Print to console to ensure we see it
        print(f"🎤 THREADING CALLBACK INVOKED: {type(audio_data_tuple)}")

        # Interrupt ongoing TTS when user starts speaking
        self.tts_engine.interrupt()

        self.total_callbacks += 1
        start_time = time.time()

        # Handle interruption first
        self.interruption_manager.handle_user_speech_detected()

        # DEBUG: Log callback invocation
        print(f"🎤 THREADING CALLBACK: Processing audio data {self.total_callbacks}")
        logger.debug(f"🎤 CALLBACK INVOKED [{time.time():.3f}]: Processing audio data")

        try:
            # Check if we're in WhisperLive mode (frontend-controlled)
            if self.whisperlive_mode:
                print("🎤 THREADING: WhisperLive mode active - skipping WebRTC audio processing")
                logger.info("🎤 WhisperLive mode: Audio processing handled by TranscriptionClient")
                
                # FastRTC needs to return empty audio to maintain the connection
                # WhisperLive handles microphone directly, we just maintain WebRTC for output
                yield EMPTY_AUDIO_YIELD_OUTPUT
                return
            
            # Check if we're using legacy WhisperLive VAD mode (STT worker disabled)
            if self.stt_worker is None:
                print("🎤 THREADING: Legacy WhisperLive mode - STT worker disabled")
                logger.info("🎤 Legacy WhisperLive mode active")
                
                # FastRTC needs to return empty audio to maintain the connection
                yield EMPTY_AUDIO_YIELD_OUTPUT
                return

            # Standard FastRTC pipeline processing (when STT worker is enabled)
            # Preprocess audio
            audio_array, sample_rate = self._preprocess_audio(audio_data_tuple)
            print(
                f"🎤 THREADING: Preprocessed audio: {audio_array.shape if audio_array is not None else None}"
            )
            if audio_array is None:
                print("🎤 THREADING: Audio preprocessing failed!")
                yield EMPTY_AUDIO_YIELD_OUTPUT
                return

            # Create new generation
            generation_id = self.pipeline_manager.create_generation()
            self.current_generation_id = generation_id

            print(f"🎤 THREADING: Created generation {generation_id}")
            logger.info(f"🎤 Created generation {generation_id} for audio processing")

            # Create audio chunk and feed to pipeline
            audio_chunk = AudioChunk(
                generation_id=generation_id,
                sample_rate=sample_rate,
                audio_data=audio_array,
                timestamp=time.time(),
                is_final=True,  # For now, treat each callback as complete audio
            )

            # Put audio into pipeline
            print(
                f"🎤 THREADING: Putting audio chunk into pipeline for generation {generation_id}"
            )
            logger.info(
                f"🎤 Threading handler putting audio chunk into pipeline for generation {generation_id}"
            )
            if not self.pipeline_manager.put_audio_input(audio_chunk):
                print(
                    f"🎤 THREADING: Failed to queue audio for generation {generation_id}"
                )
                logger.error(f"Failed to queue audio for generation {generation_id}")
                yield EMPTY_AUDIO_YIELD_OUTPUT
                return
            print(
                f"🎤 THREADING: Audio chunk queued successfully for generation {generation_id}"
            )
            logger.info(
                f"✅ Audio chunk queued successfully for generation {generation_id}"
            )

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

                    logger.debug(
                        f"📢 Yielded chunk {yielded_chunks} for generation {generation_id}"
                    )

                    # Check if this was the final chunk
                    if output_chunk.is_final:
                        logger.info(
                            f"✅ Final chunk yielded for generation {generation_id}"
                        )
                        break

                # Check if generation is complete
                if state and state.status == GenerationStatus.COMPLETED:
                    logger.info(f"✅ Generation {generation_id} completed")
                    break

                # Check if generation failed
                if state and state.status == GenerationStatus.FAILED:
                    logger.error(
                        f"❌ Generation {generation_id} failed: {state.error_message}"
                    )
                    break

            # If no chunks were yielded, return empty
            if yielded_chunks == 0:
                logger.warning(
                    f"No audio chunks yielded for generation {generation_id}"
                )
                yield EMPTY_AUDIO_YIELD_OUTPUT
            else:
                self.successful_callbacks += 1

            # Log processing time
            processing_time = time.time() - start_time
            logger.debug(f"⏱️ Total callback processing time: {processing_time:.3f}s")

        except Exception as e:
            logger.error(f"❌ Error in audio stream processing: {e}")
            yield EMPTY_AUDIO_YIELD_OUTPUT

    def _start_whisperlive_microphone(self):
        """Start WhisperLive microphone streaming (like working voiceagent example)."""
        try:
            if self.whisper_live_active:
                print("🎤 MICROPHONE: WhisperLive already active")
                return
                
            print("🎤 MICROPHONE: Initializing WhisperLive TranscriptionClient for microphone")
            from whisper_live.client import TranscriptionClient
            
            # Initialize TranscriptionClient with microphone input (like voiceagent example)
            self.whisper_live_client = TranscriptionClient(
                host=self.whisper_live_host,
                port=self.whisper_live_port,
                lang="en",
                translate=False,
                model="base",
                use_vad=True,
                save_output_recording=False,
                log_transcription=False,
                transcription_callback=self._whisperlive_transcription_callback,
                # Gentle VAD settings for better sensitivity (from voiceagent)
                send_last_n_segments=5,
                no_speech_thresh=0.3,  # More sensitive
                clip_audio=False,
                same_output_threshold=5,
            )
            
            self.whisper_live_active = True
            print("🎤 MICROPHONE: WhisperLive TranscriptionClient initialized")
            
            # Start microphone streaming in separate thread (like voiceagent)
            def stream_microphone():
                try:
                    print("🎤 MICROPHONE: Starting microphone streaming thread")
                    # Call client() with no parameters for microphone input
                    self.whisper_live_client()
                    print("🎤 MICROPHONE: Microphone streaming finished")
                except Exception as e:
                    print(f"🎤 MICROPHONE ERROR: Streaming error: {e}")
                    logger.error(f"WhisperLive microphone streaming error: {e}")
                    self.whisper_live_active = False
            
            # Start streaming thread
            streaming_thread = threading.Thread(target=stream_microphone, daemon=True)
            streaming_thread.start()
            
            print("🎤 MICROPHONE: WhisperLive microphone streaming started successfully")
            logger.info("✅ WhisperLive microphone streaming started")
            
        except Exception as e:
            print(f"🎤 MICROPHONE ERROR: Failed to start microphone streaming: {e}")
            logger.error(f"Failed to start WhisperLive microphone streaming: {e}")
            self.whisper_live_active = False
            self.whisper_live_client = None

    def _whisperlive_transcription_callback(self, text, segments):
        """Transcription callback from WhisperLive (like voiceagent example)."""
        try:
            if text and text.strip():
                print(f"🎤 TRANSCRIPTION CALLBACK: '{text}'")
                logger.info(f"🎤 WhisperLive transcription: {text}")
                
                # Store transcription for processing in main thread
                with self.transcription_lock:
                    self.last_transcription = text.strip()
                
                # Only process complete sentences (like voiceagent)
                cleaned_text = text.strip()
                if self._is_complete_sentence(cleaned_text):
                    print(f"🎤 COMPLETE SENTENCE: '{cleaned_text}'")
                    logger.info(f"🎤 Processing complete sentence: {cleaned_text}")
                    self._send_transcription_to_llm(cleaned_text)
                else:
                    print(f"🎤 PARTIAL: '{cleaned_text}' (waiting for completion)")
                    
        except Exception as e:
            print(f"🎤 TRANSCRIPTION CALLBACK ERROR: {e}")
            logger.error(f"WhisperLive transcription callback error: {e}")

    def _is_complete_sentence(self, text: str) -> bool:
        """Check if transcription appears to be a complete sentence."""
        if not text:
            return False
            
        # Remove leading/trailing whitespace
        text = text.strip()
        
        # Check for sentence-ending punctuation
        if text.endswith(('.', '!', '?')):
            return True
            
        # Check for pause indicators (multiple spaces, which might indicate silence)
        if '  ' in text:  # Double space often indicates a pause
            return True
            
        # Check minimum length for processing (avoid processing single words)
        if len(text.split()) >= 3:  # At least 3 words
            return True
            
        return False

    def start_whisperlive_streaming(self):
        """Start WhisperLive microphone streaming (API endpoint)."""
        try:
            if self.whisper_live_active:
                print("🎤 API: WhisperLive already active")
                return {"status": "already_active", "message": "WhisperLive is already streaming"}
                
            print("🎤 API: Starting WhisperLive microphone streaming")
            from whisper_live.client import TranscriptionClient
            
            # Configuration from environment (like voiceagent)
            from dotenv import load_dotenv
            import os
            env_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", ".env.development")
            load_dotenv(env_path)
            
            whisper_live_host = os.environ.get("WHISPER_LIVE_HOST", "localhost")
            whisper_live_port = int(os.environ.get("WHISPER_LIVE_PORT", "9091"))
            
            print(f"🎤 API: WhisperLive config - Host: {whisper_live_host}, Port: {whisper_live_port}")
            
            # Initialize TranscriptionClient with microphone input (exactly like voiceagent)
            self.whisper_live_client = TranscriptionClient(
                host=whisper_live_host,
                port=whisper_live_port,
                lang="en",
                translate=False,
                model="base",
                use_vad=True,
                save_output_recording=False,
                log_transcription=False,
                transcription_callback=self._whisperlive_transcription_callback,
                # Gentle VAD settings for better sensitivity (from voiceagent)
                send_last_n_segments=5,
                no_speech_thresh=0.3,  # More sensitive
                clip_audio=False,
                same_output_threshold=5,
            )
            
            self.whisper_live_active = True
            self.whisperlive_mode = True
            print("🎤 API: WhisperLive TranscriptionClient initialized")
            
            # Start microphone streaming in separate thread (like voiceagent)
            def stream_microphone():
                try:
                    print("🎤 API: Starting microphone streaming thread")
                    # Call client() with no parameters for microphone input
                    self.whisper_live_client()
                    print("🎤 API: Microphone streaming finished")
                except Exception as e:
                    print(f"🎤 API STREAMING ERROR: {e}")
                    logger.error(f"WhisperLive microphone streaming error: {e}")
                    self.whisper_live_active = False
            
            # Start streaming thread
            streaming_thread = threading.Thread(target=stream_microphone, daemon=True)
            streaming_thread.start()
            
            print("🎤 API: WhisperLive microphone streaming started successfully")
            logger.info("✅ WhisperLive microphone streaming started via API")
            
            return {"status": "started", "message": "WhisperLive streaming started successfully"}
            
        except Exception as e:
            print(f"🎤 API ERROR: Failed to start microphone streaming: {e}")
            logger.error(f"Failed to start WhisperLive microphone streaming via API: {e}")
            self.whisper_live_active = False
            self.whisper_live_client = None
            self.whisperlive_mode = False
            return {"status": "error", "message": f"Failed to start WhisperLive: {str(e)}"}

    def stop_whisperlive_streaming(self):
        """Stop WhisperLive microphone streaming (API endpoint)."""
        try:
            if not self.whisper_live_active:
                print("🎤 API: WhisperLive not active")
                return {"status": "not_active", "message": "WhisperLive is not currently streaming"}
            
            print("🎤 API: Stopping WhisperLive microphone streaming")
            self.whisper_live_active = False
            self.whisperlive_mode = False
            
            if self.whisper_live_client:
                self.whisper_live_client.close_all_clients()
                self.whisper_live_client = None
            
            print("🎤 API: WhisperLive microphone streaming stopped successfully")
            logger.info("✅ WhisperLive microphone streaming stopped via API")
            
            return {"status": "stopped", "message": "WhisperLive streaming stopped successfully"}
            
        except Exception as e:
            print(f"🎤 API ERROR: Failed to stop microphone streaming: {e}")
            logger.error(f"Failed to stop WhisperLive microphone streaming via API: {e}")
            return {"status": "error", "message": f"Failed to stop WhisperLive: {str(e)}"}


    def _send_transcription_to_llm(self, transcription_text: str):
        """Send transcription to LLM worker for processing."""
        try:
            if self.llm_worker and transcription_text.strip():
                print(f"🧠 LLM: Sending transcription to LLM worker: '{transcription_text}'")
                
                # Create a generation for the transcription
                generation_id = self.pipeline_manager.create_generation()
                print(f"🧠 LLM: Created generation {generation_id}")
                
                # Send transcription to LLM worker using TranscriptionChunk
                from ..core.pipeline_manager import TranscriptionChunk
                import time
                
                # Create transcription chunk for LLM processing
                transcription_chunk = TranscriptionChunk(
                    generation_id=generation_id,
                    text=transcription_text,
                    confidence=1.0,  # High confidence from WhisperLive
                    is_partial=False,  # Complete sentence from WhisperLive
                    timestamp=time.time()
                )
                
                # Put directly into transcription queue so LLM worker picks it up
                try:
                    print(f"🧠 LLM DEBUG: Queue size before put: {self.pipeline_manager.transcription_queue.qsize()}")
                    self.pipeline_manager.transcription_queue.put(transcription_chunk, timeout=1.0)
                    success = True
                    print(f"🧠 LLM DEBUG: Queue size after put: {self.pipeline_manager.transcription_queue.qsize()}")
                except Exception as queue_error:
                    print(f"🧠 LLM ERROR: Failed to queue transcription: {queue_error}")
                    success = False
                if success:
                    print(f"🧠 LLM: Transcription queued for generation {generation_id}")
                    logger.info(f"🧠 LLM processing generation {generation_id}: {transcription_text}")
                    # Debug: Check if LLM worker is actually running
                    if self.llm_worker:
                        worker_stats = self.llm_worker.get_worker_stats()
                        print(f"🧠 LLM DEBUG: Worker stats: {worker_stats}")
                    else:
                        print(f"🧠 LLM ERROR: LLM worker is None!")
                else:
                    print(f"🧠 LLM ERROR: Failed to queue transcription for generation {generation_id}")
                    logger.error(f"Failed to queue transcription for generation {generation_id}")
                
        except Exception as e:
            print(f"🧠 LLM ERROR: Failed to send transcription to LLM: {e}")
            logger.error(f"LLM worker error: {e}")
            import traceback
            traceback.print_exc()

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
                logger.error(
                    f"Invalid audio data format: expected tuple of length 2, got {type(audio_data_tuple)}"
                )
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
        success_rate = (
            self.successful_callbacks / self.total_callbacks
            if self.total_callbacks > 0
            else 0
        )

        return {
            "running": self.pipeline_manager.running,
            "runtime_seconds": runtime,
            "total_callbacks": self.total_callbacks,
            "successful_callbacks": self.successful_callbacks,
            "success_rate": success_rate,
            "current_generation": self.current_generation_id,
            "pipeline_stats": self.pipeline_manager.get_pipeline_stats(),
            "worker_stats": {
                "stt": self.stt_worker.get_worker_stats() if self.stt_worker is not None else "disabled_whisper_live_vad",
                "llm": self.llm_worker.get_worker_stats() if self.llm_worker else None,
                "tts": self.tts_worker.get_worker_stats() if self.tts_worker else None,
            },
        }
