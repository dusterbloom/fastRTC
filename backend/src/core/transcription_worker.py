"""
TranscriptionWorker Process

Isolated process for faster-whisper transcription based on RealtimeSTT pattern.
Runs faster-whisper synchronously without async/threading conflicts.
"""

import os
import time
import queue
import logging
import threading
import multiprocessing
import numpy as np
from typing import Optional

# Import faster_whisper
try:
    import faster_whisper
except ImportError:
    raise ImportError("faster-whisper is required for transcription")

from ..utils.logging import get_logger

logger = get_logger(__name__)

TIME_SLEEP = 0.01  # Sleep time for polling loops


class TranscriptionWorker:
    """
    Isolated process worker for faster-whisper transcription.
    
    Based on RealtimeSTT's successful pattern:
    - Runs in separate process (not thread)
    - Uses synchronous faster-whisper calls only
    - Communicates via multiprocessing.Pipe
    - No async/await complexity
    """
    
    def __init__(
        self,
        conn,
        model_path: str = "Systran/faster-whisper-large-v3",
        device: str = "cuda",
        compute_type: str = "int8_float16",
        beam_size: int = 1,
        vad_filter: bool = False,
        language: Optional[str] = None,
        initial_prompt: Optional[str] = None,
        normalize_audio: bool = True,
        shutdown_event=None
    ):
        """
        Initialize transcription worker.
        
        Args:
            conn: Multiprocessing connection for communication
            model_path: Path or name of faster-whisper model
            device: Device to run model on ('cuda' or 'cpu')
            compute_type: Compute type for model
            beam_size: Beam size for transcription
            vad_filter: Whether to use VAD filter
            language: Target language (None for auto-detect)
            initial_prompt: Initial prompt for model
            normalize_audio: Whether to normalize audio
            shutdown_event: Event to signal shutdown
        """
        self.conn = conn
        self.model_path = model_path
        self.device = device
        self.compute_type = compute_type
        self.beam_size = beam_size
        self.vad_filter = vad_filter
        self.language = language
        self.initial_prompt = initial_prompt
        self.normalize_audio = normalize_audio
        self.shutdown_event = shutdown_event or multiprocessing.Event()
        
        # Internal queue for processing requests
        self.queue = queue.Queue()
        
        # Model will be initialized in run()
        self.model = None
        
    def poll_connection(self):
        """Poll the connection for incoming transcription requests."""
        while not self.shutdown_event.is_set():
            try:
                if self.conn.poll(0.01):
                    data = self.conn.recv()
                    self.queue.put(data)
                else:
                    time.sleep(TIME_SLEEP)
            except Exception as e:
                logger.error(f"Error receiving data from connection: {e}")
                time.sleep(TIME_SLEEP)
                
    def run(self):
        """Main worker process loop."""
        # Set up signal handling for clean shutdown
        import signal
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        
        logger.info(f"🎤 Initializing faster-whisper model: {self.model_path}")
        
        try:
            # Handle CUDA initialization in separate process (RealtimeTTS pattern)
            device = self.device
            compute_type = self.compute_type
            
            logger.info(f"🎤 Initializing model with device: {device}, compute_type: {compute_type}")
            logger.info(f"🎤 Model path: {self.model_path}")
            
            # Try CUDA first, fallback to CPU if it fails
            if device == "cuda":
                try:
                    import torch
                    if torch.cuda.is_available():
                        # Initialize CUDA in this process
                        torch.cuda.init()
                        logger.info("🎤 CUDA initialized successfully in worker process")
                    else:
                        logger.warning("🎤 CUDA not available, falling back to CPU")
                        device = "cpu"
                        compute_type = "int8"
                except Exception as cuda_error:
                    logger.warning(f"🎤 CUDA initialization failed: {cuda_error}, falling back to CPU")
                    device = "cpu"
                    compute_type = "int8"
            
            # Initialize faster-whisper model
            logger.info(f"🎤 Loading faster-whisper model: {self.model_path}")
            self.model = faster_whisper.WhisperModel(
                model_size_or_path=self.model_path,
                device=device,
                compute_type=compute_type
            )
            
            logger.info(f"✅ Model initialized with device: {device}, compute_type: {compute_type}")
            
            # Warmup transcription with dummy audio
            logger.info("🎤 Running model warmup...")
            dummy_audio = np.zeros(16000, dtype=np.float32)  # 1 second of silence
            segments, info = self.model.transcribe(dummy_audio, language="en", beam_size=1)
            warmup_text = " ".join(segment.text for segment in segments)
            logger.info(f"🎤 Model warmup completed: '{warmup_text}'")
            
        except Exception as e:
            # Try CPU fallback if everything else fails
            logger.error(f"❌ Error initializing faster-whisper model: {e}")
            if device != "cpu":
                logger.info("🎤 Attempting final fallback to CPU...")
                try:
                    self.model = faster_whisper.WhisperModel(
                        model_size_or_path=self.model_path,
                        device="cpu",
                        compute_type="int8"
                    )
                    logger.info("✅ CPU fallback successful")
                except Exception as cpu_error:
                    logger.error(f"❌ CPU fallback also failed: {cpu_error}")
                    self.conn.send(('error', f"Model initialization failed: {cpu_error}"))
                    return
            else:
                self.conn.send(('error', f"Model initialization failed: {e}"))
                return
            
        logger.info("✅ faster-whisper model initialized successfully")
        
        # Send ready signal
        self.conn.send(('ready', 'Model initialized'))
        
        # Start connection polling thread
        polling_thread = threading.Thread(target=self.poll_connection, daemon=True)
        polling_thread.start()
        
        # Main processing loop
        try:
            while not self.shutdown_event.is_set():
                try:
                    # Get transcription request from queue
                    audio_data, language, use_prompt = self.queue.get(timeout=0.1)
                    
                    try:
                        logger.debug(f"🎤 WORKER: Transcribing audio with language {language}")
                        start_time = time.time()
                        
                        # Validate audio data
                        if audio_data is None or audio_data.size == 0:
                            logger.error("🎤 WORKER: Received empty audio data")
                            self.conn.send(('error', "Empty audio data"))
                            continue
                            
                        # Normalize audio if requested
                        if self.normalize_audio and audio_data.size > 0:
                            peak = np.max(np.abs(audio_data))
                            if peak > 0:
                                audio_data = (audio_data / peak) * 0.95
                                
                        # Prepare prompt
                        prompt = self.initial_prompt if use_prompt and self.initial_prompt else None
                        
                        # **SYNCHRONOUS TRANSCRIPTION** - The key to success!
                        segments, info = self.model.transcribe(
                            audio_data,
                            language=language if language else None,
                            beam_size=self.beam_size,
                            initial_prompt=prompt,
                            vad_filter=self.vad_filter,
                            temperature=0.0
                        )
                        
                        # Extract transcription text
                        transcription = " ".join(segment.text for segment in segments).strip()
                        
                        elapsed = time.time() - start_time
                        logger.debug(f"🎤 WORKER: Transcription completed: '{transcription}' in {elapsed:.3f}s")
                        
                        # Send successful result
                        self.conn.send(('success', (transcription, info)))
                        
                    except Exception as e:
                        logger.error(f"🎤 WORKER: Transcription error: {e}")
                        self.conn.send(('error', str(e)))
                        
                except queue.Empty:
                    continue
                except KeyboardInterrupt:
                    logger.info("🎤 WORKER: Received keyboard interrupt")
                    break
                except Exception as e:
                    logger.error(f"🎤 WORKER: General error: {e}")
                    
        finally:
            # Cleanup
            logger.info("🎤 WORKER: Shutting down transcription worker")
            if self.conn:
                try:
                    self.conn.close()
                except:
                    pass
            self.shutdown_event.set()
            

def start_transcription_worker(
    conn,
    model_path: str = "/mnt/c/Users/PC/Dev/fastRTC/models/models--Systran--faster-whisper-large-v3/snapshots/edaa852ec7e145841d8ffdb056a99866b5f0a478", 
    device: str = "cuda",
    compute_type: str = "int8_float16",
    beam_size: int = 1,
    vad_filter: bool = False,
    language: Optional[str] = None,
    initial_prompt: Optional[str] = None,
    normalize_audio: bool = True,
    shutdown_event=None
):
    """
    Entry point for starting transcription worker process.
    
    This function is called by multiprocessing.Process to start the worker.
    """
    worker = TranscriptionWorker(
        conn=conn,
        model_path=model_path,
        device=device,
        compute_type=compute_type,
        beam_size=beam_size,
        vad_filter=vad_filter,
        language=language,
        initial_prompt=initial_prompt,
        normalize_audio=normalize_audio,
        shutdown_event=shutdown_event
    )
    worker.run()