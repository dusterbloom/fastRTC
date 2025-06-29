"""
Multiprocessing-based Transcription Worker

Based on RealtimeSTT's TranscriptionWorker implementation.
Runs faster-whisper in a separate process to avoid threading issues.
"""

import os
import sys
import time
import queue
import signal
import logging
import threading
import multiprocessing as mp
from typing import Optional, Union, List, Tuple, Any

import numpy as np
from faster_whisper import WhisperModel, BatchedInferencePipeline

from ....core.interfaces import TranscriptionResult
from ....utils.logging import get_logger

logger = get_logger(__name__)


class TranscriptionWorker:
    """
    Multiprocessing-based transcription worker that runs faster-whisper
    in a separate process to avoid threading conflicts.
    
    Based on RealtimeSTT's TranscriptionWorker implementation.
    """
    
    def __init__(
        self,
        conn,
        stdout_pipe,
        model_path: str,
        device: str = "cuda",
        compute_type: str = "int8_float16",
        beam_size: int = 1,
        batch_size: int = 0,
        language: Optional[str] = None,
        temperature: float = 0.0,
        vad_filter: bool = True
    ):
        """
        Initialize transcription worker.
        
        Args:
            conn: Connection pipe for receiving transcription requests
            stdout_pipe: Pipe for sending stdout messages back to main process
            model_path: Path to whisper model
            device: Device to use ("cuda" or "cpu")
            compute_type: Compute type for CTranslate2
            beam_size: Beam size for transcription
            batch_size: Batch size (0 to disable batching)
            language: Target language code
            temperature: Temperature for transcription
            vad_filter: Whether to use VAD filter
        """
        self.conn = conn
        self.stdout_pipe = stdout_pipe
        self.model_path = model_path
        self.device = device
        self.compute_type = compute_type
        self.beam_size = beam_size
        self.batch_size = batch_size
        self.language = language
        self.temperature = temperature
        self.vad_filter = vad_filter
        
        self.queue = queue.Queue()
        self.model = None
        self.shutdown_event = mp.Event()
        
    def custom_print(self, *args, **kwargs):
        """Send print messages back to main process."""
        message = ' '.join(map(str, args))
        try:
            self.stdout_pipe.send(message)
        except (BrokenPipeError, EOFError, OSError):
            pass
    
    def poll_connection(self):
        """Poll connection for incoming transcription requests."""
        while not self.shutdown_event.is_set():
            try:
                if self.conn.poll(0.01):
                    data = self.conn.recv()
                    self.queue.put(data)
                else:
                    time.sleep(0.02)
            except Exception as e:
                logging.error(f"Error receiving data from connection: {e}")
                time.sleep(0.02)
    
    def run(self):
        """Main worker process loop."""
        # Ignore SIGINT in worker process
        if hasattr(signal, 'SIGINT'):
            signal.signal(signal.SIGINT, signal.SIG_IGN)
        
        # Redirect print to custom handler
        if __name__ == "__main__":
            __builtins__['print'] = self.custom_print
        
        logging.info(f"Initializing faster_whisper transcription model {self.model_path}")
        
        try:
            # Initialize model in worker process
            logging.info(f"Loading model on device: {self.device} with compute_type: {self.compute_type}")
            logging.info(f"Model path: {self.model_path}")
            
            # Add more detailed GPU info in worker process
            if self.device == "cuda":
                try:
                    import torch
                    if torch.cuda.is_available():
                        logging.info(f"Worker process GPU info: {torch.cuda.get_device_name()}")
                        logging.info(f"Worker process GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
                    else:
                        logging.warning("CUDA not available in worker process, falling back to CPU")
                        self.device = "cpu"
                        self.compute_type = "int8"
                except Exception as e:
                    logging.warning(f"GPU check failed in worker: {e}, using CPU")
                    self.device = "cpu"
                    self.compute_type = "int8"
            
            self.model = WhisperModel(
                model_size_or_path=self.model_path,
                device=self.device,
                compute_type=self.compute_type
            )
            
            # Create batched pipeline if batch_size > 0
            if self.batch_size > 0:
                self.model = BatchedInferencePipeline(model=self.model)
            
            # Warm up model with dummy audio
            dummy_audio = np.zeros(16000, dtype=np.float32)  # 1 second of silence
            segments, info = self.model.transcribe(
                dummy_audio,
                language="en",
                beam_size=1,
                vad_filter=False
            )
            _ = " ".join(segment.text for segment in segments)
            
            logging.info("Faster_whisper transcription model initialized successfully")
            
        except Exception as e:
            logging.error(f"Error initializing transcription model: {e}")
            self.conn.send(('error', f"Model initialization failed: {e}"))
            return
        
        # Start polling thread
        polling_thread = threading.Thread(target=self.poll_connection)
        polling_thread.daemon = True
        polling_thread.start()
        
        # Send ready signal
        self.conn.send(('ready', None))
        
        try:
            # Main processing loop
            while not self.shutdown_event.is_set():
                try:
                    # Get transcription request
                    audio_data, sample_rate, language = self.queue.get(timeout=0.1)
                    
                    try:
                        logging.info(f"🎤 Processing audio: shape={audio_data.shape}, sr={sample_rate}, lang={language}")
                        start_time = time.time()
                        
                        # Detailed audio analysis
                        logging.info(f"🔍 Audio analysis: min={audio_data.min():.6f}, max={audio_data.max():.6f}, "
                                   f"mean={audio_data.mean():.6f}, rms={np.sqrt(np.mean(audio_data**2)):.6f}")
                        
                        # Ensure audio is float32
                        if audio_data.dtype != np.float32:
                            audio_data = audio_data.astype(np.float32)
                            logging.debug("🔧 Converted audio to float32")
                        
                        # Normalize audio if needed
                        max_val = np.max(np.abs(audio_data))
                        if max_val > 1.0:
                            audio_data = audio_data / max_val
                            logging.debug(f"🔧 Normalized audio by factor {max_val}")
                        
                        # Check for silent audio
                        rms = np.sqrt(np.mean(audio_data**2))
                        if rms < 1e-6:
                            logging.warning(f"⚠️ Audio appears to be silent (RMS: {rms:.8f})")
                        
                        # Transcribe audio
                        transcribe_kwargs = {
                            'language': language or self.language,
                            'beam_size': self.beam_size,
                            'temperature': self.temperature,
                            'vad_filter': self.vad_filter
                        }
                        
                        if self.batch_size > 0:
                            transcribe_kwargs['batch_size'] = self.batch_size
                        
                        logging.info(f"🚀 Starting transcription with kwargs: {transcribe_kwargs}")
                        segments, info = self.model.transcribe(audio_data, **transcribe_kwargs)
                        
                        # Collect transcription text
                        text_parts = []
                        segment_count = 0
                        for segment in segments:
                            segment_count += 1
                            segment_text = segment.text.strip()
                            text_parts.append(segment_text)
                            logging.debug(f"📝 Segment {segment_count}: '{segment_text}'")
                        
                        full_text = " ".join(text_parts).strip()
                        
                        # Get language info
                        detected_language = getattr(info, 'language', language or 'en')
                        confidence = getattr(info, 'language_probability', 1.0)
                        
                        elapsed = time.time() - start_time
                        logging.info(f"✅ Transcription completed: '{full_text}' (segments: {segment_count}, "
                                   f"lang: {detected_language}, conf: {confidence:.3f}) in {elapsed:.3f}s")
                        
                        # Send result back
                        result = {
                            'text': full_text,
                            'language': detected_language,
                            'confidence': confidence
                        }
                        self.conn.send(('success', result))
                        
                    except Exception as e:
                        logging.error(f"Transcription error: {e}")
                        self.conn.send(('error', str(e)))
                        
                except queue.Empty:
                    continue
                except KeyboardInterrupt:
                    logging.debug("Worker process interrupted")
                    break
                except Exception as e:
                    logging.error(f"Error in worker process: {e}")
                    
        finally:
            # Cleanup
            if __name__ == "__main__":
                __builtins__['print'] = print
            self.conn.close()
            self.stdout_pipe.close()
            self.shutdown_event.set()
            polling_thread.join(timeout=1.0)


def transcription_worker_main(
    conn,
    stdout_pipe,
    model_path: str,
    device: str = "cuda",
    compute_type: str = "int8_float16",
    beam_size: int = 1,
    batch_size: int = 0,
    language: Optional[str] = None,
    temperature: float = 0.0,
    vad_filter: bool = True
):
    """
    Main function for transcription worker process.
    
    This function is called when starting the worker process.
    """
    worker = TranscriptionWorker(
        conn=conn,
        stdout_pipe=stdout_pipe,
        model_path=model_path,
        device=device,
        compute_type=compute_type,
        beam_size=beam_size,
        batch_size=batch_size,
        language=language,
        temperature=temperature,
        vad_filter=vad_filter
    )
    worker.run()