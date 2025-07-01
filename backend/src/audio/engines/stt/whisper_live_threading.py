"""
WhisperLive STT Threading Wrapper

Pure threading implementation that wraps WhisperLive async client
to avoid event loop conflicts in threading pipelines.
"""

import threading
import asyncio
import queue
import time
import json
import numpy as np
import websockets
from typing import Optional, Dict, Any, Union
from pathlib import Path

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from src.audio.engines.stt.base import BaseSTTEngine
from src.core.interfaces import AudioData, TranscriptionResult
from src.core.exceptions import STTError
from src.utils.logging import get_logger

# Remove dependency on whisper-live client - use direct WebSocket
WHISPER_LIVE_AVAILABLE = True

logger = get_logger(__name__)


class WhisperLiveThreadingSTT(BaseSTTEngine):
    """
    Threading-compatible WhisperLive STT engine using queue-based architecture.
    
    Uses direct WebSocket connection with producer-consumer pattern for real-time processing.
    """

    def __init__(
        self,
        server_host: str = "localhost",
        server_port: int = 9090,
        model: str = "small",
        language: str = "en",
        translate: bool = False,
        use_vad: bool = True,  # Use WhisperLive VAD instead of FastRTC VAD
    ):
        super().__init__()

        self.server_host = server_host
        self.server_port = server_port
        self.model = model
        self.language = language
        self.translate = translate
        self.use_vad = use_vad
        
        # WebSocket URL for direct connection (WhisperLive uses base URL, no path)
        self.ws_url = f"ws://{server_host}:{server_port}"

        # Queue-based communication (clean producer-consumer pattern)
        self.audio_chunk_queue = queue.Queue(maxsize=128)
        self.transcript_queue = queue.Queue(maxsize=32)

        # Threading components
        self.websocket_worker: Optional[threading.Thread] = None
        self.running = threading.Event()
        self.initialized = False
        
        # Current transcription state for polling
        self.current_transcription = {"text": "", "complete": False}
        self.transcription_lock = threading.Lock()

        logger.info(
            f"🎤 WhisperLive Threading STT initialized (server: {server_host}:{server_port})"
        )

    def initialize(self) -> bool:
        """Initialize the threading STT engine (synchronous)."""
        try:
            # Start WebSocket worker thread with queue-based architecture
            self.running.set()
            self.websocket_worker = threading.Thread(
                target=self._websocket_worker_main, daemon=True
            )
            self.websocket_worker.start()

            # Wait for WebSocket connection to establish
            start_time = time.time()
            while not self.initialized and time.time() - start_time < 10.0:
                time.sleep(0.1)

            if not self.initialized:
                raise STTError("WhisperLive WebSocket connection timeout")

            self._is_available = True
            logger.info("✅ WhisperLive Threading STT initialized successfully")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to initialize WhisperLive Threading STT: {e}")
            self._is_available = False
            return False

    def _websocket_worker_main(self):
        """Main WebSocket worker thread with clean queue-based architecture."""
        try:
            # Run the async WebSocket loop in this thread
            asyncio.run(self._websocket_loop())
        except Exception as e:
            logger.error(f"WebSocket worker thread error: {e}")

    async def _websocket_loop(self):
        """Clean WebSocket loop with producer-consumer pattern."""
        try:
            print(f"🎤 WHISPER-LIVE: Connecting to {self.ws_url}")
            
            # Configure WebSocket with aggressive keepalive to prevent timeouts during audio streaming
            async with websockets.connect(
                self.ws_url,
                ping_interval=5,   # Send ping every 5 seconds (more frequent)
                ping_timeout=3,    # Wait 3 seconds for pong (faster response)
                close_timeout=5,   # Wait 5 seconds for close
                max_size=None      # Remove size limit for large audio streams
            ) as websocket:
                print(f"🎤 WHISPER-LIVE: Connected successfully")
                
                # Send initial configuration message (as per WhisperLive protocol)
                config_message = {
                    "uid": "threading_client",
                    "language": self.language,
                    "task": "translate" if self.translate else "transcribe",
                    "model": self.model,
                    "use_vad": True,   # Enable WhisperLive VAD - FastRTC VAD disabled
                    "save_output_recording": False,
                    "enable_llm": False
                }
                config_json = json.dumps(config_message)
                await websocket.send(config_json)
                print(f"🎤 WHISPER-LIVE: Sent config: {config_json}")
                
                # Wait a moment for server to process config
                await asyncio.sleep(0.1)
                
                self.initialized = True
                
                # Start receiver task
                receive_task = asyncio.create_task(self._receive_transcriptions(websocket))
                
                # Main send loop
                chunk_count = 0
                while self.running.is_set():
                    try:
                        # Get audio chunk from queue (slightly longer timeout to handle bursts)
                        audio_chunk = self.audio_chunk_queue.get(timeout=0.1)
                        
                        # Handle END_OF_AUDIO signal (special case)
                        if isinstance(audio_chunk, bytes) and audio_chunk == b"END_OF_AUDIO":
                            print(f"🎤 WHISPER-LIVE: Sending END_OF_AUDIO signal to server")
                            await websocket.send(audio_chunk)
                            print(f"🎤 WHISPER-LIVE: END_OF_AUDIO signal sent, waiting for final transcription...")
                            continue
                        
                        # Convert numpy array to bytes if needed (normal audio data)
                        if isinstance(audio_chunk, np.ndarray):
                            # Ensure audio is float32 as expected by WhisperLive
                            if audio_chunk.dtype != np.float32:
                                audio_chunk = audio_chunk.astype(np.float32)
                            audio_bytes = audio_chunk.tobytes()
                        else:
                            audio_bytes = audio_chunk
                            
                        # Send to WhisperLive server
                        await websocket.send(audio_bytes)
                        chunk_count += 1
                        print(f"🎤 WHISPER-LIVE: Sent audio chunk {chunk_count} ({len(audio_bytes)} bytes, {len(audio_bytes)//4} float32 samples)")
                        
                        # Every 5 chunks, check if websocket is still alive (more frequent monitoring)
                        if chunk_count % 5 == 0:
                            print(f"🎤 WHISPER-LIVE: WebSocket health check - closed={websocket.closed}")
                            if websocket.closed:
                                print(f"🎤 WHISPER-LIVE: ERROR - WebSocket closed unexpectedly during transmission!")
                                break
                        
                    except queue.Empty:
                        continue
                    except websockets.exceptions.ConnectionClosed as e:
                        logger.error(f"WebSocket connection closed: {e}")
                        print(f"🎤 WHISPER-LIVE: Connection closed during audio transmission - code: {e.code}, reason: {e.reason}")
                        break
                    except Exception as e:
                        logger.error(f"Error sending audio: {e}")
                        print(f"🎤 WHISPER-LIVE: WebSocket error - state: closed={websocket.closed}, error: {e}")
                        # Try to continue for transient errors
                        await asyncio.sleep(0.1)
                
                # Clean shutdown
                receive_task.cancel()
                
        except Exception as e:
            logger.error(f"WebSocket connection error: {e}")
            # Don't set initialized=False here to allow reconnection attempts
            print(f"🎤 WHISPER-LIVE: Connection error, but keeping engine available for retry")

    async def _receive_transcriptions(self, websocket):
        """Receive transcription results from WhisperLive server."""
        print(f"🎤 WHISPER-LIVE: Receiver task started, waiting for messages...")
        try:
            async for message in websocket:
                print(f"🎤 WHISPER-LIVE: Raw message received: {type(message)} - {repr(message)}")
                
                try:
                    # Try to parse as JSON first
                    if isinstance(message, str):
                        data = json.loads(message)
                        print(f"🎤 WHISPER-LIVE: Parsed JSON: {data}")
                        
                        # Handle different message types
                        if isinstance(data, dict):
                            message_type = data.get("message", "")
                            
                            # Handle SERVER_READY message
                            if message_type == "SERVER_READY":
                                print(f"🎤 WHISPER-LIVE: Server ready with backend: {data.get('backend', 'unknown')}")
                                continue
                            
                            # Extract transcription text from various possible JSON structures
                            text = None
                            text = data.get("text", data.get("transcript", data.get("transcription", "")))
                            segments = data.get("segments", [])
                            
                            # Also try to extract from segments
                            if segments and isinstance(segments, list):
                                segment_texts = [seg.get("text", "") for seg in segments if isinstance(seg, dict)]
                                if segment_texts:
                                    segment_text = " ".join(segment_texts).strip()
                                    if segment_text:
                                        text = segment_text
                            
                            if text and text.strip():
                                print(f"🎤 WHISPER-LIVE: Extracted text: '{text}'")
                                
                                # Update current transcription state
                                with self.transcription_lock:
                                    self.current_transcription["text"] = text
                                    self.current_transcription["complete"] = True
                                
                                # Put in transcript queue for polling
                                try:
                                    self.transcript_queue.put_nowait(text)
                                    print(f"🎤 WHISPER-LIVE: Added to transcript queue: '{text}'")
                                except queue.Full:
                                    logger.warning("Transcript queue full, dropping result")
                            else:
                                print(f"🎤 WHISPER-LIVE: No transcription text in message: {data}")
                    
                    elif isinstance(message, bytes):
                        # Try to decode bytes message
                        try:
                            text_message = message.decode('utf-8')
                            print(f"🎤 WHISPER-LIVE: Decoded bytes message: '{text_message}'")
                            # Try parsing decoded message as JSON
                            data = json.loads(text_message)
                            print(f"🎤 WHISPER-LIVE: Parsed decoded JSON: {data}")
                        except (UnicodeDecodeError, json.JSONDecodeError) as e:
                            print(f"🎤 WHISPER-LIVE: Could not decode/parse bytes message: {e}")
                    
                    else:
                        print(f"🎤 WHISPER-LIVE: Unknown message type: {type(message)}")
                            
                except json.JSONDecodeError as e:
                    print(f"🎤 WHISPER-LIVE: JSON decode error: {e} - Raw message: {repr(message)}")
                except Exception as e:
                    print(f"🎤 WHISPER-LIVE: Error processing message: {e} - Raw message: {repr(message)}")
                    logger.error(f"Error processing transcription: {e}")
                    
        except websockets.exceptions.ConnectionClosed as e:
            print(f"🎤 WHISPER-LIVE: WebSocket connection closed in receiver - code: {e.code}, reason: {e.reason}")
            logger.error(f"WebSocket connection closed in receiver: {e}")
        except Exception as e:
            print(f"🎤 WHISPER-LIVE: Error in receive loop: {e}")
            logger.error(f"Error receiving transcriptions: {e}")

    # Clean polling interface methods for VoicePipelineThreaded integration
    def push_audio_chunk(self, audio_chunk: np.ndarray):
        """Push audio chunk for streaming transcription (queue-based)."""
        if not self.initialized:
            logger.warning("WhisperLive not initialized, cannot push audio")
            return
        
        try:
            # Put audio chunk in queue for WebSocket worker to send
            self.audio_chunk_queue.put_nowait(audio_chunk)
            print(f"🎤 WHISPER-LIVE: Queued audio chunk ({audio_chunk.shape})")
        except queue.Full:
            logger.warning("WhisperLive audio queue full, dropping chunk")
    
    def get_transcript(self) -> str:
        """Get current transcription (for polling by VoicePipelineThreaded)."""
        try:
            # Try to get latest transcript from queue
            latest_transcript = self.transcript_queue.get_nowait()
            return latest_transcript
        except queue.Empty:
            # Return current state if no new transcripts
            with self.transcription_lock:
                return self.current_transcription["text"]
    
    def is_transcription_complete(self) -> bool:
        """Check if transcription is complete (for polling architecture)."""
        with self.transcription_lock:
            return self.current_transcription["complete"]
    
    def reset_transcription_state(self):
        """Reset transcription state for next utterance."""
        with self.transcription_lock:
            self.current_transcription["text"] = ""
            self.current_transcription["complete"] = False

    async def _transcribe_audio(self, audio: Union[AudioData, np.ndarray]) -> TranscriptionResult:
        """Transcribe audio using queue-based streaming approach."""
        # Prepare audio data
        if isinstance(audio, AudioData):
            audio_data = audio.data
        else:
            audio_data = audio

        if not isinstance(audio_data, np.ndarray):
            audio_data = np.array(audio_data)

        return await self._process_audio_streaming(audio_data)

    async def _process_audio_streaming(self, audio_data: np.ndarray) -> TranscriptionResult:
        """Process audio using direct WebSocket connection per request."""
        try:
            print(f"🎤 WHISPER-LIVE: Processing audio via direct WebSocket - shape: {audio_data.shape}")
            
            # WhisperLive requires specific audio format: float32, 16kHz, mono
            print(f"🎤 WHISPER-LIVE: Original audio - shape: {audio_data.shape}, dtype: {audio_data.dtype}")
            
            # Ensure mono audio first
            if audio_data.ndim > 1:
                audio_data = audio_data.flatten() if audio_data.shape[0] == 1 else audio_data[0]
                print(f"🎤 WHISPER-LIVE: Converted to mono - shape: {audio_data.shape}")
            
            # Convert to float32 in range [-1, 1] (WhisperLive requirement)
            if audio_data.dtype == np.int16:
                audio_data = audio_data.astype(np.float32) / 32768.0
                print(f"🎤 WHISPER-LIVE: Converted int16 to float32")
            elif audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)
                print(f"🎤 WHISPER-LIVE: Converted {audio_data.dtype} to float32")
            
            # Validate audio format matches WhisperLive expectations
            sample_rate = 16000  # WhisperLive expects 16kHz
            duration_seconds = len(audio_data) / sample_rate
            print(f"🎤 WHISPER-LIVE: Final audio format - shape: {audio_data.shape}, dtype: {audio_data.dtype}")
            print(f"🎤 WHISPER-LIVE: Sample rate: {sample_rate}Hz, duration: {duration_seconds:.2f}s")
            print(f"🎤 WHISPER-LIVE: Audio range: [{np.min(audio_data):.3f}, {np.max(audio_data):.3f}]")
            
            # Create dedicated WebSocket connection for this transcription
            transcription_result = await self._transcribe_with_websocket(audio_data)
            return transcription_result

        except Exception as e:
            print(f"🎤 WHISPER-LIVE: Error in WebSocket transcription: {e}")
            logger.error(f"WhisperLive WebSocket error: {e}")
            return TranscriptionResult(
                text="",
                confidence=0.0,
                language=self.language
            )
    
    async def _transcribe_with_websocket(self, audio_data: np.ndarray) -> TranscriptionResult:
        """Create dedicated WebSocket connection for single transcription."""
        transcription_text = ""
        
        try:
            print(f"🎤 WHISPER-LIVE: Creating dedicated WebSocket connection for transcription")
            
            # Connect with optimized settings for single transcription
            async with websockets.connect(
                self.ws_url,
                ping_interval=None,  # Disable ping for short connections
                ping_timeout=None,
                close_timeout=5
            ) as websocket:
                print(f"🎤 WHISPER-LIVE: Connected to {self.ws_url}")
                
                # Send configuration (enable VAD since FastRTC VAD is disabled)
                config_message = {
                    "uid": f"direct_client_{int(time.time())}",
                    "language": self.language,
                    "task": "translate" if self.translate else "transcribe",
                    "model": self.model,
                    "use_vad": True,   # Enable - WhisperLive handles VAD, FastRTC VAD disabled
                    "save_output_recording": True,  # Keep for debugging
                    "enable_llm": False
                }
                config_json = json.dumps(config_message)
                await websocket.send(config_json)
                print(f"🎤 WHISPER-LIVE: Sent config: {config_json}")
                
                # Wait for SERVER_READY
                try:
                    ready_message = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                    print(f"🎤 WHISPER-LIVE: Server response: {ready_message}")
                except asyncio.TimeoutError:
                    print(f"🎤 WHISPER-LIVE: No SERVER_READY received, proceeding anyway")
                
                # Send audio data in smaller, more frequent chunks with delays
                chunk_size = 1024  # Smaller chunks (was 4096)
                total_chunks = len(audio_data) // chunk_size + (1 if len(audio_data) % chunk_size != 0 else 0)
                print(f"🎤 WHISPER-LIVE: Sending {total_chunks} audio chunks (smaller chunks with timing)")
                
                for i in range(0, len(audio_data), chunk_size):
                    chunk = audio_data[i:i + chunk_size]
                    if chunk.dtype != np.float32:
                        chunk = chunk.astype(np.float32)
                    audio_bytes = chunk.tobytes()
                    await websocket.send(audio_bytes)
                    
                    chunk_num = i//chunk_size + 1
                    if chunk_num % 50 == 0:  # Log every 50 chunks instead of every chunk
                        print(f"🎤 WHISPER-LIVE: Sent chunk {chunk_num}/{total_chunks} ({len(audio_bytes)} bytes)")
                    
                    # Small delay to simulate real-time streaming
                    await asyncio.sleep(0.01)
                
                # Send END_OF_AUDIO
                print(f"🎤 WHISPER-LIVE: Sending END_OF_AUDIO signal")
                await websocket.send(b"END_OF_AUDIO")
                
                # Give server time to process before waiting for response
                print(f"🎤 WHISPER-LIVE: Allowing processing time...")
                await asyncio.sleep(2.0)  # Give server time to start processing
                
                # Wait for transcription result with longer timeout
                print(f"🎤 WHISPER-LIVE: Waiting for transcription...")
                timeout = 30.0  # Much longer timeout for 27 seconds of audio
                start_time = time.time()
                
                while (time.time() - start_time) < timeout:
                    try:
                        # Longer timeout per message to allow processing
                        message = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                        print(f"🎤 WHISPER-LIVE: Raw received: {type(message)} - {repr(message)}")
                        
                        # Handle both string and binary messages
                        if isinstance(message, str):
                            try:
                                data = json.loads(message)
                                print(f"🎤 WHISPER-LIVE: Parsed JSON: {data}")
                                
                                # Skip SERVER_READY messages
                                if data.get("message") == "SERVER_READY":
                                    continue
                                
                                # Look for any text content in the message
                                text = data.get("text", "")
                                segments = data.get("segments", [])
                                transcript = data.get("transcript", "")
                                transcription = data.get("transcription", "")
                                
                                # Try multiple ways to extract text
                                final_text = text or transcript or transcription
                                
                                if segments and isinstance(segments, list):
                                    segment_texts = [seg.get("text", "") for seg in segments if isinstance(seg, dict)]
                                    if segment_texts:
                                        final_text = " ".join(segment_texts).strip()
                                
                                if final_text and final_text.strip():
                                    transcription_text = final_text.strip()
                                    print(f"🎤 WHISPER-LIVE: Got transcription: '{transcription_text}'")
                                    break
                                else:
                                    print(f"🎤 WHISPER-LIVE: Message contains no transcription text: {data}")
                            except json.JSONDecodeError as e:
                                print(f"🎤 WHISPER-LIVE: JSON decode error: {e}, treating as raw text: {message}")
                                # Maybe it's just plain text
                                if message.strip():
                                    transcription_text = message.strip()
                                    print(f"🎤 WHISPER-LIVE: Got raw text transcription: '{transcription_text}'")
                                    break
                        
                        elif isinstance(message, bytes):
                            print(f"🎤 WHISPER-LIVE: Received binary message: {len(message)} bytes")
                            try:
                                text_message = message.decode('utf-8')
                                print(f"🎤 WHISPER-LIVE: Decoded binary as: {text_message}")
                                if text_message.strip():
                                    transcription_text = text_message.strip()
                                    print(f"🎤 WHISPER-LIVE: Got binary transcription: '{transcription_text}'")
                                    break
                            except UnicodeDecodeError:
                                print(f"🎤 WHISPER-LIVE: Could not decode binary message")
                                
                    except asyncio.TimeoutError:
                        elapsed = time.time() - start_time
                        print(f"🎤 WHISPER-LIVE: No message in 5s (total elapsed: {elapsed:.1f}s), continuing...")
                        continue
                    except websockets.exceptions.ConnectionClosed as e:
                        print(f"🎤 WHISPER-LIVE: Connection closed during wait - code: {e.code}, reason: {e.reason}")
                        break
                        
        except Exception as e:
            print(f"🎤 WHISPER-LIVE: WebSocket transcription error: {e}")
            logger.error(f"Direct WebSocket transcription error: {e}")
        
        # Return result
        if transcription_text:
            return TranscriptionResult(
                text=transcription_text,
                confidence=0.9,
                language=self.language
            )
        else:
            print(f"🎤 WHISPER-LIVE: No transcription received")
            return TranscriptionResult(
                text="",
                confidence=0.0,
                language=self.language
            )

    def transcribe_sync(self, audio: Union[AudioData, np.ndarray]) -> TranscriptionResult:
        """Synchronous transcribe interface (for compatibility)."""
        if not self.initialized:
            raise STTError("WhisperLive Threading STT not initialized")

        # Run the async method in a new event loop (for sync compatibility) 
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(self._transcribe_audio(audio))
            return result
        finally:
            loop.close()
    
    def is_transcription_complete(self) -> bool:
        """Check if transcription is complete (for polling architecture)."""
        with self.transcription_lock:
            return self.current_transcription["complete"]

    def cleanup(self):
        """Clean up resources (synchronous)."""
        try:
            self.running.clear()

            if self.websocket_worker and self.websocket_worker.is_alive():
                self.websocket_worker.join(timeout=5.0)

            self._is_available = False
            logger.info("🧹 WhisperLive Threading STT cleaned up")

        except Exception as e:
            logger.error(f"Cleanup error: {e}")

    def is_available(self) -> bool:
        """Check if engine is available."""
        return self._is_available and self.initialized

    def get_stats(self) -> Dict[str, Any]:
        """Get engine statistics."""
        stats = super().get_stats()
        stats.update(
            {
                "engine_type": "whisper-live-threading",
                "server_host": self.server_host,
                "server_port": self.server_port,
                "model": self.model,
                "initialized": self.initialized,
                "running": self.running,
            }
        )
        return stats
