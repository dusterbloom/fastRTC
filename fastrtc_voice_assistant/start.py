#!/usr/bin/env python3
"""
Colorful Gradio3 Voice Assistant
Enhanced with colorama for better terminal visualization
Focuses on essential information: user/assistant/memory/language
"""

import sys
import time
import os
import numpy as np
import asyncio
import threading
import websockets # Added
import json # Added
from pathlib import Path
from fastrtc import (
    ReplyOnPause,
    Stream,
    get_tts_model,
    AlgoOptions,
    SileroVadOptions,
    KokoroTTSOptions,
    audio_to_bytes,
)
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple, Any, AsyncGenerator
import logging
from collections import deque

# Colorama for colorful terminal output
from colorama import init, Fore, Back, Style
init(autoreset=True)  # Auto-reset colors after each print

# Add src to path for imports
# sys.path.insert(0, 'src') # No longer needed for VoiceAssistant direct import

# from src.core.voice_assistant import VoiceAssistant # Removed
# from src.core.interfaces import AudioData, TranscriptionResult # Removed
from src.utils.logging import setup_logging, get_logger
# from src.config.settings import DEFAULT_LANGUAGE # Removed, backend handles language
import traceback
from fastrtc.utils import AdditionalOutputs
from src.integration.fastrtc_bridge import FastRTCBridge

from src.config.audio_config import (
    # AUDIO_SAMPLE_RATE, # May come from backend or be fixed for TTS
    MINIMAL_SILENT_FRAME_DURATION_MS, # May not be relevant client side
    MINIMAL_SILENT_SAMPLES, # May not be relevant client side
    SILENT_AUDIO_CHUNK_ARRAY,
    SILENT_AUDIO_FRAME_TUPLE,
)
EMPTY_AUDIO_YIELD_OUTPUT = (SILENT_AUDIO_FRAME_TUPLE, AdditionalOutputs())
# from src.config.settings import load_config # Removed

# Set up logging
setup_logging()
logger = get_logger(__name__)

# Global instances (Removed)
# voice_assistant: Optional[VoiceAssistant] = None
# main_event_loop: Optional[asyncio.AbstractEventLoop] = None
# async_worker_thread: Optional[threading.Thread] = None

def print_colorful(message, color=Fore.WHITE, style=Style.NORMAL):
    """Print colorful messages with timestamp."""
    timestamp = time.strftime("%H:%M:%S")
    print(f"{Fore.CYAN}[{timestamp}]{Style.RESET_ALL} {style}{color}{message}{Style.RESET_ALL}")

def print_user(message):
    """Print user messages in blue."""
    print_colorful(f"👤 User: {message}", Fore.BLUE, Style.BRIGHT)

def print_assistant(message):
    """Print assistant messages in green."""
    print_colorful(f"🤖 Assistant: {message}", Fore.GREEN, Style.BRIGHT)

def print_memory(message):
    """Print memory-related messages in magenta."""
    print_colorful(f"🧠 Memory: {message}", Fore.MAGENTA, Style.BRIGHT)

def print_language(message):
    """Print language-related messages in yellow."""
    print_colorful(f"🌍 Language: {message}", Fore.YELLOW, Style.BRIGHT)

def print_tts(message):
    """Print TTS-related messages in cyan."""
    print_colorful(f"🎤 TTS: {message}", Fore.CYAN, Style.BRIGHT)

def print_error(message):
    """Print error messages in red."""
    print_colorful(f"❌ Error: {message}", Fore.RED, Style.BRIGHT)

def print_success(message):
    """Print success messages in green."""
    print_colorful(f"✅ Success: {message}", Fore.GREEN, Style.BRIGHT)

def print_info(message):
    """Print info messages in white."""
    print_colorful(f"ℹ️  Info: {message}", Fore.WHITE, Style.NORMAL)

def print_timing(message):
    """Print timing messages in light blue."""
    print_colorful(f"⏱️  Timing: {message}", Fore.LIGHTBLUE_EX, Style.NORMAL)

def voice_assistant_callback_rt(audio_data_tuple: tuple) -> AsyncGenerator[Tuple[Tuple[int, np.ndarray], AdditionalOutputs], None]:
    """Callback to handle audio data, connect to WebSocket backend, and stream TTS."""
    
    backend_ws_url = os.environ.get("BACKEND_WS_URL", "ws://localhost:8000/ws/v1/voice_chat")
    # It's better to require API key to be set, or handle its absence more explicitly
    backend_api_key = os.environ.get("BACKEND_API_KEY")

    if not backend_ws_url:
        print_error("BACKEND_WS_URL environment variable not set.")
        yield EMPTY_AUDIO_YIELD_OUTPUT
        return
    if not backend_api_key:
        print_error("BACKEND_API_KEY environment variable not set.")
        # Potentially allow operation without API key if backend supports it,
        # but for now, let's assume it's required.
        yield EMPTY_AUDIO_YIELD_OUTPUT
        return

    # Process incoming audio_data_tuple
    if isinstance(audio_data_tuple, tuple) and len(audio_data_tuple) == 2:
        sample_rate, raw_audio_array = audio_data_tuple
        if isinstance(raw_audio_array, np.ndarray) and len(raw_audio_array.shape) > 1:
            if raw_audio_array.shape[0] == 1: audio_array = raw_audio_array[0]
            elif raw_audio_array.shape[1] == 1: audio_array = raw_audio_array[:, 0]
            elif raw_audio_array.shape[1] > 1: audio_array = np.mean(raw_audio_array, axis=1) # Mono conversion
            else: audio_array = raw_audio_array.flatten()
        else:
            audio_array = raw_audio_array if isinstance(raw_audio_array, np.ndarray) else np.array(raw_audio_array, dtype=np.float32)
        
        if isinstance(audio_array, np.ndarray):
            if audio_array.dtype == np.int16: # Normalize if int16
                audio_array = audio_array.astype(np.float32) / 32768.0
            else: # Ensure float32
                audio_array = audio_array.astype(np.float32)
    else:
        audio_array = np.array([], dtype=np.float32)
        sample_rate = 16000 # Default if no audio comes, backend should confirm actual SR

    if audio_array.size == 0:
        # This case might be hit if FastRTC sends empty data even before user speaks.
        # Depending on desired behavior, we might not want to establish a WS connection yet.
        # For now, we yield silent frame and return, avoiding WS connection for empty input.
        # print_info("No audio input received, yielding silent frame.")
        yield SILENT_AUDIO_FRAME_TUPLE, AdditionalOutputs()
        return
    
    print_info(f"Received audio: {len(audio_array)/sample_rate:.2f}s, SR: {sample_rate}")

    async def _manage_websocket_communication(
        audio_data_np: np.ndarray,
        input_sample_rate: int
    ) -> List[Tuple[int, np.ndarray]]:
        collected_tts_chunks = []
        # Default TTS sample rate, will be updated by `session_started` message from backend.
        # Common rates are 16000, 22050, 24000, 44100, 48000.
        # Backend should ideally inform the client of the TTS output sample rate.
        tts_sample_rate_from_backend = 16000 # Fallback, e.g. if not in session_started

        try:
            print_info(f"Connecting to WebSocket: {backend_ws_url}")
            async with websockets.connect(backend_ws_url) as websocket:
                print_success("WebSocket connection established.")
                
                # 1. Send Authentication
                auth_payload = {
                    "type": "auth",
                    "api_key": backend_api_key,
                    "user_id": "gradio_client_user_01", # Consider making this unique or configurable
                    "session_id": None, # Let backend manage session creation
                    "audio_format": "float32",
                    "sample_rate": input_sample_rate
                }
                await websocket.send(json.dumps(auth_payload))
                print_info(f"Sent auth: user_id='{auth_payload['user_id']}', sample_rate={auth_payload['sample_rate']}")

                # 2. Receive Session Started Confirmation
                response_str = await websocket.recv()
                if isinstance(response_str, str):
                    response_data = json.loads(response_str)
                    print_info(f"Received from WS: {response_data}")
                    if response_data.get("type") == "session_started":
                        session_id = response_data.get("session_id")
                        # Update TTS sample rate if provided by backend
                        tts_sample_rate_from_backend = response_data.get("tts_sample_rate", tts_sample_rate_from_backend)
                        print_success(f"Session started: ID={session_id}, TTS SR: {tts_sample_rate_from_backend}")
                    elif response_data.get("type") == "error":
                        print_error(f"Authentication/Session error from backend: {response_data.get('message')}")
                        return [] # Stop processing for this turn
                    else:
                        print_error(f"Unexpected JSON response after auth: {response_data}")
                        return []
                else: # Should be a JSON string
                    print_error(f"Unexpected response type after auth (expected str, got {type(response_str)}): {response_str[:100]}")
                    return []

                # 3. Send Initial Audio
                if audio_data_np.size > 0:
                    # Ensure audio is float32 before sending
                    audio_bytes_to_send = audio_data_np.astype(np.float32).tobytes()
                    await websocket.send(audio_bytes_to_send)
                    print_info(f"Sent {len(audio_bytes_to_send)} bytes of audio data.")
                
                await websocket.send(json.dumps({"type": "audio_complete"}))
                print_info("Sent audio_complete.")

                # 4. Receive and Process Backend Messages
                while True:
                    message = await websocket.recv()
                    
                    if isinstance(message, str): # JSON message
                        data = json.loads(message)
                        # print_info(f"Received JSON from WS: {data}") # Can be verbose
                        msg_type = data.get("type")

                        if msg_type == "stt_final":
                            user_text = data.get("text", "")
                            print_user(f"{user_text}")
                        elif msg_type == "llm_response":
                            assistant_response_text = data.get("text", "")
                            print_assistant(f"{assistant_response_text}")
                        elif msg_type == "tts_complete":
                            print_success("TTS complete signal received from backend.")
                            break # End of this interaction's TTS
                        elif msg_type == "error":
                            print_error(f"Backend error during interaction: {data.get('message')}")
                            # Depending on severity, might break or just log
                        elif msg_type == "pong": # Handle keepalive if backend sends pings
                            # print_info("Received pong from backend.")
                            pass
                        else:
                            print_info(f"Received unhandled JSON message type: {msg_type}, data: {data}")

                    elif isinstance(message, bytes): # TTS audio chunk
                        audio_chunk_np = np.frombuffer(message, dtype=np.float32)
                        if audio_chunk_np.size > 0:
                            # print_tts(f"Received TTS audio chunk: {len(audio_chunk_np)} samples at {tts_sample_rate_from_backend} Hz")
                            collected_tts_chunks.append((tts_sample_rate_from_backend, audio_chunk_np))
                        # else:
                            # print_tts("Received empty TTS audio chunk (ignoring).")
                    else:
                        print_error(f"Received unexpected message type: {type(message)}")
            
            print_info("WebSocket connection closed by server or tts_complete.")

        except websockets.exceptions.ConnectionClosedOK:
            print_info("WebSocket connection closed gracefully (OK).")
        except websockets.exceptions.ConnectionClosedError as e:
            print_error(f"WebSocket connection closed with error: {e.code} {e.reason}")
        except ConnectionRefusedError:
            print_error(f"WebSocket connection refused. Is the backend server running at {backend_ws_url}?")
        except asyncio.TimeoutError:
            print_error("WebSocket operation timed out.")
        except Exception as e:
            print_error(f"Error in WebSocket communication: {type(e).__name__}: {e}")
            traceback.print_exc()
        
        return collected_tts_chunks

    # Running the async helper in the synchronous callback
    # This requires its own event loop as FastRTCBridge runs this callback in a separate thread
    loop = None
    try:
        # Get or create a new event loop for this thread if one doesn't exist or is closed
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
        except RuntimeError: # No current event loop
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        tts_chunks_to_yield = loop.run_until_complete(
            _manage_websocket_communication(audio_array, sample_rate)
        )
        
        if tts_chunks_to_yield:
            # print_success(f"Collected {len(tts_chunks_to_yield)} TTS chunks to yield.")
            num_yielded = 0
            for chunk_sr, chunk_data in tts_chunks_to_yield:
                if chunk_data.size > 0:
                    yield (chunk_sr, chunk_data), AdditionalOutputs()
                    num_yielded +=1
            if num_yielded == 0: # All chunks were empty
                print_info("All collected TTS chunks were empty, yielding silent frame.")
                yield SILENT_AUDIO_FRAME_TUPLE, AdditionalOutputs()

        else: # No chunks collected (e.g., error during WS, or backend sent no audio)
            print_info("No TTS chunks received or an error occurred, yielding silent frame.")
            yield SILENT_AUDIO_FRAME_TUPLE, AdditionalOutputs()
            
    except Exception as e:
        print_error(f"Critical error in voice_assistant_callback_rt: {type(e).__name__}: {e}")
        traceback.print_exc()
        yield EMPTY_AUDIO_YIELD_OUTPUT # Fallback
    # finally:
        # The loop management here can be tricky. If FastRTC calls this callback
        # repeatedly in the same thread, closing the loop might be problematic.
        # For now, let's not close it here, assuming the thread might reuse it or
        # that FastRTC manages thread lifecycle appropriately.
        # If issues arise, this might need revisiting.
        # if loop and not loop.is_closed():
        #     loop.close()
        #     print_info("Asyncio event loop for callback closed.")


# def setup_async_environment(): # Removed as per instructions
#     pass

# def run_coro_from_sync_thread_with_timeout(coro, timeout: float = 4.0) -> any: # Removed as per instructions
#     pass

def main():
    """Colorful main function."""
    print(f"{Fore.CYAN}{Style.BRIGHT}🎨 Colorful FastRTC Voice Assistant (WebSocket Mode){Style.RESET_ALL}")
    print(f"{Fore.YELLOW}{'=' * 50}{Style.RESET_ALL}")
    
    # setup_async_environment() # Call removed

    bridge = FastRTCBridge()
    # The callback's type signature is now an AsyncGenerator,
    # but FastRTCBridge expects a regular generator.
    # The way voice_assistant_callback_rt is structured (running an event loop internally
    # and yielding) should still be compatible with FastRTCBridge's expectation
    # of a synchronous generator.
    bridge.create_stream(voice_assistant_callback_rt)
    
    try:
        print(f"{Fore.GREEN}{Style.BRIGHT}🚀 Assistant (WebSocket Mode) starting... Ensure backend is running.{Style.RESET_ALL}")
        print(f"{Fore.WHITE}   Make sure BACKEND_WS_URL and BACKEND_API_KEY environment variables are set.{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}{'=' * 50}{Style.RESET_ALL}")
        
        bridge.launch_stream()
        
    except KeyboardInterrupt:
        print_info("Shutting down...")
    except Exception as e:
        print_error(f"Launch error: {type(e).__name__}: {e}")
        traceback.print_exc()
    finally:
        print(f"\n{Fore.CYAN}{Style.BRIGHT}👋 Session ended{Style.RESET_ALL}")

if __name__ == "__main__":
    main()