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
from typing import Dict, List, Optional, Tuple, Any
import logging
from collections import deque

# Colorama for colorful terminal output
from colorama import init, Fore, Back, Style
init(autoreset=True)  # Auto-reset colors after each print

# Add src to path for imports
sys.path.insert(0, 'src')

from src.core.voice_assistant import VoiceAssistant
from src.core.interfaces import AudioData, TranscriptionResult
from src.utils.logging import setup_logging, get_logger
from src.config.settings import DEFAULT_LANGUAGE
import traceback
from fastrtc.utils import AdditionalOutputs
from src.integration.fastrtc_bridge import FastRTCBridge

AUDIO_SAMPLE_RATE = 16000
MINIMAL_SILENT_FRAME_DURATION_MS = 20
MINIMAL_SILENT_SAMPLES = int(AUDIO_SAMPLE_RATE * (MINIMAL_SILENT_FRAME_DURATION_MS / 1000.0))
SILENT_AUDIO_CHUNK_ARRAY = np.zeros(MINIMAL_SILENT_SAMPLES, dtype=np.float32)
SILENT_AUDIO_FRAME_TUPLE = (AUDIO_SAMPLE_RATE, SILENT_AUDIO_CHUNK_ARRAY)
EMPTY_AUDIO_YIELD_OUTPUT = (SILENT_AUDIO_FRAME_TUPLE, AdditionalOutputs())

# Set up logging
setup_logging()
logger = get_logger(__name__)

# Global instances
voice_assistant: Optional[VoiceAssistant] = None
main_event_loop: Optional[asyncio.AbstractEventLoop] = None
async_worker_thread: Optional[threading.Thread] = None

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

def voice_assistant_callback_rt(audio_data_tuple: tuple):
    """Colorful callback with essential information only."""
    global voice_assistant

    if not voice_assistant:
        print_error("Voice assistant not initialized!")
        yield EMPTY_AUDIO_YIELD_OUTPUT
        return

    try:
        # Process audio (simplified from original)
        if isinstance(audio_data_tuple, tuple) and len(audio_data_tuple) == 2:
            sample_rate, raw_audio_array = audio_data_tuple
            
            if isinstance(raw_audio_array, np.ndarray) and len(raw_audio_array.shape) > 1:
                if raw_audio_array.shape[0] == 1:
                    audio_array = raw_audio_array[0]
                elif raw_audio_array.shape[1] == 1:
                    audio_array = raw_audio_array[:, 0]
                elif raw_audio_array.shape[1] > 1:
                    audio_array = np.mean(raw_audio_array, axis=1)
                else:
                    audio_array = raw_audio_array.flatten()
            else:
                audio_array = raw_audio_array if isinstance(raw_audio_array, np.ndarray) else np.array(raw_audio_array, dtype=np.float32)
            
            if isinstance(audio_array, np.ndarray):
                if audio_array.dtype == np.int16:
                    audio_array = audio_array.astype(np.float32) / 32768.0
                else:
                    audio_array = audio_array.astype(np.float32)
        else:
            audio_array = np.array([], dtype=np.float32)
            sample_rate = 16000

        if audio_array.size == 0:
            yield SILENT_AUDIO_FRAME_TUPLE, AdditionalOutputs()
            return

        # Language names mapping
        lang_names = {
            'a': 'American English', 'b': 'British English', 'i': 'Italian', 
            'e': 'Spanish', 'f': 'French', 'p': 'Portuguese', 
            'j': 'Japanese', 'z': 'Chinese', 'h': 'Hindi'
        }

        # STT processing
        user_text = ""
        detected_language = DEFAULT_LANGUAGE
        
        if audio_array.size > 0:
            print_info(f"Processing audio ({len(audio_array) / sample_rate:.1f}s)")
            
            try:
                transcription_result = run_coro_from_sync_thread_with_timeout(
                    voice_assistant.stt_engine.transcribe_with_sample_rate(audio_array, sample_rate),
                    timeout=8.0
                )
                
                if hasattr(transcription_result, 'text'):
                    user_text = transcription_result.text.strip()
                    detected_language = voice_assistant.detect_language_from_text(user_text)
                else:
                    user_text = str(transcription_result).strip()
                    detected_language = DEFAULT_LANGUAGE
                
                # Language detection
                if detected_language != voice_assistant.current_language:
                    voice_assistant.current_language = detected_language
                    lang_name = lang_names.get(detected_language, 'Unknown')
                    print_language(f"Switched to {lang_name} ({detected_language})")
                
            except Exception as e:
                print_error(f"STT failed: {e}")
                user_text = ""
                detected_language = DEFAULT_LANGUAGE

        if not user_text.strip():
            yield SILENT_AUDIO_FRAME_TUPLE, AdditionalOutputs()
            return

        print_user(user_text)

        # LLM response with memory tracking
        start_turn_time = time.monotonic()
        
        try:
            # Check memory state before LLM call
            memory_context_before = None
            if hasattr(voice_assistant, 'memory_manager') and voice_assistant.memory_manager:
                try:
                    memory_context_before = run_coro_from_sync_thread_with_timeout(
                        voice_assistant.memory_manager.get_conversation_context(),
                        timeout=20.0
                    )
                    if memory_context_before:
                        print_memory(f"Context loaded ({len(str(memory_context_before))} chars)")
                except Exception as e:
                    print_memory(f"Context load failed: {e}")
            else:
                print_memory("No memory manager available")
            
            # Make LLM call
            assistant_response_text = run_coro_from_sync_thread_with_timeout(
                voice_assistant.get_llm_response_smart(user_text),
                timeout=10.0
            )
            
            turn_processing_time = time.monotonic() - start_turn_time
            
            # Check memory state after LLM call
            memory_context_after = None
            if hasattr(voice_assistant, 'memory_manager') and voice_assistant.memory_manager:
                try:
                    memory_context_after = run_coro_from_sync_thread_with_timeout(
                        voice_assistant.memory_manager.get_conversation_context(),
                        timeout=4.0
                    )
                    if memory_context_before and memory_context_after:
                        if str(memory_context_before) != str(memory_context_after):
                            print_memory("Context updated with new conversation")
                        else:
                            print_memory("Context unchanged")
                except Exception as e:
                    print_memory(f"Context check failed: {e}")
            
        except TimeoutError:
            assistant_response_text = "Let me think about that and get back to you quickly."
            print_error("LLM request timed out")
        except Exception as e:
            assistant_response_text = "I encountered an error processing your request."
            print_error(f"LLM error: {e}")
        
        turn_processing_time = time.monotonic() - start_turn_time
        print_assistant(assistant_response_text)
        print_timing(f"Response generated in {turn_processing_time:.2f}s")

        # TTS processing
        additional_outputs = AdditionalOutputs()
        
        # Language conversion for TTS
        if len(voice_assistant.current_language) > 1:
            kokoro_language = voice_assistant.convert_to_kokoro_language(voice_assistant.current_language)
            voice_assistant.current_language = kokoro_language
            print_language(f"Converted to Kokoro format: {kokoro_language}")
        
        tts_voices_to_try = voice_assistant.get_voices_for_language(voice_assistant.current_language)
        tts_voices_to_try.append(None)
        
        print_tts(f"Using language '{voice_assistant.current_language}' with {len(tts_voices_to_try)} voice options")
        
        tts_success = False
        for voice_id in tts_voices_to_try:
            try:
                print_tts(f"Trying voice: {voice_id or 'default'}")
                
                chunk_count = 0
                total_samples = 0
                
                for current_sr, current_chunk_array in voice_assistant.stream_tts_synthesis(
                    assistant_response_text, voice_id, voice_assistant.current_language
                ):
                    if isinstance(current_chunk_array, np.ndarray) and current_chunk_array.size > 0:
                        chunk_count += 1
                        total_samples += current_chunk_array.size
                        chunk_size = min(1024, current_chunk_array.size)
                        for i in range(0, current_chunk_array.size, chunk_size):
                            mini_chunk = current_chunk_array[i:i+chunk_size]
                            if mini_chunk.size > 0:
                                yield (current_sr, mini_chunk), additional_outputs
                
                tts_success = True
                print_success(f"TTS completed with voice '{voice_id or 'default'}' ({chunk_count} chunks, {total_samples} samples)")
                break
                    
            except Exception as e:
                print_error(f"TTS failed with voice '{voice_id}': {e}")
                continue
                
        if not tts_success:
            print_error("All TTS attempts failed")
            yield SILENT_AUDIO_FRAME_TUPLE, additional_outputs

    except Exception as e:
        print_error(f"Critical error in callback: {e}")
        yield EMPTY_AUDIO_YIELD_OUTPUT

def setup_async_environment():
    """Setup async environment with colorful output."""
    global main_event_loop, voice_assistant, async_worker_thread
    
    print_info("Creating VoiceAssistant instance...")
    voice_assistant = VoiceAssistant()

    def run_async_loop_in_thread():
        global main_event_loop, voice_assistant
        main_event_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(main_event_loop)
        
        if voice_assistant:
            print_info("Initializing async components...")
            main_event_loop.run_until_complete(voice_assistant.initialize_async())
            
            # Check memory manager initialization
            if hasattr(voice_assistant, 'memory_manager') and voice_assistant.memory_manager:
                print_memory("Memory manager initialized successfully")
            else:
                print_memory("Memory manager not available")
        else:
            print_error("Voice assistant instance is None in async thread")
            return

        try:
            main_event_loop.run_forever()
        except KeyboardInterrupt:
            print_info("Async loop interrupted")
        finally:
            if voice_assistant and main_event_loop and not main_event_loop.is_closed():
                print_info("Cleaning up assistant resources...")
                main_event_loop.run_until_complete(voice_assistant.cleanup_async())
            if main_event_loop and not main_event_loop.is_closed():
                 main_event_loop.close()
            print_info("Async event loop closed")

    async_worker_thread = threading.Thread(target=run_async_loop_in_thread, daemon=True, name="AsyncWorkerThread")
    async_worker_thread.start()

    # Wait for initialization
    for _ in range(100):
        if main_event_loop and main_event_loop.is_running() and voice_assistant:
            print_success("Async environment ready")
            return
        time.sleep(0.1)
    print_error("Async environment setup timeout")

def run_coro_from_sync_thread_with_timeout(coro, timeout: float = 4.0) -> any:
    """Run coroutine with timeout."""
    global main_event_loop
    if main_event_loop and main_event_loop.is_running():
        import asyncio
        import time
        
        start_time = time.monotonic()
        
        future = asyncio.run_coroutine_threadsafe(
            asyncio.wait_for(coro, timeout=timeout),
            main_event_loop
        )
        try:
            result = future.result(timeout=timeout + 1.0)
            elapsed = time.monotonic() - start_time
            return result
        except asyncio.TimeoutError:
            elapsed = time.monotonic() - start_time
            print_error(f"Async task timed out after {timeout}s")
            raise TimeoutError(f"Operation timed out after {timeout}s")
        except Exception as e:
            elapsed = time.monotonic() - start_time
            print_error(f"Async task error: {e}")
            return "I encountered an error processing your request."
    else:
        print_error("Event loop not available")
        return "My processing system is not ready."

def main():
    """Colorful main function."""
    print(f"{Fore.CYAN}{Style.BRIGHT}🎨 Colorful FastRTC Voice Assistant{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}{'=' * 50}{Style.RESET_ALL}")
    
    setup_async_environment()

    bridge = FastRTCBridge()
    bridge.create_stream(voice_assistant_callback_rt)
    
    try:
        print(f"{Fore.GREEN}{Style.BRIGHT}💡 Test the assistant with:{Style.RESET_ALL}")
        print(f"{Fore.WHITE}   • 'My name is [Your Name]'{Style.RESET_ALL}")
        print(f"{Fore.WHITE}   • 'What is my name?'{Style.RESET_ALL}")
        print(f"{Fore.WHITE}   • 'I like [something]'{Style.RESET_ALL}")
        print(f"{Fore.WHITE}   • 'What do you know about me?'{Style.RESET_ALL}")
        print(f"{Fore.WHITE}   • 'Tell me about myself'{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}{'=' * 50}{Style.RESET_ALL}")
        
        bridge.launch_stream()
        
    except KeyboardInterrupt:
        print_info("Shutting down...")
    except Exception as e:
        print_error(f"Launch error: {e}")
    finally:
        print(f"\n{Fore.CYAN}{Style.BRIGHT}👋 Session ended{Style.RESET_ALL}")

if __name__ == "__main__":
    main()