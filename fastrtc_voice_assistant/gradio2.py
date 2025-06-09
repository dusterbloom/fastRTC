#!/usr/bin/env python3
"""
FastRTC Voice Assistant with Refactored Architecture - Simple V4-Style Interface
Uses the refactored voice assistant architecture with the same UI as V4
"""

import sys
import time
import os
import numpy as np
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
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

from fastrtc.utils import AdditionalOutputs
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional
import logging
from collections import deque

# Add src to path for imports
sys.path.insert(0, 'src')

# Import refactored components
from src.core.voice_assistant import VoiceAssistant
from src.core.interfaces import AudioData, TranscriptionResult
from src.utils.logging import setup_logging, get_logger
from src.config.settings import DEFAULT_LANGUAGE
import traceback

# Set up logging (same as V4)
logging.basicConfig(level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("aioice.ice").setLevel(logging.WARNING)
logging.getLogger("aiortc").setLevel(logging.WARNING)
logging.getLogger("phonemizer").setLevel(logging.WARNING)

logger = get_logger(__name__)

# --- Configuration (Same as V4) ---
USE_OLLAMA_FOR_CONVERSATION = True
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_CONVERSATIONAL_MODEL = os.getenv("OLLAMA_CONVERSATIONAL_MODEL", "llama3:8b-instruct-q4_K_M")

# LM Studio settings (only used if USE_OLLAMA_FOR_CONVERSATION is False)
LM_STUDIO_URL = os.getenv("LM_STUDIO_URL", "http://192.168.1.5:1234/v1")
LM_STUDIO_MODEL = os.getenv("LM_STUDIO_MODEL", "mistral-nemo-instruct-2407")

# A-MEM settings
AMEM_LLM_MODEL = os.getenv("AMEM_LLM_MODEL", "llama3.2:3b")
AMEM_EMBEDDER_MODEL = os.getenv("AMEM_EMBEDDER_MODEL", "all-MiniLM-L6-v2")

KOKORO_PREFERRED_VOICE = "af_heart"
KOKORO_FALLBACK_VOICE_1 = "af_alloy"
KOKORO_FALLBACK_VOICE_2 = "af_bella"

AUDIO_SAMPLE_RATE = 16000
MINIMAL_SILENT_FRAME_DURATION_MS = 20
MINIMAL_SILENT_SAMPLES = int(AUDIO_SAMPLE_RATE * (MINIMAL_SILENT_FRAME_DURATION_MS / 1000.0))
SILENT_AUDIO_CHUNK_ARRAY = np.zeros(MINIMAL_SILENT_SAMPLES, dtype=np.float32)
SILENT_AUDIO_FRAME_TUPLE = (AUDIO_SAMPLE_RATE, SILENT_AUDIO_CHUNK_ARRAY)
EMPTY_AUDIO_YIELD_OUTPUT = (SILENT_AUDIO_FRAME_TUPLE, AdditionalOutputs())

# Language code mappings for Kokoro TTS (same as V4)
WHISPER_TO_KOKORO_LANG = {
    'en': 'a',    # American English (default)
    'it': 'i',    # Italian  
    'es': 'e',    # Spanish
    'fr': 'f',    # French
    'de': 'a',    # German -> fallback to English (not natively supported)
    'pt': 'p',    # Portuguese -> Brazilian Portuguese
    'ja': 'j',    # Japanese
    'ko': 'a',    # Korean -> fallback to English (not natively supported)
    'zh': 'z',    # Chinese -> Mandarin Chinese
    'hi': 'h',    # Hindi
}

# Language to voice mapping with official Kokoro voice names (same as V4)
KOKORO_VOICE_MAP = {
    'a': ['af_heart', 'af_bella', 'af_sarah'],                    # American English (best quality voices)
    'b': ['bf_emma', 'bf_isabella', 'bm_george'],                 # British English  
    'i': ['if_sara', 'im_nicola'],                                # Italian
    'e': ['ef_dora', 'em_alex', 'em_santa'],                      # Spanish
    'f': ['ff_siwis'],                                             # French (only one voice)
    'p': ['pf_dora', 'pm_alex', 'pm_santa'],                      # Brazilian Portuguese
    'j': ['jf_alpha', 'jf_gongitsune', 'jm_kumo'],               # Japanese
    'z': ['zf_xiaobei', 'zf_xiaoni', 'zm_yunjian', 'zm_yunxi'],  # Mandarin Chinese
    'h': ['hf_alpha', 'hf_beta', 'hm_omega'],                     # Hindi
}

# TTS language mapping for Kokoro (same as V4)
KOKORO_TTS_LANG_MAP = {
    'a': 'en-us',  # American English
    'b': 'en-gb',  # British English
    'i': 'it',     # Italian
    'e': 'es',     # Spanish
    'f': 'fr-fr',  # French
    'p': 'pt-br',  # Brazilian Portuguese
    'j': 'ja-jp',  # Japanese
    'z': 'zh-cn',  # Mandarin Chinese
    'h': 'hi-in',  # Hindi
}

DEFAULT_LANGUAGE = 'a'  # American English

print("🧠 FastRTC Voice Assistant - Refactored Architecture with V4 Interface")
print("=" * 75)

def print_status(message):
    """Same print_status function as V4"""
    timestamp = time.strftime("%H:%M:%S")
    logger.info(f"{message}")

# Global instances (same pattern as V4)
voice_assistant: Optional[VoiceAssistant] = None
main_event_loop: Optional[asyncio.AbstractEventLoop] = None
async_worker_thread: Optional[threading.Thread] = None

def setup_async_environment():
    """Setup async environment using V4 pattern but with refactored VoiceAssistant"""
    global main_event_loop, voice_assistant, async_worker_thread
    
    # Create refactored voice assistant instead of SmartVoiceAssistant
    voice_assistant = VoiceAssistant()

    def run_async_loop_in_thread():
        global main_event_loop, voice_assistant
        main_event_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(main_event_loop)
        
        if voice_assistant:
            # Initialize refactored components
            main_event_loop.run_until_complete(voice_assistant.initialize_async())
        else:
            print_status("🚨 Voice assistant instance is None in async thread. Cannot initialize.")
            return

        try:
            main_event_loop.run_forever()
        except KeyboardInterrupt:
            print_status("Async loop interrupted in thread.")
        finally:
            if voice_assistant and main_event_loop and not main_event_loop.is_closed():
                print_status("Cleaning up assistant resources in async thread...")
                main_event_loop.run_until_complete(voice_assistant.cleanup_async())
            if main_event_loop and not main_event_loop.is_closed():
                 main_event_loop.close()
            print_status("Async event loop closed.")

    async_worker_thread = threading.Thread(target=run_async_loop_in_thread, daemon=True, name="AsyncWorkerThread")
    async_worker_thread.start()

    # Wait for initialization (same pattern as V4)
    for _ in range(100):
        if main_event_loop and main_event_loop.is_running() and voice_assistant:
            print_status("✅ Async environment and refactored components are ready.")
            return
        time.sleep(0.1)
    print_status("⚠️ Async environment or components did not confirm readiness in time.")

def run_coro_from_sync_thread_with_timeout(coro, timeout: float = 4.0) -> any:
    """Run coroutine with timeout to prevent WebRTC disconnections (same as V4)"""
    global main_event_loop
    if main_event_loop and main_event_loop.is_running():
        import asyncio
        import time
        
        # DIAGNOSTIC: Log timing details
        start_time = time.monotonic()
        logger.info(f"🔧 Async Debug: Starting coroutine with timeout={timeout}s")
        
        future = asyncio.run_coroutine_threadsafe(
            asyncio.wait_for(coro, timeout=timeout),
            main_event_loop
        )
        try:
            result = future.result(timeout=timeout + 1.0)  # Add 1s buffer
            elapsed = time.monotonic() - start_time
            logger.info(f"🔧 Async Debug: Coroutine completed in {elapsed:.2f}s")
            return result
        except asyncio.TimeoutError:
            elapsed = time.monotonic() - start_time
            logger.error(f"🔧 Async Debug: Timeout after {elapsed:.2f}s (limit was {timeout}s)")
            print_status(f"❌ Async task timed out after {timeout}s")
            raise TimeoutError(f"Operation timed out after {timeout}s")
        except Exception as e:
            elapsed = time.monotonic() - start_time
            logger.error(f"🔧 Async Debug: Error after {elapsed:.2f}s: {e}")
            print_status(f"❌ Error in async task: {e}")
            return "I encountered an error processing your request."
    else:
        print_status("❌ Event loop not available")
        return "My processing system is not ready."

def voice_assistant_callback_rt(audio_data_tuple: tuple):
    """
    Main callback function using V4 pattern but with refactored components.
    Same signature and flow as V4, but using refactored VoiceAssistant.
    """
    global voice_assistant

    # DIAGNOSTIC: Log callback entry
    logger.info(f"🔧 Callback Debug: Entry - audio_data_tuple type={type(audio_data_tuple)}")
    if isinstance(audio_data_tuple, tuple) and len(audio_data_tuple) == 2:
        sr, arr = audio_data_tuple
        if hasattr(arr, 'shape'):
            logger.info(f"🔧 Callback Debug: Input audio - SR={sr}, shape={arr.shape}, dtype={getattr(arr, 'dtype', 'unknown')}")
        else:
            logger.info(f"🔧 Callback Debug: Input audio - SR={sr}, type={type(arr)}")

    if not voice_assistant:
        logger.error("🔧 Callback Debug: voice_assistant is None!")
        yield EMPTY_AUDIO_YIELD_OUTPUT
        return

    try:
        # CRITICAL FIX: Skip audio processor for STT - use raw audio directly like debug_audio_pipeline.py
        logger.info(f"🔧 Callback Debug: Using RAW audio directly (bypassing processor)")
        
        if isinstance(audio_data_tuple, tuple) and len(audio_data_tuple) == 2:
            sample_rate, raw_audio_array = audio_data_tuple
            
            # CRITICAL FIX: Proper audio array handling for shape (1, N) -> (N,)
            if isinstance(raw_audio_array, np.ndarray) and len(raw_audio_array.shape) > 1:
                if raw_audio_array.shape[0] == 1:  # Shape like (1, 384000) - single channel
                    audio_array = raw_audio_array[0]  # Extract the actual audio data: (384000,)
                elif raw_audio_array.shape[1] == 1:  # Shape like (384000, 1) - single column
                    audio_array = raw_audio_array[:, 0]  # Extract: (384000,)
                elif raw_audio_array.shape[1] > 1:  # Multi-channel like (384000, 2)
                    audio_array = np.mean(raw_audio_array, axis=1)  # Convert to mono: (384000,)
                else:
                    audio_array = raw_audio_array.flatten()
            else:
                audio_array = raw_audio_array if isinstance(raw_audio_array, np.ndarray) else np.array(raw_audio_array, dtype=np.float32)
            
            # Convert to float32 (same as voice_assistant.process_audio_array)
            if isinstance(audio_array, np.ndarray):
                if audio_array.dtype == np.int16:
                    # Convert int16 to float32 properly
                    audio_array = audio_array.astype(np.float32) / 32768.0
                else:
                    audio_array = audio_array.astype(np.float32)
        else:
            audio_array = np.array([], dtype=np.float32)
            sample_rate = 16000
        
        logger.info(f"🔧 Callback Debug: RAW audio - SR={sample_rate}, shape={audio_array.shape}, size={audio_array.size}")

        if audio_array.size == 0:
            logger.info(f"🔧 Callback Debug: Empty audio array, yielding silent frame")
            yield SILENT_AUDIO_FRAME_TUPLE, AdditionalOutputs()
            return

        # Language names (same as V4)
        lang_names = {
            'a': 'American English', 'b': 'British English', 'i': 'Italian', 
            'e': 'Spanish', 'f': 'French', 'p': 'Portuguese', 
            'j': 'Japanese', 'z': 'Chinese', 'h': 'Hindi'
        }

        # SIMPLIFIED STT processing with proper sample rate handling
        user_text = ""
        detected_language = DEFAULT_LANGUAGE
        
        if audio_array.size > 0:
            # DIAGNOSTIC: Log audio data details
            logger.info(f"🔧 STT Debug: Audio array shape={audio_array.shape}, dtype={audio_array.dtype}, size={audio_array.size}")
            logger.info(f"🔧 STT Debug: Sample rate={sample_rate}, duration={len(audio_array) / sample_rate:.2f}s")
            logger.info(f"🔧 STT Debug: Audio range=[{np.min(audio_array):.6f}, {np.max(audio_array):.6f}]")
            
            # CRITICAL FIX: Use new method with sample rate and longer timeout
            logger.info(f"🔧 STT Debug: Using sample rate aware STT call")
            try:
                transcription_result = run_coro_from_sync_thread_with_timeout(
                    voice_assistant.stt_engine.transcribe_with_sample_rate(audio_array, sample_rate),
                    timeout=8.0  # Increased timeout for better reliability
                )
                
                # Extract text and language from enhanced STT result
                if hasattr(transcription_result, 'text'):
                    user_text = transcription_result.text.strip()
                    # FIXED: Use MediaPipe language detection ONLY (ignore STT language detection)
                    logger.info(f"🔧 Language Debug: Ignoring STT language detection, using MediaPipe only")
                    detected_language = voice_assistant.detect_language_from_text(user_text)
                    logger.info(f"🌍 Language detected by MediaPipe: {detected_language}")
                else:
                    # Error case - use string directly
                    user_text = str(transcription_result).strip()
                    detected_language = DEFAULT_LANGUAGE
                
                print_status(f"📝 Transcribed: '{user_text}'")
                
                # Update current language
                if detected_language != voice_assistant.current_language:
                    voice_assistant.current_language = detected_language
                    lang_name = lang_names.get(detected_language, 'Unknown')
                    print_status(f"🌍 Language switched to: {lang_name} ({detected_language})")
                else:
                    lang_name = lang_names.get(detected_language, 'Unknown')
                    print_status(f"🌍 Language confirmed: {lang_name} ({detected_language})")
                    
            except Exception as e:
                logger.error(f"🔧 STT Debug: Enhanced STT failed: {e}")
                user_text = ""
                detected_language = DEFAULT_LANGUAGE

        if not user_text.strip():
            yield SILENT_AUDIO_FRAME_TUPLE, AdditionalOutputs()
            return

        print_status(f"👤 User: {user_text}")

        # LLM response using refactored service
        start_turn_time = time.monotonic()
        try:
            assistant_response_text = run_coro_from_sync_thread_with_timeout(
                voice_assistant.get_llm_response_smart(user_text),
                timeout=4.0
            )
        except TimeoutError:
            assistant_response_text = "Let me think about that and get back to you quickly."
        
        turn_processing_time = time.monotonic() - start_turn_time
        print_status(f"🤖 Assistant: {assistant_response_text}")
        print_status(f"⏱️ Turn Processing Time: {turn_processing_time:.2f}s")

        # TTS using refactored engine
        additional_outputs = AdditionalOutputs()
        
        # DIAGNOSTIC: Check if language is already in Kokoro format
        logger.info(f"🔧 Language Debug: Current language before conversion: '{voice_assistant.current_language}'")
        
        # Only convert if it's NOT already in Kokoro format (single letter)
        if len(voice_assistant.current_language) > 1:
            kokoro_language = voice_assistant.convert_to_kokoro_language(voice_assistant.current_language)
            logger.info(f"🔧 Language Debug: Converting '{voice_assistant.current_language}' → '{kokoro_language}'")
            voice_assistant.current_language = kokoro_language
        else:
            logger.info(f"🔧 Language Debug: Language '{voice_assistant.current_language}' already in Kokoro format")
        
        tts_voices_to_try = voice_assistant.get_voices_for_language(voice_assistant.current_language)
        tts_voices_to_try.append(None)
        
        print_status(f"🎤 TTS using language '{voice_assistant.current_language}' with voices: {tts_voices_to_try[:3]}")
        
        tts_success = False
        for voice_id in tts_voices_to_try:
            try:
                # DIAGNOSTIC: Log TTS engine status
                logger.info(f"🔧 TTS Debug: Attempting TTS with voice='{voice_id}', language='{voice_assistant.current_language}'")
                logger.info(f"🔧 TTS Debug: TTS engine available: {voice_assistant.tts_engine.is_available()}")
                logger.info(f"🔧 TTS Debug: TTS engine type: {type(voice_assistant.tts_engine)}")
                
                # FIXED: Use refactored TTS engine streaming method
                logger.info(f"🔧 TTS Debug: Using refactored TTS engine stream_synthesis method")
                
                chunk_count = 0
                total_samples = 0
                
                # Use the refactored engine's streaming method
                for current_sr, current_chunk_array in voice_assistant.stream_tts_synthesis(
                    assistant_response_text, voice_id, voice_assistant.current_language
                ):
                    if isinstance(current_chunk_array, np.ndarray) and current_chunk_array.size > 0:
                        chunk_count += 1
                        total_samples += current_chunk_array.size
                        # Yield smaller chunks to prevent timeouts
                        chunk_size = min(1024, current_chunk_array.size)
                        for i in range(0, current_chunk_array.size, chunk_size):
                            mini_chunk = current_chunk_array[i:i+chunk_size]
                            if mini_chunk.size > 0:
                                yield (current_sr, mini_chunk), additional_outputs
                
                tts_success = True
                logger.info(f"✅ TTS stream completed using refactored engine. Voice: {voice_id}, Chunks: {chunk_count}, Samples: {total_samples}")
                print_status(f"✅ TTS SUCCESS using refactored engine with voice: {voice_id}")
                break
                    
            except Exception as e:
                logger.error(f"❌ TTS failed with voice '{voice_id}': {e}")
                print_status(f"❌ TTS failed with voice '{voice_id}': {e}")
                # Add more detailed error logging
                import traceback
                logger.error(f"TTS Error traceback: {traceback.format_exc()}")
                continue
                
        if not tts_success:
            print_status(f"❌ All TTS attempts failed")
            yield SILENT_AUDIO_FRAME_TUPLE, additional_outputs

    except Exception as e:
        print_status(f"❌ CRITICAL Error: {e}")
        import traceback
        traceback.print_exc()
        
        # Graceful error recovery (same as V4)
        try:
            error_msg = "Sorry, I encountered an error. Please try again."
            # Use fallback TTS for error message
            yield SILENT_AUDIO_FRAME_TUPLE, AdditionalOutputs()
        except Exception:
            yield EMPTY_AUDIO_YIELD_OUTPUT

if __name__ == "__main__":
    # Same setup as V4
    setup_async_environment()

    # Same threshold logic as V4
    if not voice_assistant:
        threshold = 0.15
    else:
        threshold = 0.15  # Use default since we don't have audio_processor.noise_floor

    print_status("🌐 Creating FastRTC stream with refactored architecture...")
    try:
        # Exactly same Stream creation as V4
        stream = Stream(
            ReplyOnPause(
                voice_assistant_callback_rt,
                can_interrupt=True,
                algo_options=AlgoOptions(
                    audio_chunk_duration=2.0,
                    started_talking_threshold=0.15,
                    speech_threshold=threshold
                ),
                model_options=SileroVadOptions(
                    threshold=0.3,
                    min_speech_duration_ms=250,
                    min_silence_duration_ms=2000,
                    speech_pad_ms=200,
                    window_size_samples=512
                )
            ),
            modality="audio", mode="send-receive",
            track_constraints={
                "echoCancellation": True,
                "noiseSuppression": True,
                "autoGainControl": True,
                "sampleRate": {"ideal": 16000},
                "sampleSize": {"ideal": 16},
                "channelCount": {"exact": 1},
                "latency": {"ideal": 0.01},
            }
        )
        
        # Same output messages as V4
        print("=" * 70)
        print("🚀 FastRTC Voice Assistant with Refactored Architecture Ready!")
        print("🎤 Using refactored STT, TTS, Memory, and LLM services")
        print("="*70)
        print("💡 Test Commands:")
        print("   • 'My name is [Your Name]'")
        print("   • 'What is my name?' / 'Who am I?'")
        print("   • 'I like [something interesting]'")
        print("   • Ask questions in supported languages.")
        print("\n🛑 To stop: Press Ctrl+C in the terminal")
        print("=" * 70)

        # Same UI launch as V4
        stream.ui.launch(server_name="0.0.0.0", server_port=7860, quiet=False, share=False)

    except KeyboardInterrupt:
        print_status("🛑 KeyboardInterrupt received. Shutting down...")
    except Exception as e:
        print_status(f"❌ Launch error or unhandled exception in main: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Same cleanup as V4
        print_status("🏁 Main thread initiating shutdown sequence...")
        if main_event_loop and not main_event_loop.is_closed():
            print_status("Requesting async event loop to stop...")
            main_event_loop.call_soon_threadsafe(main_event_loop.stop)

        if async_worker_thread and async_worker_thread.is_alive():
            print_status("Waiting for async worker thread to join...")
            async_worker_thread.join(timeout=15)
            if async_worker_thread.is_alive():
                print_status("⚠️ Async worker thread did not join in time.")

        print_status("👋 Refactored voice assistant shutdown process complete.")
        sys.exit(0)