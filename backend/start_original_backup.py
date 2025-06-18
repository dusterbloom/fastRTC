
#!/usr/bin/env python3
"""
Colorful FastRTC Voice Assistant – FastAPI edition
-------------------------------------------------
Launch with::

    fastRTC/python -m uvicorn backend.start:app --reload --port 8000

This file keeps all the original logic (audio > STT > LLM > TTS), but
wraps the Stream in a FastAPI app so that any WebRTC-capable front-end
(e.g. Rohan Prichard’s React demo) can talk to it.
"""

from __future__ import annotations

# ────────────────────────── STANDARD LIB  ────────────────────────────
import asyncio
import os
import sys
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, List, Optional, Tuple

# ────────────────────────── THIRD-PARTY  ─────────────────────────────
import numpy as np
from colorama import Back, Fore, Style, init
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastrtc import (ReplyOnPause, Stream, audio_to_bytes, get_tts_model, SileroVadOptions, AlgoOptions)
from fastrtc.utils import AdditionalOutputs

# ────────────────────────── APP INTERNALS  ───────────────────────────
# Make local packages importable
sys.path.insert(0, str(Path(__file__).parent.resolve()))
from src.config.audio_config import (  # noqa: E402
    AUDIO_SAMPLE_RATE,
    MINIMAL_SILENT_FRAME_DURATION_MS,
    MINIMAL_SILENT_SAMPLES,
    SILENT_AUDIO_CHUNK_ARRAY,
    SILENT_AUDIO_FRAME_TUPLE,
)
from src.config.settings import DEFAULT_LANGUAGE, load_config  # noqa: E402
from src.core.interfaces import AudioData, TranscriptionResult  # noqa: E402
from src.core.voice_assistant import VoiceAssistant  # noqa: E402
from src.utils.logging import get_logger, setup_logging  # noqa: E402
from src.utils.sota_adaptive_vad import SimpleSOTAAdaptiveVAD, VADConfig

# ────────────────────────── INITIAL SET-UP  ─────────────────────────
init(autoreset=True)                # colour logging
setup_logging()
logger = get_logger(__name__)

# Globals the rest of the file relies on
voice_assistant: Optional[VoiceAssistant] = None
main_event_loop: Optional[asyncio.AbstractEventLoop] = None
async_worker_thread: Optional[threading.Thread] = None

# ────────────────────────── PRINT HELPERS  ──────────────────────────
def _timestamp() -> str:
    return time.strftime("%H:%M:%S")


def print_colorful(msg: str, color=Fore.WHITE, style=Style.NORMAL) -> None:
    print(f"{Fore.CYAN}[{_timestamp()}]{Style.RESET_ALL} {style}{color}{msg}{Style.RESET_ALL}")


def print_user(m):        print_colorful(f" User: {m}", Fore.BLUE, Style.BRIGHT)
def print_assistant(m):   print_colorful(f" Assistant: {m}", Fore.GREEN, Style.BRIGHT)
def print_memory(m):      print_colorful(f" Memory: {m}", Fore.MAGENTA, Style.BRIGHT)
def print_language(m):    print_colorful(f" Language: {m}", Fore.YELLOW, Style.BRIGHT)
def print_tts(m):         print_colorful(f" TTS: {m}", Fore.CYAN, Style.BRIGHT)
def print_error(m):       print_colorful(f"❌ Error: {m}", Fore.RED, Style.BRIGHT)
def print_success(m):     print_colorful(f"✅ Success: {m}", Fore.GREEN, Style.BRIGHT)
def print_info(m):        print_colorful(f"ℹ️ Info: {m}", Fore.WHITE, Style.NORMAL)
def print_timing(m):      print_colorful(f"⏱️ Timing: {m}", Fore.LIGHTBLUE_EX, Style.NORMAL)

# ────────────────────────── AUDIO CALLBACK  ─────────────────────────
def voice_assistant_callback_rt(audio_data_tuple: tuple):
    """
    Enhanced callback with SOTA adaptive VAD.
    """
    global assistant_processor, adaptive_vad

    # --- FIX: Initialize the adaptive VAD here, only once ---
    if 'adaptive_vad' not in globals():
        vad_config = VADConfig() # Uses defaults from the sota_adaptive_vad.py file
        adaptive_vad = SimpleSOTAAdaptiveVAD(vad_config)
    
    try:
        # Process audio input (same as before)
        if isinstance(audio_data_tuple, tuple) and len(audio_data_tuple) == 2:
            sample_rate, raw_audio_array = audio_data_tuple
            
            # Audio preprocessing (keep your existing code)
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

        # --- ENHANCED VAD ADAPTATION ---
        # Calculate speech duration
        speech_duration_s = len(audio_array) / sample_rate if sample_rate > 0 else 0
        
        # Record the turn with full audio analysis
        adaptive_vad.record_turn(speech_duration_s, audio_array, sample_rate)
        
        # Get updated VAD options
        new_vad_options: SileroVadOptions = adaptive_vad.get_current_vad_options(speech_duration_s)
        
        # Get VAD status for logging
        vad_status = adaptive_vad.get_status()
        print_info(f"VAD: silence={vad_status['current_silence_ms']}ms, "
                  f"recovery={vad_status['in_recovery']}, "
                  f"SNR={vad_status['avg_snr']}")
        
        if (assistant_processor and hasattr(assistant_processor, 'model_options') and
            (abs(assistant_processor.model_options.min_silence_duration_ms -
                 new_vad_options.min_silence_duration_ms) > 100 or
             vad_status['in_recovery'])):
            
            print_info(f"Updating VAD parameters: silence_ms -> {new_vad_options.min_silence_duration_ms}")
            assistant_processor.model_options = new_vad_options

        # Prepare for STT processing
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
        logger.exception("Full traceback for critical callback error:")
        yield EMPTY_AUDIO_YIELD_OUTPUT


# ────────────────────────── ASYNC ENVIRONMENT  ───────────────────────
def setup_async_environment() -> None:
    """
    Starts an event-loop in a background thread and initialises the
    VoiceAssistant instance so it’s ready when WebRTC traffic arrives.
    """
    global main_event_loop, voice_assistant, async_worker_thread

    print_info("Creating VoiceAssistant instance …")
    voice_assistant = VoiceAssistant(config=load_config())

    def _run_loop() -> None:
        global main_event_loop
        main_event_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(main_event_loop)

        if voice_assistant:
            print_info("Initialising async components …")
            main_event_loop.run_until_complete(voice_assistant.initialize_async())
            print_success("VoiceAssistant async components ready")

        main_event_loop.run_forever()

    async_worker_thread = threading.Thread(
        target=_run_loop, daemon=True, name="AsyncWorkerThread"
    )
    async_worker_thread.start()

    # Spin-wait until the loop is up
    for _ in range(100):
        if main_event_loop and main_event_loop.is_running():
            return
        time.sleep(0.1)
    print_error("Async environment setup timeout")


def run_coro_from_sync_thread_with_timeout(coro, timeout: float = 4.0) -> Any:
    """
    Utility helper needed by the callback – unchanged.
    """
    global main_event_loop
    if not (main_event_loop and main_event_loop.is_running()):
        raise RuntimeError("Event loop not available")

    future = asyncio.run_coroutine_threadsafe(
        asyncio.wait_for(coro, timeout=timeout), main_event_loop
    )
    return future.result(timeout=timeout + 1)


# ────────────────────────── BUILD THE STREAM  ────────────────────────
setup_async_environment()  # initialise before we build the Stream

# Prepare the empty audio frame tuple and additional outputs
assistant_processor = ReplyOnPause(
    voice_assistant_callback_rt,
    can_interrupt=True,
    algo_options=AlgoOptions(
        # This is the GATEKEEPER. We are making it extremely sensitive.
        speech_threshold=0.05,  # Drastically lower: will detect even quiet speech.
        started_talking_threshold=0.1,
        audio_chunk_duration=0.5 # Process audio in smaller chunks for responsiveness
    ),
    model_options=SileroVadOptions(
        # This is the TIMER. We are keeping it very patient.
        threshold=0.2,                  # More sensitive model threshold
        min_speech_duration_ms=150,     # Catches very short words like "a" or "I"
        min_silence_duration_ms=4000,   # PATIENCE: Waits 4 seconds of pure silence
        speech_pad_ms=500               # Generous buffer at the end of your speech
    )
)

# Now, create the stream using the processor instance we just created.
assistant_stream = Stream(
    assistant_processor,
    modality="audio",
    mode="send-receive",
)


# ────────────────────────── FASTAPI APP  ────────────────────────────
app = FastAPI(title="FastRTC Voice Assistant", version="1.0.0")

# CORS so the React/Vite front-end can hit the endpoints
origins = [
    "http://localhost:5173",               # Original Vite dev-server port
    "http://localhost:3001",               # Next.js dev-server port from screenshot
    "http://127.0.0.1:3001",             # Explicit IP for Next.js dev-server
    os.getenv("FRONTEND_URL", ""),         # prod domain, if set
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o for o in origins if o],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount FastRTC routes → /assistant/webrtc/offer  &  /assistant/ws
assistant_stream.mount(app, path="/assistant")

# Optional: serve a built SPA from /frontend/dist
_frontend_dist = Path(__file__).parent / "frontend" / "dist"
if _frontend_dist.exists():
    app.mount("/", StaticFiles(directory=_frontend_dist, html=True), name="spa")

# ────────────────────────── ENTRY POINT  ────────────────────────────
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "backend.start:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        reload=True,
    )