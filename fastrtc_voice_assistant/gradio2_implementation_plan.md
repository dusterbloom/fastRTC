# Gradio2.py Implementation Plan

## Overview

Create `gradio2.py` that maintains the **exact same UI and flow** as `voice_assistant_with_memory_V4_MULTILINGUAL.py` but uses the **refactored architecture** from `src/`. This provides the simple, working V4 interface while leveraging all the benefits of the new modular system.

## Key Principles

1. **Keep V4's Simplicity**: Direct FastRTC Stream with ReplyOnPause callback
2. **Use Refactored Code**: Import and use the new [`VoiceAssistant`](src/core/voice_assistant.py) class
3. **Same UI**: FastRTC's built-in web interface (`stream.ui.launch()`)
4. **Same Logging**: Simple [`print_status()`](voice_assistant_with_memory_V4_MULTILINGUAL.py:143) function like V4

## Architecture Comparison

### V4 Structure (Keep This Pattern)
```
voice_assistant_with_memory_V4_MULTILINGUAL.py
├── Imports (FastRTC + monolithic components)
├── Configuration constants
├── print_status function
├── SmartVoiceAssistant class (monolithic)
├── setup_async_environment function
├── voice_assistant_callback_rt function
└── Main execution with Stream.ui.launch()
```

### Gradio2.py Structure (New)
```
gradio2.py
├── Imports (FastRTC + refactored components)
├── Configuration constants (same as V4)
├── print_status function (same as V4)
├── setup_async_environment (V4 pattern + refactored VoiceAssistant)
├── voice_assistant_callback_rt (V4 pattern + refactored calls)
└── Main execution (exactly same as V4)
```

## Implementation Details

### 1. Imports Section
```python
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
```

### 2. Configuration Constants (Same as V4)
```python
# Same configuration as V4
USE_OLLAMA_FOR_CONVERSATION = True
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_CONVERSATIONAL_MODEL = os.getenv("OLLAMA_CONVERSATIONAL_MODEL", "llama3:8b-instruct-q4_K_M")

# Audio constants (same as V4)
AUDIO_SAMPLE_RATE = 16000
MINIMAL_SILENT_FRAME_DURATION_MS = 20
MINIMAL_SILENT_SAMPLES = int(AUDIO_SAMPLE_RATE * (MINIMAL_SILENT_FRAME_DURATION_MS / 1000.0))
SILENT_AUDIO_CHUNK_ARRAY = np.zeros(MINIMAL_SILENT_SAMPLES, dtype=np.float32)
SILENT_AUDIO_FRAME_TUPLE = (AUDIO_SAMPLE_RATE, SILENT_AUDIO_CHUNK_ARRAY)
EMPTY_AUDIO_YIELD_OUTPUT = (SILENT_AUDIO_FRAME_TUPLE, AdditionalOutputs())

# Language mappings (same as V4)
WHISPER_TO_KOKORO_LANG = {
    'en': 'a', 'it': 'i', 'es': 'e', 'fr': 'f', 'de': 'a',
    'pt': 'p', 'ja': 'j', 'ko': 'a', 'zh': 'z', 'hi': 'h',
}

KOKORO_VOICE_MAP = {
    'a': ['af_heart', 'af_bella', 'af_sarah'],
    'b': ['bf_emma', 'bf_isabella', 'bm_george'],
    'i': ['if_sara', 'im_nicola'],
    'e': ['ef_dora', 'em_alex', 'em_santa'],
    'f': ['ff_siwis'],
    'p': ['pf_dora', 'pm_alex', 'pm_santa'],
    'j': ['jf_alpha', 'jf_gongitsune', 'jm_kumo'],
    'z': ['zf_xiaobei', 'zf_xiaoni', 'zm_yunjian', 'zm_yunxi'],
    'h': ['hf_alpha', 'hf_beta', 'hm_omega'],
}

DEFAULT_LANGUAGE = 'a'
```

### 3. Logging Setup (Same as V4)
```python
# Set up logging (same as V4)
logging.basicConfig(level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("aioice.ice").setLevel(logging.WARNING)
logging.getLogger("aiortc").setLevel(logging.WARNING)
logging.getLogger("phonemizer").setLevel(logging.WARNING)

logger = get_logger(__name__)

print("🧠 FastRTC Voice Assistant - Refactored Architecture with V4 Interface")
print("=" * 75)

def print_status(message):
    """Same print_status function as V4"""
    timestamp = time.strftime("%H:%M:%S")
    logger.info(f"{message}")
```

### 4. Global Variables (Same as V4)
```python
# Global instances (same pattern as V4)
voice_assistant: Optional[VoiceAssistant] = None
main_event_loop: Optional[asyncio.AbstractEventLoop] = None
async_worker_thread: Optional[threading.Thread] = None
```

### 5. Async Environment Setup (V4 Pattern + Refactored Components)
```python
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
```

### 6. Utility Functions (Same as V4)
```python
def run_coro_from_sync_thread_with_timeout(coro, timeout: float = 4.0) -> any:
    """Run coroutine with timeout to prevent WebRTC disconnections (same as V4)"""
    global main_event_loop
    if main_event_loop and main_event_loop.is_running():
        import asyncio
        future = asyncio.run_coroutine_threadsafe(
            asyncio.wait_for(coro, timeout=timeout), 
            main_event_loop
        )
        try:
            return future.result(timeout=timeout + 1.0)
        except asyncio.TimeoutError:
            print_status(f"❌ Async task timed out after {timeout}s")
            raise TimeoutError(f"Operation timed out after {timeout}s")
        except Exception as e:
            print_status(f"❌ Error in async task: {e}")
            return "I encountered an error processing your request."
    else:
        print_status("❌ Event loop not available")
        return "My processing system is not ready."
```

### 7. Main Callback Function (V4 Pattern + Refactored Calls)
```python
def voice_assistant_callback_rt(audio_data_tuple: tuple):
    """
    Main callback function using V4 pattern but with refactored components.
    Same signature and flow as V4, but using refactored VoiceAssistant.
    """
    global voice_assistant

    if not voice_assistant:
        yield EMPTY_AUDIO_YIELD_OUTPUT
        return

    try:
        # Same audio processing pattern as V4
        sample_rate, audio_array = voice_assistant.process_audio_array(audio_data_tuple)

        if audio_array.size == 0:
            yield SILENT_AUDIO_FRAME_TUPLE, AdditionalOutputs()
            return

        # Language names (same as V4)
        lang_names = {
            'a': 'American English', 'b': 'British English', 'i': 'Italian', 
            'e': 'Spanish', 'f': 'French', 'p': 'Portuguese', 
            'j': 'Japanese', 'z': 'Chinese', 'h': 'Hindi'
        }

        # STT processing using refactored engine
        user_text = ""
        detected_language = DEFAULT_LANGUAGE
        
        if audio_array.size > 0:
            # Use refactored STT engine
            audio_data_obj = AudioData(
                samples=audio_array,
                sample_rate=sample_rate,
                duration=len(audio_array) / sample_rate
            )
            
            transcription_result = run_coro_from_sync_thread_with_timeout(
                voice_assistant.stt_engine.transcribe(audio_data_obj),
                timeout=4.0
            )
            
            user_text = transcription_result.text.strip()
            print_status(f"📝 Transcribed: '{user_text}'")
            
            # Language detection using refactored detector
            detected_language = voice_assistant.detect_language_from_text(user_text)
            
            # Update current language
            if detected_language != voice_assistant.current_language:
                voice_assistant.current_language = detected_language
                lang_name = lang_names.get(detected_language, 'Unknown')
                print_status(f"🌍 Language switched to: {lang_name} ({detected_language})")

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
        tts_voices_to_try = voice_assistant.get_voices_for_language(voice_assistant.current_language)
        tts_voices_to_try.append(None)
        
        print_status(f"🎤 TTS using language '{voice_assistant.current_language}' with voices: {tts_voices_to_try[:3]}")
        
        tts_success = False
        for voice_id in tts_voices_to_try:
            try:
                # Use refactored TTS engine
                audio_data = run_coro_from_sync_thread_with_timeout(
                    voice_assistant.tts_engine.synthesize_speech(
                        assistant_response_text,
                        voice_id=voice_id,
                        language=voice_assistant.current_language
                    ),
                    timeout=4.0
                )
                
                if audio_data:
                    # Yield audio chunks (same pattern as V4)
                    chunk_size = min(1024, len(audio_data))
                    for i in range(0, len(audio_data), chunk_size):
                        chunk = audio_data[i:i+chunk_size]
                        if len(chunk) > 0:
                            yield (sample_rate, chunk), additional_outputs
                    
                    tts_success = True
                    print_status(f"✅ TTS SUCCESS using voice: {voice_id}")
                    break
                    
            except Exception as e:
                print_status(f"❌ TTS failed with voice '{voice_id}': {e}")
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
```

### 8. Main Execution (Exactly Same as V4)
```python
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
```

## Key Benefits

1. **Same UI/UX**: Users get the exact same FastRTC interface they're used to from V4
2. **Same Simplicity**: No complex Gradio components, just the working V4 pattern
3. **Refactored Backend**: All the benefits of the new modular architecture
4. **Easy Migration**: Direct replacement of V4 with minimal user-facing changes
5. **Proven Pattern**: Uses the working V4 approach that users are satisfied with
6. **Better Error Handling**: Leverages the improved error handling from refactored components
7. **Better Testing**: Can leverage all the unit and integration tests from the refactored system

## Testing Strategy

- **Same test commands** as V4 work
- **Same UI behavior** as V4
- **Same voice commands** as V4
- **Enhanced backend** with better error handling and modularity
- **Can run existing tests** on the refactored components

## Usage

```bash
# Run exactly like V4
cd fastrtc_voice_assistant
python gradio2.py

# Same URL as V4
# Open browser to: http://localhost:7860
```

This implementation provides the **best of both worlds**: the simple, proven V4 interface that users love, powered by the robust, testable, and maintainable refactored architecture.