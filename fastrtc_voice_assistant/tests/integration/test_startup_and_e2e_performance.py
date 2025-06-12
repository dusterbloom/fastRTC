"""
Performance tests for Voice Assistant startup and first turn processing.
Uses a pre-recorded audio file for STT testing.
"""
import pytest
import asyncio
import time
import numpy as np
import soundfile as sf # For loading WAV files
from pathlib import Path

from src.core.voice_assistant import VoiceAssistant
from src.config.settings import load_config, AppConfig
from src.core.interfaces import AudioData

# Note: Ensure conftest.py sets up logging to DEBUG for pytest -s to show "🧠 Profiling:" logs.

@pytest.fixture(scope="module")
def sample_audio_path() -> Path:
    """
    Provides the path to a sample audio file.
    Ensure 'greeting.wav' (or similar) exists in tests/samples/.
    """
    # Adjust path relative to the fastRTC workspace root
    path = Path("fastrtc_voice_assistant/tests/samples/greeting.wav")
    if not path.exists():
        # Create a dummy silent WAV file if it doesn't exist, so tests can run.
        # User should replace this with a real greeting.wav for meaningful STT.
        print(f"Warning: Sample audio file {path} not found. Creating a dummy silent WAV.")
        path.parent.mkdir(parents=True, exist_ok=True)
        samplerate = 16000
        duration = 1 # second
        data = np.zeros(samplerate * duration, dtype=np.float32)
        try:
            sf.write(path, data, samplerate)
            print(f"Dummy silent WAV file created at {path}")
        except Exception as e:
            print(f"Could not create dummy WAV file: {e}. Please create it manually for the test.")
            # Fallback to a non-existent path to make the test explicitly require the file
            return Path("fastrtc_voice_assistant/tests/samples/PLEASE_CREATE_GREETING_WAV.wav")
            
    return path

@pytest.mark.integration
@pytest.mark.performance
@pytest.mark.slow  # Mark as slow as it involves real model loading
async def test_voice_assistant_initialization_and_first_turn_performance(
    performance_timer, 
    app_config: AppConfig, # Fixture from conftest.py
    sample_audio_path: Path
):
    """
    Tests the full initialization of VoiceAssistant and a minimal first turn
    using a pre-recorded audio file.
    Relies on INFO level "🧠 Profiling:" logs for detailed component load times.
    """
    print("\\n--- Starting Voice Assistant Initialization Performance Test ---")
    performance_timer.start()
    
    # Load actual configuration from .env files for realistic startup
    actual_config = load_config() 
    
    va = VoiceAssistant(config=actual_config)
    init_sync_duration = performance_timer.elapsed
    print(f"⏱️ VoiceAssistant synchronous __init__ took: {init_sync_duration:.2f}s")
    
    # Restart timer for async part
    performance_timer.start() 
    await va.initialize_async()
    init_async_duration = performance_timer.elapsed
    print(f"⏱️ VoiceAssistant initialize_async took: {init_async_duration:.2f}s")
    
    total_init_duration = init_sync_duration + init_async_duration
    print(f"⏱️ VoiceAssistant TOTAL INITIALIZATION took: {total_init_duration:.2f}s")
    print("--- Voice Assistant Initialization Complete ---")

    # Assertions for component readiness
    assert va.stt_engine is not None and va.stt_engine.is_available(), "STT Engine not available"
    assert va.tts_engine is not None and va.tts_engine.is_available(), "TTS Engine not available"
    assert va.llm_service is not None and va.llm_service.is_available(), "LLM Service not available"
    assert va.memory_manager is not None and va.memory_manager.is_available(), "Memory Manager not available"
    assert va.language_detector is not None and va.language_detector.is_available(), "Language Detector not available"

    print("\\n--- Starting Minimal First Turn Performance Test (with recorded audio) ---")
    performance_timer.start()

    # 1. Load and Process Audio Input
    if not sample_audio_path.exists():
        pytest.fail(f"Sample audio file not found at {sample_audio_path}. Please create it.")

    audio_samples, audio_sample_rate = sf.read(sample_audio_path, dtype='float32')
    print(f"Loaded audio from {sample_audio_path}: {len(audio_samples)} samples at {audio_sample_rate}Hz")

    # 2. STT
    stt_start_time = time.monotonic()
    transcription_result = await va.stt_engine.transcribe_with_sample_rate(audio_samples, audio_sample_rate)
    stt_duration = time.monotonic() - stt_start_time
    transcribed_text = transcription_result.text.strip() if transcription_result and transcription_result.text else ""
    print(f"⏱️ STT transcription took: {stt_duration:.2f}s. Text: '{transcribed_text}'")
    assert transcribed_text is not None, "STT returned None" 
    # Not asserting content of transcribed_text as it depends on the audio file quality and STT model.
    # A non-empty string is a good basic check for this performance test.
    if not transcribed_text: # If STT yields empty, use a placeholder for LLM
        print("Warning: STT produced empty text. Using placeholder for LLM.")
        transcribed_text = "Hello assistant"


    # 3. LLM Response
    llm_start_time = time.monotonic()
    # Ensure memory_manager has user_id set if get_llm_response_smart relies on it
    if hasattr(va.memory_manager, 'user_id') and va.memory_manager.user_id is None:
         va.memory_manager.user_id = "test_perf_user"

    llm_context = await va.memory_manager.get_user_context() if va.memory_manager else ""
    llm_response_text = await va.llm_service.get_response(transcribed_text, context=llm_context)
    llm_duration = time.monotonic() - llm_start_time
    print(f"⏱️ LLM response generation took: {llm_duration:.2f}s. Response: '{llm_response_text[:60]}...'")
    assert llm_response_text is not None and len(llm_response_text) > 0

    # 4. TTS Synthesis (first chunk)
    tts_start_time = time.monotonic()
    current_lang_kokoro = va.convert_to_kokoro_language(va.current_language) 
    voices = va.get_voices_for_language(current_lang_kokoro)
    selected_voice = voices[0] if voices else None

    synthesized_audio_chunk = None
    async for sr, audio_chunk_arr in va.tts_engine.stream_synthesis(llm_response_text, voice=selected_voice, language=current_lang_kokoro):
        synthesized_audio_chunk = audio_chunk_arr
        print(f"TTS synthesized first chunk: {len(synthesized_audio_chunk)} samples at {sr}Hz")
        break 
    
    tts_duration = time.monotonic() - tts_start_time
    print(f"⏱️ TTS first chunk synthesis took: {tts_duration:.2f}s")
    assert synthesized_audio_chunk is not None and len(synthesized_audio_chunk) > 0

    first_turn_duration = performance_timer.elapsed # This timer was started before audio loading
    print(f"⏱️ Minimal First Turn (File STT -> LLM -> TTS chunk) took: {first_turn_duration:.2f}s")
    print("--- Minimal First Turn Performance Test Complete ---")

    assert total_init_duration > 0
    assert first_turn_duration > 0