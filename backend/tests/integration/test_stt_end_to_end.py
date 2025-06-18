import pytest
import soundfile as sf
from pathlib import Path
import numpy as np
from scipy.signal import resample

from src.core.voice_assistant import VoiceAssistant
from src.config.settings import load_config

@pytest.mark.asyncio
async def test_real_audio_stt_pipeline():
    """
    End-to-end async test: real speech audio -> VoiceAssistant pipeline -> STT output.
    Fails if STT returns None, '...', or empty string.
    """
    # 1. Check for audio file
    audio_path = Path("backend/tests/samples/greeting.wav")
    assert audio_path.exists(), (
        "Please provide a real speech WAV file at tests/samples/greeting.wav"
    )

    # 2. Load audio
    audio_samples, sample_rate = sf.read(audio_path, dtype='float32')
    assert audio_samples is not None and len(audio_samples) > 0, "Audio file is empty"

    # Convert stereo to mono if needed
    if len(audio_samples.shape) == 2:
        audio_samples = audio_samples.mean(axis=1)

    # Resample to 16000 Hz if needed
    target_sr = 16000
    if sample_rate != target_sr:
        num_samples = int(len(audio_samples) * target_sr / sample_rate)
        audio_samples = resample(audio_samples, num_samples)
        sample_rate = target_sr

    # 3. Instantiate real components and initialize async
    config = load_config()
    va = VoiceAssistant(config=config)  # Use loaded config, real STT engine
    await va.initialize_async()

    # 4. Run STT asynchronously
    transcription_result = await va.stt_engine.transcribe_with_sample_rate(audio_samples, sample_rate)
    transcribed_text = getattr(transcription_result, "text", None)

    import logging
    logging.info("STT RAW RESULT: %r", transcription_result)
    logging.info("STT TRANSCRIPTION: %r", transcribed_text)

    # Always print the transcription, even if the test passes
    print("\n================= STT RAW RESULT =================", flush=True)
    print(repr(transcription_result), flush=True)
    print("================= STT TRANSCRIPTION =================", flush=True)
    print(transcribed_text, flush=True)
    print("====================================================\n", flush=True)

    # 5. Assert STT output is valid, and always show the value if it fails
    if transcribed_text is None or transcribed_text.strip() == "" or transcribed_text.strip() == "...":
        import pytest
        pytest.fail(f"STT transcription was invalid: {repr(transcription_result)}")

    # Language detection using MediaPipe
    from src.audio.language.detector import MediaPipeLanguageDetector
    lang_detector = MediaPipeLanguageDetector()
    kokoro_lang, confidence = lang_detector.detect_language(transcribed_text)
    print("================= MEDIAPIPE LANGUAGE DETECTION =================", flush=True)
    print(f"Detected language: {kokoro_lang} (confidence: {confidence})", flush=True)
    print("===============================================================", flush=True)

    # TTS synthesis test
    tts_engine = va.tts_engine
    voices = tts_engine.get_available_voices(kokoro_lang)
    tts_voice = voices[0] if voices else None
    tts_text = "This is a test of the TTS system. The quick brown fox jumps over the lazy dog."
    print("================= TTS SYNTHESIS =================", flush=True)
    print(f"Using voice: {tts_voice}, language: {kokoro_lang}", flush=True)
    # Use the async TTS API to avoid event loop errors
    audio_data = await tts_engine._synthesize_text(tts_text, tts_voice, kokoro_lang)
    print(f"TTS audio: sr={audio_data.sample_rate}, shape={audio_data.samples.shape}, dtype={audio_data.samples.dtype}, min={audio_data.samples.min()}, max={audio_data.samples.max()}", flush=True)
    assert audio_data.samples.size > 0, "TTS produced no audio output"
    print("==================================================", flush=True)

    # Save TTS output to a WAV file for manual listening
    tts_wav_path = "tts_test_output.wav"
    sf.write(tts_wav_path, audio_data.samples, audio_data.sample_rate)
    print(f"TTS output saved to {tts_wav_path}", flush=True)

    # Clean up async background tasks to avoid "no running event loop" errors
    await va.memory_manager.shutdown()

    # Test passes if STT, language detection, and TTS are valid
    expected_phrase = "The stale smell of old beer lingers"
    assert expected_phrase.lower() in transcribed_text.lower(), (
        f"Expected phrase not found in transcription: {repr(transcription_result)}"
    )
@pytest.mark.asyncio
async def test_stt_llm_tts_streaming_pipeline():
    """
    End-to-end async test: real speech audio -> STT -> LLM -> streaming TTS (as in production).
    Streams TTS output, concatenates chunks, and writes to WAV for listening.
    Uses a thread executor for streaming TTS to avoid event loop conflicts.
    """
    import asyncio

    # 1. Check for audio file
    audio_path = Path("backend/tests/samples/greeting.wav")
    assert audio_path.exists(), (
        "Please provide a real speech WAV file at tests/samples/greeting.wav"
    )

    # 2. Load audio
    audio_samples, sample_rate = sf.read(audio_path, dtype='float32')
    assert audio_samples is not None and len(audio_samples) > 0, "Audio file is empty"

    # Convert stereo to mono if needed
    if len(audio_samples.shape) == 2:
        audio_samples = audio_samples.mean(axis=1)

    # Resample to 16000 Hz if needed
    target_sr = 16000
    if sample_rate != target_sr:
        num_samples = int(len(audio_samples) * target_sr / sample_rate)
        audio_samples = resample(audio_samples, num_samples)
        sample_rate = target_sr

    # 3. Instantiate real components and initialize async
    config = load_config()
    va = VoiceAssistant(config=config)
    await va.initialize_async()

    # 4. Run STT asynchronously
    transcription_result = await va.stt_engine.transcribe_with_sample_rate(audio_samples, sample_rate)
    transcribed_text = getattr(transcription_result, "text", None)

    print("\n================= STT RAW RESULT =================", flush=True)
    print(repr(transcription_result), flush=True)
    print("================= STT TRANSCRIPTION =================", flush=True)
    print(transcribed_text, flush=True)
    print("====================================================\n", flush=True)

    # 5. Assert STT output is valid
    if transcribed_text is None or transcribed_text.strip() == "" or transcribed_text.strip() == "...":
        import pytest
        pytest.fail(f"STT transcription was invalid: {repr(transcription_result)}")

    # 6. Run LLM on STT output
    llm_response = await va.get_llm_response_smart(transcribed_text)
    print("================= LLM RESPONSE =================", flush=True)
    print(llm_response, flush=True)
    print("================================================\n", flush=True)
    assert llm_response is not None and llm_response.strip() != "", "LLM response was empty"

    # 7. Streaming TTS on LLM output (run in executor to avoid event loop conflicts)
    from src.audio.language.detector import MediaPipeLanguageDetector
    lang_detector = MediaPipeLanguageDetector()
    kokoro_lang, confidence = lang_detector.detect_language(llm_response)
    print("================= MEDIAPIPE LANGUAGE DETECTION (LLM) =================", flush=True)
    print(f"Detected language: {kokoro_lang} (confidence: {confidence})", flush=True)
    print("======================================================================", flush=True)

    tts_engine = va.tts_engine
    voices = tts_engine.get_available_voices(kokoro_lang)
    tts_voice = voices[0] if voices else None

    print("================= STREAMING TTS SYNTHESIS =================", flush=True)
    print(f"Using voice: {tts_voice}, language: {kokoro_lang}", flush=True)

    def run_streaming_tts():
        streamed_chunks = []
        sample_rate = None
        total_samples = 0
        for sr, chunk in va.stream_tts_synthesis(llm_response, tts_voice, kokoro_lang):
            if chunk is not None and hasattr(chunk, "size") and chunk.size > 0:
                streamed_chunks.append(chunk)
                total_samples += chunk.size
                sample_rate = sr
                print(f"Streamed chunk: shape={chunk.shape}, dtype={chunk.dtype}, min={chunk.min()}, max={chunk.max()}", flush=True)
        return streamed_chunks, sample_rate, total_samples

    loop = asyncio.get_event_loop()
    streamed_chunks, sample_rate, total_samples = await loop.run_in_executor(None, run_streaming_tts)

    print(f"Total streamed chunks: {len(streamed_chunks)}, total samples: {total_samples}", flush=True)
    assert len(streamed_chunks) > 0, "No audio chunks were streamed from TTS"
    assert total_samples > 0, "Streamed TTS produced no audio samples"
    print("==========================================================\n", flush=True)

    # Concatenate all chunks and write to WAV for listening
    if sample_rate is not None and total_samples > 0:
        full_audio = np.concatenate(streamed_chunks)
        wav_path = "stt_llm_tts_streamed_output.wav"
        sf.write(wav_path, full_audio, sample_rate)
        print(f"Streamed TTS output saved to {wav_path}", flush=True)

    # Clean up async background tasks
    await va.memory_manager.shutdown()