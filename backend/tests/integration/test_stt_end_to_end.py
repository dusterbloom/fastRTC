import pytest
import soundfile as sf
from pathlib import Path
import numpy as np
from scipy.signal import resample
import difflib

from src.core.voice_assistant import VoiceAssistant
from src.config.settings import load_config

# Mapping of audio files to language codes and canonical phrases
# Updated to match actual audio content based on STT output
LANGUAGE_TEST_DATA = [
    {
        "audio_path": "backend/tests/samples/audio_en.wav",
        "language": "en",
        "canonical_phrases": [
            "The stale smell of old beer lingers. It takes heat to bring out the odor. A cold dip restores health and zest. A salt pickle tastes fine with ham. Tacos al pastor are my favorite. A zestful food is the hot cross bun.",
            "The stale smell of old beer lingers",
            "A cold dip restores health and zest"
        ]
    },
    {
        "audio_path": "backend/tests/samples/audio_it.wav",
        "language": "it",
        "canonical_phrases": [
            "Ciao, mi chiamo Beatrice, ho 23 anni e abito in Italia. Mi piace molto andare in giro a fare compere e studiare.",
            "Ciao, mi chiamo Beatrice",
            "Mi piace molto andare in giro a fare compere e studiare"
        ]
    },
    {
        "audio_path": "backend/tests/samples/audio_es.wav",
        "language": "es",
        "canonical_phrases": [
            "Hola, ¿cómo estás? ¿Cómo te llamas?",
            "Hola, ¿cómo estás?",
            "¿Cómo te llamas?"
        ]
    }
]

def fuzzy_match(transcribed, candidates, threshold=0.8):
    """Return (best_match, similarity) if above threshold, else (None, 0)."""
    best_score = 0
    best_phrase = None
    for phrase in candidates:
        score = difflib.SequenceMatcher(None, transcribed.lower(), phrase.lower()).ratio()
        if score > best_score:
            best_score = score
            best_phrase = phrase
    if best_score >= threshold:
        return best_phrase, best_score
    return None, best_score

@pytest.mark.asyncio
@pytest.mark.parametrize("test_case", LANGUAGE_TEST_DATA)
async def test_real_audio_stt_pipeline_multilang(test_case):
    """
    E2E async test: real speech audio -> VoiceAssistant pipeline -> STT, language detection, TTS.
    Parameterized for multiple languages and canonical phrases.
    """
    audio_path = Path(test_case["audio_path"])
    language = test_case["language"]
    canonical_phrases = test_case["canonical_phrases"]

    # 1. Check for audio file
    assert audio_path.exists(), f"Audio file not found: {audio_path}"

    # 2. Load audio
    audio_samples, sample_rate = sf.read(audio_path, dtype='float32')
    assert audio_samples is not None and len(audio_samples) > 0, f"Audio file {audio_path} is empty"

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

    print(f"\n=== [{language}] STT RAW RESULT ===\n{repr(transcription_result)}\n")
    print(f"=== [{language}] STT TRANSCRIPTION ===\n{transcribed_text}\n")

    # 5. Assert STT output is valid
    assert transcribed_text is not None and transcribed_text.strip() and transcribed_text.strip() != "...", \
        f"STT transcription was invalid: {repr(transcription_result)}"

    # 6. Fuzzy match STT output to canonical phrases
    best_match, similarity = fuzzy_match(transcribed_text, canonical_phrases, threshold=0.8)
    print(f"=== [{language}] FUZZY MATCH ===\nBest match: {best_match}\nSimilarity: {similarity:.2f}\n")
    assert best_match is not None, (
        f"STT output did not match any canonical phrase for {language} (best similarity: {similarity:.2f})"
    )

    # 7. Language detection using MediaPipe
    from src.audio.language.detector import HybridLanguageDetector
    lang_detector = HybridLanguageDetector(confidence_threshold=0.6)
    kokoro_lang, confidence = lang_detector.detect_language(transcribed_text)
    print(f"=== [{language}] LANGUAGE DETECTION ===\nDetected: {kokoro_lang} (confidence: {confidence:.2f})\n")
    
    # Map expected language to Kokoro language codes
    expected_kokoro_mapping = {
        'en': 'a',  # American English
        'it': 'i',  # Italian
        'es': 'e'   # Spanish
    }
    expected_kokoro_lang = expected_kokoro_mapping.get(language, 'a')
    
    assert kokoro_lang == expected_kokoro_lang, f"Detected language '{kokoro_lang}' != expected '{expected_kokoro_lang}' (input lang: {language})"
    assert confidence >= 0.6, f"Language detection confidence too low: {confidence:.2f}"

    # 8. TTS synthesis test
    tts_engine = va.tts_engine
    voices = tts_engine.get_available_voices(kokoro_lang)
    tts_voice = voices[0] if voices else None
    tts_text = best_match  # Use the matched phrase for TTS
    print(f"=== [{language}] TTS SYNTHESIS ===\nUsing voice: {tts_voice}, language: {kokoro_lang}\n")
    audio_data = await tts_engine._synthesize_text(tts_text, tts_voice, kokoro_lang)
    print(f"TTS audio: sr={audio_data.sample_rate}, shape={audio_data.samples.shape}, "
          f"dtype={audio_data.samples.dtype}, min={audio_data.samples.min()}, max={audio_data.samples.max()}\n")
    assert audio_data.samples.size > 0, "TTS produced no audio output"

    # 9. Save TTS output to a WAV file for manual listening  
    tts_wav_path = f"tts_test_output_{language}_detected_{kokoro_lang}_voice_{tts_voice}.wav"
    sf.write(tts_wav_path, audio_data.samples, audio_data.sample_rate)
    print(f"TTS output saved to {tts_wav_path}\n")

    # 10. Clean up async background tasks
    try:
        await va.memory_manager.shutdown()
    except Exception as e:
        print(f"Warning: Error during memory manager shutdown: {e}")
        # Continue with test completion
    
    # Additional cleanup for VoiceAssistant
    try:
        if hasattr(va, 'http_session') and va.http_session:
            await va.http_session.close()
    except Exception as e:
        print(f"Warning: Error closing HTTP session: {e}")