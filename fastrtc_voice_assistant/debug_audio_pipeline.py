#!/usr/bin/env python3
"""
Audio Pipeline Debug Test
Tests STT and TTS components to identify issues with Italian recognition and audio output.
"""

import sys
import os
import numpy as np
import asyncio
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, 'src')

# Import components
from src.core.voice_assistant import VoiceAssistant
from src.audio.engines.stt.huggingface_stt import HuggingFaceSTTEngine
from src.audio.engines.tts.kokoro_tts import KokoroTTSEngine
from src.audio.processors.bluetooth_processor import BluetoothAudioProcessor
from src.config.language_config import KOKORO_VOICE_MAP, KOKORO_TTS_LANG_MAP
from fastrtc import KokoroTTSOptions, audio_to_bytes
import logging

# Import for real audio recording
try:
    import sounddevice as sd
    AUDIO_RECORDING_AVAILABLE = True
except ImportError:
    print("⚠️ sounddevice not available. Install with: pip install sounddevice")
    AUDIO_RECORDING_AVAILABLE = False

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_test_audio_italian():
    """Create a synthetic Italian-like audio signal for testing."""
    # Create a simple sine wave that might resemble speech patterns
    sample_rate = 16000
    duration = 2.0  # 2 seconds
    samples = int(sample_rate * duration)
    
    # Create a complex waveform that might resemble Italian speech
    t = np.linspace(0, duration, samples)
    
    # Mix multiple frequencies to simulate speech
    freq1 = 200  # Base frequency
    freq2 = 400  # Harmonic
    freq3 = 800  # Higher harmonic
    
    audio = (0.3 * np.sin(2 * np.pi * freq1 * t) + 
             0.2 * np.sin(2 * np.pi * freq2 * t) + 
             0.1 * np.sin(2 * np.pi * freq3 * t))
    
    # Add some noise to make it more realistic
    noise = np.random.normal(0, 0.05, samples)
    audio = audio + noise
    
    # Normalize
    audio = audio / np.max(np.abs(audio)) * 0.8
    
    return sample_rate, audio.astype(np.float32)

def create_test_audio_english():
    """Create a synthetic English-like audio signal for testing."""
    sample_rate = 16000
    duration = 2.0
    samples = int(sample_rate * duration)
    
    t = np.linspace(0, duration, samples)
    
    # Different frequency pattern for English
    freq1 = 150
    freq2 = 300
    freq3 = 600
    
    audio = (0.4 * np.sin(2 * np.pi * freq1 * t) + 
             0.3 * np.sin(2 * np.pi * freq2 * t) + 
             0.2 * np.sin(2 * np.pi * freq3 * t))
    
    noise = np.random.normal(0, 0.03, samples)
    audio = audio + noise
    audio = audio / np.max(np.abs(audio)) * 0.8
    
    return sample_rate, audio.astype(np.float32)

def record_audio(duration=5, sample_rate=16000):
    """Record audio from microphone."""
    if not AUDIO_RECORDING_AVAILABLE:
        print("❌ Audio recording not available. Install sounddevice: pip install sounddevice")
        return None, None
    
    print(f"🎤 Recording for {duration} seconds... Speak now!")
    try:
        audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype=np.float32)
        sd.wait()  # Wait until recording is finished
        print("✅ Recording completed!")
        return sample_rate, audio_data.flatten()
    except Exception as e:
        print(f"❌ Recording failed: {e}")
        return None, None

async def test_real_speech():
    """Test STT with real speech - Italian and English."""
    print("=" * 60)
    print("🧪 TESTING REAL SPEECH (ITALIAN & ENGLISH)")
    print("=" * 60)
    
    if not AUDIO_RECORDING_AVAILABLE:
        print("❌ Skipping real speech test - sounddevice not available")
        return
    
    try:
        stt_engine = HuggingFaceSTTEngine()
        print(f"✅ STT Engine created: {stt_engine.model_id}")
        
        if not stt_engine.is_available():
            print("❌ STT Engine not available, skipping tests")
            return
        
        # Test 1: Italian speech
        print("\n🔍 Test 1: Italian Speech")
        print("🇮🇹 Please speak in ITALIAN for 5 seconds...")
        print("   Examples: 'Ciao, come stai?', 'Mi chiamo Marco', 'Voglio parlare italiano'")
        input("Press Enter when ready to record Italian...")
        
        sample_rate, italian_audio = record_audio(duration=5)
        if italian_audio is not None:
            print(f"📊 Audio stats: SR={sample_rate}, Samples={len(italian_audio)}, Duration={len(italian_audio)/sample_rate:.2f}s")
            print(f"📊 Audio range: [{np.min(italian_audio):.3f}, {np.max(italian_audio):.3f}]")
            
            result = await stt_engine.transcribe(italian_audio)
            print(f"📝 STT Result: '{result.text}'")
            print(f"🌍 Detected Language: {result.language}")
            print(f"🎯 Confidence: {result.confidence}")
            
            # Check if it contains Italian words
            italian_words = ['ciao', 'come', 'stai', 'sono', 'mi', 'chiamo', 'voglio', 'parlare', 'italiano']
            found_italian = any(word in result.text.lower() for word in italian_words)
            print(f"🇮🇹 Contains Italian words: {found_italian}")
        
        # Test 2: English speech
        print("\n🔍 Test 2: English Speech")
        print("🇺🇸 Please speak in ENGLISH for 5 seconds...")
        print("   Examples: 'Hello, how are you?', 'My name is John', 'I want to speak English'")
        input("Press Enter when ready to record English...")
        
        sample_rate, english_audio = record_audio(duration=5)
        if english_audio is not None:
            print(f"📊 Audio stats: SR={sample_rate}, Samples={len(english_audio)}, Duration={len(english_audio)/sample_rate:.2f}s")
            print(f"📊 Audio range: [{np.min(english_audio):.3f}, {np.max(english_audio):.3f}]")
            
            result = await stt_engine.transcribe(english_audio)
            print(f"📝 STT Result: '{result.text}'")
            print(f"🌍 Detected Language: {result.language}")
            print(f"🎯 Confidence: {result.confidence}")
            
            # Check if it contains English words
            english_words = ['hello', 'how', 'are', 'you', 'my', 'name', 'is', 'want', 'speak', 'english']
            found_english = any(word in result.text.lower() for word in english_words)
            print(f"🇺🇸 Contains English words: {found_english}")
        
        # Test 3: Quick comparison
        print("\n🔍 Test 3: Quick Italian vs English comparison")
        print("🇮🇹 Say something SHORT in Italian (3 seconds)...")
        input("Press Enter when ready...")
        
        sample_rate, quick_italian = record_audio(duration=3)
        if quick_italian is not None:
            result = await stt_engine.transcribe(quick_italian)
            print(f"📝 Italian: '{result.text}' (Language: {result.language})")
        
        print("🇺🇸 Say something SHORT in English (3 seconds)...")
        input("Press Enter when ready...")
        
        sample_rate, quick_english = record_audio(duration=3)
        if quick_english is not None:
            result = await stt_engine.transcribe(quick_english)
            print(f"📝 English: '{result.text}' (Language: {result.language})")
        
    except Exception as e:
        print(f"❌ Real speech test failed: {e}")
        import traceback
        traceback.print_exc()

async def test_language_detection_improvements():
    """Test the improved language detection specifically."""
    print("=" * 60)
    print("🧪 TESTING IMPROVED LANGUAGE DETECTION")
    print("=" * 60)
    
    try:
        stt_engine = HuggingFaceSTTEngine()
        print(f"✅ STT Engine created: {stt_engine.model_id}")
        
        if not stt_engine.is_available():
            print("❌ STT Engine not available, skipping tests")
            return
        
        # Test with known Italian and English phrases
        test_phrases = [
            ("Ciao, come stai? Mi chiamo Marco.", "Italian"),
            ("Hello, how are you? My name is John.", "English"),
            ("Voglio parlare italiano con te.", "Italian"),
            ("I want to speak English with you.", "English"),
            ("Buongiorno, che bella giornata!", "Italian"),
            ("Good morning, what a beautiful day!", "English")
        ]
        
        print("\n🔍 Testing language detection with known phrases:")
        for phrase, expected_lang in test_phrases:
            print(f"\n📝 Testing: '{phrase}' (Expected: {expected_lang})")
            
            # Create a simple audio signal (we're mainly testing the text-based fallback)
            sample_rate = 16000
            duration = 2.0
            samples = int(sample_rate * duration)
            # Simple sine wave
            t = np.linspace(0, duration, samples)
            audio = 0.1 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
            
            # We'll manually test the text-based detection by creating a mock result
            # This simulates what happens when Whisper doesn't return language info
            print(f"🔤 Text-based language detection test:")
            
            # Test the text-based detection logic directly
            text_lower = phrase.lower()
            detected_lang = None
            
            if any(word in text_lower for word in ['ciao', 'come', 'stai', 'sono', 'mi', 'chiamo', 'voglio', 'parlare', 'italiano', 'buongiorno', 'che', 'bella', 'giornata']):
                detected_lang = 'it'
            elif any(word in text_lower for word in ['hello', 'how', 'are', 'you', 'my', 'name', 'is', 'want', 'speak', 'english', 'good', 'morning', 'what', 'beautiful', 'day']):
                detected_lang = 'en'
            
            print(f"🎯 Detected language: {detected_lang}")
            print(f"✅ Correct: {(detected_lang == 'it' and expected_lang == 'Italian') or (detected_lang == 'en' and expected_lang == 'English')}")
        
        print(f"\n🔍 Testing with real speech recording:")
        if AUDIO_RECORDING_AVAILABLE:
            print("🇮🇹 Say 'Ciao come stai' in Italian (3 seconds)...")
            input("Press Enter when ready...")
            
            sample_rate, audio = record_audio(duration=3)
            if audio is not None:
                result = await stt_engine.transcribe(audio)
                print(f"📝 Transcribed: '{result.text}'")
                print(f"🌍 Detected Language: {result.language}")
                print(f"🔤 Text contains Italian words: {any(word in result.text.lower() for word in ['ciao', 'come', 'stai'])}")
        else:
            print("❌ Audio recording not available for real speech test")
        
    except Exception as e:
        print(f"❌ Language detection test failed: {e}")
        import traceback
        traceback.print_exc()

async def test_stt_engine():
    """Test the STT engine with different audio inputs."""
    print("=" * 60)
    print("🧪 TESTING STT ENGINE")
    print("=" * 60)
    
    try:
        stt_engine = HuggingFaceSTTEngine()
        print(f"✅ STT Engine created: {stt_engine.model_id}")
        print(f"✅ STT Engine available: {stt_engine.is_available()}")
        
        if not stt_engine.is_available():
            print("❌ STT Engine not available, skipping tests")
            return
        
        # Test 1: Italian-like audio
        print("\n🔍 Test 1: Italian-like synthetic audio")
        sample_rate, italian_audio = create_test_audio_italian()
        print(f"📊 Audio stats: SR={sample_rate}, Samples={len(italian_audio)}, Duration={len(italian_audio)/sample_rate:.2f}s")
        print(f"📊 Audio range: [{np.min(italian_audio):.3f}, {np.max(italian_audio):.3f}]")
        
        # Test with audio_to_bytes conversion
        try:
            audio_bytes = audio_to_bytes((sample_rate, italian_audio))
            print(f"✅ audio_to_bytes conversion successful: {type(audio_bytes)}")
        except Exception as e:
            print(f"❌ audio_to_bytes conversion failed: {e}")
            audio_bytes = italian_audio
        
        result = await stt_engine.transcribe(italian_audio)
        print(f"📝 STT Result: '{result.text}'")
        print(f"🌍 Detected Language: {result.language}")
        print(f"🎯 Confidence: {result.confidence}")
        
        # Test 2: English-like audio
        print("\n🔍 Test 2: English-like synthetic audio")
        sample_rate, english_audio = create_test_audio_english()
        result = await stt_engine.transcribe(english_audio)
        print(f"📝 STT Result: '{result.text}'")
        print(f"🌍 Detected Language: {result.language}")
        
        # Test 3: Real Italian text (if we can synthesize it first)
        print("\n🔍 Test 3: Testing with known Italian text")
        italian_text = "Ciao, come stai oggi?"
        print(f"🎯 Target text: '{italian_text}'")
        
        # We'll test this after TTS tests
        
    except Exception as e:
        print(f"❌ STT Engine test failed: {e}")
        import traceback
        traceback.print_exc()

async def test_tts_engine():
    """Test the TTS engine with different languages and voices."""
    print("\n" + "=" * 60)
    print("🧪 TESTING TTS ENGINE")
    print("=" * 60)
    
    try:
        tts_engine = KokoroTTSEngine()
        print(f"✅ TTS Engine created")
        print(f"✅ TTS Engine available: {tts_engine.is_available()}")
        
        if not tts_engine.is_available():
            print("❌ TTS Engine not available, skipping tests")
            return None, None
        
        # Test 1: English synthesis
        print("\n🔍 Test 1: English TTS")
        english_text = "Hello, this is a test in English."
        english_voice = "af_heart"
        
        try:
            result = await tts_engine.synthesize(english_text, voice=english_voice, language="a")
            print(f"✅ English TTS successful: {len(result.samples)} samples, {result.duration:.2f}s")
            print(f"📊 Audio stats: SR={result.sample_rate}, Range=[{np.min(result.samples):.3f}, {np.max(result.samples):.3f}]")
            english_audio = result.samples
        except Exception as e:
            print(f"❌ English TTS failed: {e}")
            english_audio = None
        
        # Test 2: Italian synthesis
        print("\n🔍 Test 2: Italian TTS")
        italian_text = "Ciao, come stai oggi?"
        italian_voice = "if_sara"
        
        try:
            result = await tts_engine.synthesize(italian_text, voice=italian_voice, language="i")
            print(f"✅ Italian TTS successful: {len(result.samples)} samples, {result.duration:.2f}s")
            print(f"📊 Audio stats: SR={result.sample_rate}, Range=[{np.min(result.samples):.3f}, {np.max(result.samples):.3f}]")
            italian_audio = result.samples
        except Exception as e:
            print(f"❌ Italian TTS failed: {e}")
            italian_audio = None
        
        # Test 3: Direct streaming like V4
        print("\n🔍 Test 3: Direct streaming (V4 style)")
        try:
            options = KokoroTTSOptions(speed=1.05, lang="it", voice="if_sara")
            chunk_count = 0
            total_samples = 0
            
            for tts_output_item in tts_engine.tts_model.stream_tts_sync(italian_text, options):
                if isinstance(tts_output_item, tuple) and len(tts_output_item) == 2:
                    sr, chunk = tts_output_item
                    if isinstance(chunk, np.ndarray) and chunk.size > 0:
                        chunk_count += 1
                        total_samples += chunk.size
                        print(f"  📦 Chunk {chunk_count}: {chunk.size} samples, SR={sr}")
                        if chunk_count >= 3:  # Limit output
                            print(f"  ... (stopping after 3 chunks)")
                            break
            
            print(f"✅ Direct streaming successful: {chunk_count} chunks, {total_samples} total samples")
            
        except Exception as e:
            print(f"❌ Direct streaming failed: {e}")
            import traceback
            traceback.print_exc()
        
        return english_audio, italian_audio
        
    except Exception as e:
        print(f"❌ TTS Engine test failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None

async def test_round_trip():
    """Test TTS -> STT round trip to see if audio is preserved correctly."""
    print("\n" + "=" * 60)
    print("🧪 TESTING ROUND TRIP (TTS -> STT)")
    print("=" * 60)
    
    try:
        stt_engine = HuggingFaceSTTEngine()
        tts_engine = KokoroTTSEngine()
        
        if not (stt_engine.is_available() and tts_engine.is_available()):
            print("❌ Engines not available for round trip test")
            return
        
        # Test 1: English round trip
        print("\n🔍 Test 1: English round trip")
        original_text = "Hello world, this is a test."
        print(f"🎯 Original: '{original_text}'")
        
        # TTS
        tts_result = await tts_engine.synthesize(original_text, voice="af_heart", language="a")
        print(f"🔊 TTS: {len(tts_result.samples)} samples generated")
        
        # STT
        stt_result = await stt_engine.transcribe(tts_result.samples)
        print(f"📝 STT: '{stt_result.text}'")
        print(f"🌍 Language: {stt_result.language}")
        print(f"✅ Match: {original_text.lower() in stt_result.text.lower()}")
        
        # Test 2: Italian round trip
        print("\n🔍 Test 2: Italian round trip")
        original_text = "Ciao, come stai?"
        print(f"🎯 Original: '{original_text}'")
        
        # TTS
        tts_result = await tts_engine.synthesize(original_text, voice="if_sara", language="i")
        print(f"🔊 TTS: {len(tts_result.samples)} samples generated")
        
        # STT
        stt_result = await stt_engine.transcribe(tts_result.samples)
        print(f"📝 STT: '{stt_result.text}'")
        print(f"🌍 Language: {stt_result.language}")
        print(f"✅ Contains Italian: {'ciao' in stt_result.text.lower() or 'come' in stt_result.text.lower()}")
        
    except Exception as e:
        print(f"❌ Round trip test failed: {e}")
        import traceback
        traceback.print_exc()

async def test_voice_assistant_integration():
    """Test the full VoiceAssistant integration."""
    print("\n" + "=" * 60)
    print("🧪 TESTING VOICE ASSISTANT INTEGRATION")
    print("=" * 60)
    
    try:
        voice_assistant = VoiceAssistant()
        await voice_assistant.initialize_async()
        print("✅ VoiceAssistant initialized")
        
        # Test audio processing
        print("\n🔍 Testing audio processing pipeline")
        sample_rate, test_audio = create_test_audio_italian()
        
        # Process through audio processor
        processed_sr, processed_audio = voice_assistant.audio_processor.process(test_audio)
        print(f"📊 Audio processing: {len(test_audio)} -> {len(processed_audio)} samples")
        
        # Test STT
        print("\n🔍 Testing STT through VoiceAssistant")
        from src.core.interfaces import AudioData
        audio_data = AudioData(
            samples=processed_audio,
            sample_rate=processed_sr,
            duration=len(processed_audio) / processed_sr
        )
        
        stt_result = await voice_assistant.stt_engine.transcribe(audio_data)
        print(f"📝 STT Result: '{stt_result.text}'")
        print(f"🌍 Language: {stt_result.language}")
        
        # Test language detection
        print("\n🔍 Testing language detection")
        detected_lang = voice_assistant.language_detector.detect_language(stt_result.text)
        print(f"🔍 Detected language: {detected_lang}")
        
        # Test voice mapping
        voices = voice_assistant.voice_mapper.get_voices_for_language(detected_lang)
        print(f"🎤 Available voices: {voices}")
        
        await voice_assistant.cleanup_async()
        
    except Exception as e:
        print(f"❌ VoiceAssistant integration test failed: {e}")
        import traceback
        traceback.print_exc()

async def main():
    """Run all audio pipeline tests."""
    print("🚀 STARTING AUDIO PIPELINE DEBUG TESTS")
    print("=" * 80)
    
    # Start with real speech test - most important!
    await test_real_speech()
    
    # Test the improved language detection
    await test_language_detection_improvements()
    
    # Then test other components
    await test_stt_engine()
    english_audio, italian_audio = await test_tts_engine()
    await test_round_trip()
    await test_voice_assistant_integration()
    
    print("\n" + "=" * 80)
    print("🏁 AUDIO PIPELINE DEBUG TESTS COMPLETED")
    print("=" * 80)
    
    # Summary
    print("\n📋 SUMMARY:")
    print("- Check STT transcription results for accuracy")
    print("- Check TTS audio generation for quality")
    print("- Check round-trip preservation")
    print("- Check language detection accuracy")
    print("- Look for any error patterns in the logs above")

if __name__ == "__main__":
    asyncio.run(main())