#!/usr/bin/env python3
"""
Simple test to debug threading pipeline startup issues.
"""

import sys
import os
import time
import numpy as np

# Add the backend directory to the Python path
sys.path.insert(0, os.path.dirname(__file__))

from src.integration.threading_callback_handler import ThreadingCallbackHandler
from src.audio import STTEngine, KokoroTTSEngine, VoiceMapper
from src.core.voice_assistant import VoiceAssistant

print("🔧 Threading Debug Test")
print("=" * 50)

def create_test_audio():
    """Create simple test audio."""
    duration = 2.0  # seconds
    sample_rate = 16000
    samples = int(duration * sample_rate)
    
    # Create a simple sine wave
    t = np.linspace(0, duration, samples, False)
    frequency = 440  # A4 note
    audio = 0.3 * np.sin(2 * np.pi * frequency * t).astype(np.float32)
    
    return (sample_rate, audio)

def test_threading_startup():
    """Test threading pipeline startup."""
    try:
        print("🚀 Testing threading pipeline startup...")
        
        # Initialize components
        print("🔍 Initializing STT engine...")
        stt_engine = STTEngine()
        print("✅ STT engine initialized")
        
        print("🔍 Initializing TTS engine...")
        tts_engine = KokoroTTSEngine()
        print("✅ TTS engine initialized")
        
        print("🔍 Initializing voice mapper...")
        voice_mapper = VoiceMapper()
        print("✅ Voice mapper initialized")
        
        print("🔍 Initializing voice assistant...")
        voice_assistant = VoiceAssistant()
        print("✅ Voice assistant initialized")
        
        print("🔍 Creating threading handler...")
        handler = ThreadingCallbackHandler(
            voice_assistant=voice_assistant,
            stt_engine=stt_engine,
            tts_engine=tts_engine,
            voice_mapper=voice_mapper
        )
        print("✅ Threading handler created")
        
        print("🔍 Starting threading handler...")
        handler.start()
        print("✅ Threading handler started")
        
        # Wait a bit to ensure everything is running
        time.sleep(2)
        
        print("🔍 Testing audio processing...")
        audio_data = create_test_audio()
        
        # Process one audio chunk
        results = list(handler.process_audio_stream(audio_data))
        print(f"✅ Got {len(results)} results")
        
        print("🔍 Stopping threading handler...")
        handler.stop()
        print("✅ Threading handler stopped")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in threading startup test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_threading_startup()
    if success:
        print("✅ Threading startup test completed successfully")
    else:
        print("❌ Threading startup test failed")
        sys.exit(1)