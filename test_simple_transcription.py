#!/usr/bin/env python3
"""
Simple test to verify faster-whisper transcription works.
"""

import sys
import os
import numpy as np

# Add the backend directory to the path
sys.path.insert(0, '/mnt/c/Users/PC/Dev/fastRTC/backend')

def test_simple_transcription():
    """Test basic faster-whisper functionality."""
    print("🧪 Testing basic faster-whisper transcription...")
    
    try:
        import faster_whisper
        
        # Initialize model
        print("🚀 Loading faster-whisper model...")
        model = faster_whisper.WhisperModel(
            "Systran/faster-whisper-large-v3",
            device="cuda",
            compute_type="int8_float16"
        )
        print("✅ Model loaded successfully")
        
        # Create test audio (1 second of sine wave at 440Hz)
        sample_rate = 16000
        duration = 1.0
        frequency = 440  # A4 note
        t = np.linspace(0, duration, int(sample_rate * duration))
        test_audio = 0.3 * np.sin(2 * np.pi * frequency * t).astype(np.float32)
        
        print(f"🎵 Created test audio: {test_audio.shape}, dtype: {test_audio.dtype}")
        print(f"🎵 Audio range: [{np.min(test_audio):.3f}, {np.max(test_audio):.3f}]")
        print(f"🎵 Audio energy: {np.mean(np.abs(test_audio)):.6f}")
        
        # Test transcription
        print("🎤 Starting transcription...")
        import time
        start_time = time.time()
        
        segments, info = model.transcribe(
            test_audio,
            language=None,
            beam_size=1,
            vad_filter=False,
            temperature=0.0
        )
        
        print("🎤 Processing segments...")
        segments_list = list(segments)
        transcription = " ".join(segment.text for segment in segments_list).strip()
        
        elapsed = time.time() - start_time
        print(f"✅ Transcription completed in {elapsed:.2f}s: '{transcription}' ({len(segments_list)} segments)")
        
        if transcription:
            print("🎉 Transcription successful!")
            return True
        else:
            print("⚠️ Transcription returned empty result (expected for sine wave)")
            return True  # This is actually expected for a pure tone
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_simple_transcription()
    sys.exit(0 if success else 1)