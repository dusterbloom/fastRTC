#!/usr/bin/env python3
"""
Test faster-whisper with real audio file.
"""

import sys
import os
import numpy as np

# Add the backend directory to the path
sys.path.insert(0, '/mnt/c/Users/PC/Dev/fastRTC/backend')

def test_real_audio():
    """Test faster-whisper with real audio file."""
    print("🧪 Testing faster-whisper with real audio...")
    
    audio_file = "/mnt/c/Users/PC/Dev/fastRTC/backend/tests/samples/audio_en.wav"
    
    try:
        import faster_whisper
        import librosa
        
        # Load audio file
        print(f"🎵 Loading audio file: {audio_file}")
        audio_data, sample_rate = librosa.load(audio_file, sr=16000)
        print(f"🎵 Loaded audio: {audio_data.shape}, dtype: {audio_data.dtype}, sr: {sample_rate}")
        print(f"🎵 Audio range: [{np.min(audio_data):.3f}, {np.max(audio_data):.3f}]")
        print(f"🎵 Audio energy: {np.mean(np.abs(audio_data)):.6f}")
        print(f"🎵 Duration: {len(audio_data) / sample_rate:.2f}s")
        
        # Initialize model
        print("🚀 Loading faster-whisper model...")
        model = faster_whisper.WhisperModel(
            "Systran/faster-whisper-large-v3",
            device="cuda",
            compute_type="int8_float16"
        )
        print("✅ Model loaded successfully")
        
        # Test transcription
        print("🎤 Starting transcription...")
        import time
        start_time = time.time()
        
        segments, info = model.transcribe(
            audio_data,
            language=None,
            beam_size=1,
            vad_filter=False,
            temperature=0.0
        )
        
        print("🎤 Processing segments...")
        segments_list = list(segments)
        transcription = " ".join(segment.text for segment in segments_list).strip()
        
        elapsed = time.time() - start_time
        print(f"✅ Transcription completed in {elapsed:.2f}s:")
        print(f"📝 Result: '{transcription}' ({len(segments_list)} segments)")
        
        # Print individual segments
        for i, segment in enumerate(segments_list):
            print(f"   Segment {i}: [{segment.start:.2f}s-{segment.end:.2f}s] '{segment.text}'")
        
        if transcription:
            print("🎉 Real audio transcription successful!")
            return True
        else:
            print("⚠️ Transcription returned empty result")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_real_audio()
    sys.exit(0 if success else 1)