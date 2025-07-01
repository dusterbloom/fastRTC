#!/usr/bin/env python3
"""
Test WhisperLive using the official TranscriptionClient
This is the correct way to use WhisperLive for audio file transcription
"""

import sys
import os
sys.path.append('/mnt/c/Users/PC/Dev/fastRTC/backend/venv/lib/python3.10/site-packages')

try:
    from whisper_live.client import TranscriptionClient
    print("✅ WhisperLive client imported successfully")
except ImportError as e:
    print(f"❌ Failed to import WhisperLive client: {e}")
    sys.exit(1)

def test_whisperlive_proper():
    """Test WhisperLive using the official client method"""
    
    # Create client with proper configuration
    client = TranscriptionClient(
        "localhost", 
        9090, 
        lang="en", 
        translate=False, 
        model="small",
        use_vad=False,  # Disable VAD since FastRTC handles it
        save_output_recording=True,
        output_recording_filename="./whisperlive_test_output.wav"
    )
    
    print("🎤 WhisperLive client created successfully")
    
    # Test with audio file
    audio_file = 'tests/samples/audio_en.wav'
    
    if not os.path.exists(audio_file):
        print(f"❌ Audio file not found: {audio_file}")
        return
    
    print(f"🎵 Testing transcription of: {audio_file}")
    
    try:
        # Use the official client method for audio file transcription
        result = client(audio_file)
        print(f"✅ Transcription completed!")
        print(f"📝 Result: {result}")
        
    except Exception as e:
        print(f"❌ Transcription failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_whisperlive_proper()