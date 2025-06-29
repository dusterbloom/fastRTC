#!/usr/bin/env python3
"""
Test audio transcription with VAD disabled
"""

import os
import sys
import numpy as np
import asyncio
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent / "backend"))

# Set environment for GPU STT
os.environ['STT_BACKEND'] = 'faster_gpu'
os.environ['LOG_LEVEL'] = 'INFO'

from backend.src.utils.logging import get_logger
from backend.src.audio.engines.stt.faster_whisper_gpu_stt import FasterWhisperGPUSTT
from backend.src.core.interfaces import AudioData

logger = get_logger(__name__)

def load_wav_file(file_path):
    """Load WAV file using scipy."""
    try:
        from scipy.io import wavfile
        sr, audio = wavfile.read(file_path)
        
        # Handle stereo
        if audio.ndim > 1:
            audio = audio[:, 0]
        
        # Convert to float32
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0
        elif audio.dtype == np.int32:
            audio = audio.astype(np.float32) / 2147483648.0
        else:
            audio = audio.astype(np.float32)
            
        return audio, sr
    except Exception as e:
        print(f"❌ Failed to load audio: {e}")
        return None, None

async def test_no_vad():
    """Test STT with VAD disabled."""
    audio_file = "/mnt/c/Users/PC/Dev/fastRTC/backend/tests/samples/audio_en.wav"
    
    print(f"🧪 Testing STT with VAD disabled: {audio_file}")
    
    try:
        # Load audio file
        audio_samples, sample_rate = load_wav_file(audio_file)
        if audio_samples is None:
            return False
        
        duration = len(audio_samples) / sample_rate
        print(f"🎵 Audio duration: {duration:.2f}s")
        
        # Create STT engine with VAD disabled
        print("📝 Creating STT engine with VAD disabled...")
        stt = FasterWhisperGPUSTT()
        
        # Disable VAD filter
        stt.vad_filter = False
        print(f"🔧 VAD filter disabled: {stt.vad_filter}")
        
        if not stt.is_available():
            print("❌ STT engine not available")
            return False
        
        # Test full audio with VAD disabled
        print(f"🚀 Transcribing full audio with VAD disabled...")
        audio_data = AudioData(samples=audio_samples, sample_rate=sample_rate, duration=duration)
        
        start_time = asyncio.get_event_loop().time()
        result = await stt._transcribe_audio(audio_data)
        end_time = asyncio.get_event_loop().time()
        
        processing_time = end_time - start_time
        print(f"✅ VAD disabled result: '{result.text}' (confidence: {result.confidence:.3f}, time: {processing_time:.3f}s)")
        
        # Test with VAD enabled for comparison
        print(f"🚀 Transcribing full audio with VAD enabled...")
        stt.vad_filter = True
        
        start_time = asyncio.get_event_loop().time()
        result_vad = await stt._transcribe_audio(audio_data)
        end_time = asyncio.get_event_loop().time()
        
        processing_time_vad = end_time - start_time
        print(f"✅ VAD enabled result: '{result_vad.text}' (confidence: {result_vad.confidence:.3f}, time: {processing_time_vad:.3f}s)")
        
        # Compare results
        print(f"\n📊 Comparison:")
        print(f"   VAD disabled: '{result.text}' ({len(result.text)} chars)")
        print(f"   VAD enabled:  '{result_vad.text}' ({len(result_vad.text)} chars)")
        
        if len(result.text) > len(result_vad.text):
            print("🔍 VAD disabled produced longer transcription!")
        elif len(result.text) < len(result_vad.text):
            print("🔍 VAD enabled produced longer transcription!")
        else:
            print("🔍 Both produced same length transcription")
        
        # Cleanup
        stt.shutdown()
        print("🧹 STT engine shutdown complete")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        print(f"❌ Traceback: {traceback.format_exc()}")
        return False

if __name__ == "__main__":
    success = asyncio.run(test_no_vad())
    if success:
        print("\n✅ VAD test completed!")
        sys.exit(0)
    else:
        print("\n❌ VAD test failed!")
        sys.exit(1)