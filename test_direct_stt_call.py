#!/usr/bin/env python3
"""
Test direct STT call like the working streaming implementation
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
from backend.src.audio.engines.stt import STTEngine

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

async def test_direct_call():
    """Test direct _transcribe_audio call like working streaming implementation."""
    audio_file = "/mnt/c/Users/PC/Dev/fastRTC/backend/tests/samples/audio_en.wav"
    
    print(f"🧪 Testing direct _transcribe_audio call: {audio_file}")
    
    try:
        # Load audio file
        audio_samples, sample_rate = load_wav_file(audio_file)
        if audio_samples is None:
            return False
        
        duration = len(audio_samples) / sample_rate
        print(f"🎵 Audio duration: {duration:.2f}s")
        
        # Resample to 16kHz like the working implementation
        TARGET_SAMPLE_RATE = 16000
        if sample_rate != TARGET_SAMPLE_RATE:
            from scipy.signal import resample
            num_samples = int(len(audio_samples) * TARGET_SAMPLE_RATE / sample_rate)
            audio_samples = resample(audio_samples, num_samples)
            print(f"🔄 Resampled {sample_rate}Hz → {TARGET_SAMPLE_RATE}Hz")
            sample_rate = TARGET_SAMPLE_RATE
        
        # Initialize STT engine
        print("📝 Initializing STT engine...")
        stt = STTEngine()
        
        if not stt.is_available():
            print("❌ STT engine not available")
            return False
        
        # Test direct call like working streaming implementation
        print("🚀 Testing direct _transcribe_audio call...")
        start_time = asyncio.get_event_loop().time()
        
        # Call _transcribe_audio directly with numpy array (like working implementation)
        result = await stt._transcribe_audio(audio_samples)
        
        end_time = asyncio.get_event_loop().time()
        processing_time = end_time - start_time
        
        print(f"✅ Direct call result: '{result.text}' (confidence: {result.confidence:.3f}, time: {processing_time:.3f}s)")
        
        # Compare with AudioData wrapper approach
        print("🚀 Testing with AudioData wrapper...")
        from backend.src.core.interfaces import AudioData
        
        duration = len(audio_samples) / sample_rate
        audio_data = AudioData(samples=audio_samples, sample_rate=sample_rate, duration=duration)
        
        start_time = asyncio.get_event_loop().time()
        result_wrapped = await stt._transcribe_audio(audio_data)
        end_time = asyncio.get_event_loop().time()
        
        processing_time_wrapped = end_time - start_time
        print(f"✅ Wrapped call result: '{result_wrapped.text}' (confidence: {result_wrapped.confidence:.3f}, time: {processing_time_wrapped:.3f}s)")
        
        # Compare results
        print(f"\n📊 Comparison:")
        print(f"   Direct call:  '{result.text}' ({len(result.text)} chars)")
        print(f"   Wrapped call: '{result_wrapped.text}' ({len(result_wrapped.text)} chars)")
        
        if result.text == result_wrapped.text:
            print("✅ Both approaches produce identical results!")
        else:
            print("⚠️ Results differ between approaches")
        
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
    success = asyncio.run(test_direct_call())
    if success:
        print("\n✅ Direct call test completed!")
        sys.exit(0)
    else:
        print("\n❌ Direct call test failed!")
        sys.exit(1)