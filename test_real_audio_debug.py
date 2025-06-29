#!/usr/bin/env python3
"""
Debug real audio transcription to verify it's actually working
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
os.environ['LOG_LEVEL'] = 'DEBUG'

from backend.src.utils.logging import get_logger
from backend.src.audio.engines.stt.faster_whisper_gpu_stt import FasterWhisperGPUSTT
from backend.src.core.interfaces import AudioData

logger = get_logger(__name__)

def load_wav_file(file_path):
    """Load WAV file using scipy."""
    try:
        from scipy.io import wavfile
        sr, audio = wavfile.read(file_path)
        print(f"📁 Raw audio loaded: shape={audio.shape}, dtype={audio.dtype}, sr={sr}")
        
        # Handle stereo
        if audio.ndim > 1:
            print(f"🔧 Converting stereo to mono: {audio.shape} -> taking first channel")
            audio = audio[:, 0]
        
        # Convert to float32
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0
        elif audio.dtype == np.int32:
            audio = audio.astype(np.float32) / 2147483648.0
        else:
            audio = audio.astype(np.float32)
            
        print(f"📁 Processed audio: shape={audio.shape}, dtype={audio.dtype}, sr={sr}")
        print(f"🔍 Audio stats: min={audio.min():.6f}, max={audio.max():.6f}, rms={np.sqrt(np.mean(audio**2)):.6f}")
        
        return audio, sr
    except Exception as e:
        print(f"❌ Failed to load audio: {e}")
        return None, None

async def test_direct_stt():
    """Test STT directly without any caching."""
    audio_file = "/mnt/c/Users/PC/Dev/fastRTC/backend/tests/samples/audio_en.wav"
    
    print(f"🧪 Testing direct STT with: {audio_file}")
    
    # Check if file exists
    if not Path(audio_file).exists():
        print(f"❌ Audio file not found: {audio_file}")
        return False
    
    try:
        # Load audio file
        print("📁 Loading audio file...")
        audio_samples, sample_rate = load_wav_file(audio_file)
        
        if audio_samples is None:
            print("❌ Failed to load audio file")
            return False
        
        duration = len(audio_samples) / sample_rate
        print(f"🎵 Audio duration: {duration:.2f}s")
        
        # Create fresh STT engine instance
        print("📝 Creating fresh STT engine...")
        stt = FasterWhisperGPUSTT()
        
        if not stt.is_available():
            print("❌ STT engine not available")
            return False
        
        print("✅ STT engine initialized")
        
        # Test with different audio segments to verify it's actually transcribing
        segments = [
            (0, min(3, duration)),  # First 3 seconds
            (max(0, duration/2 - 1.5), min(duration, duration/2 + 1.5)),  # Middle 3 seconds
            (max(0, duration - 3), duration)  # Last 3 seconds
        ]
        
        for i, (start, end) in enumerate(segments):
            start_sample = int(start * sample_rate)
            end_sample = int(end * sample_rate)
            segment_audio = audio_samples[start_sample:end_sample]
            segment_duration = len(segment_audio) / sample_rate
            
            print(f"\n🎯 Testing segment {i+1}: {start:.1f}s-{end:.1f}s ({segment_duration:.1f}s)")
            print(f"🔍 Segment stats: shape={segment_audio.shape}, rms={np.sqrt(np.mean(segment_audio**2)):.6f}")
            
            # Create AudioData
            audio_data = AudioData(samples=segment_audio, sample_rate=sample_rate, duration=segment_duration)
            
            # Transcribe
            print(f"🚀 Transcribing segment {i+1}...")
            start_time = asyncio.get_event_loop().time()
            result = await stt._transcribe_audio(audio_data)
            end_time = asyncio.get_event_loop().time()
            
            processing_time = end_time - start_time
            print(f"✅ Segment {i+1} result: '{result.text}' (confidence: {result.confidence:.3f}, time: {processing_time:.3f}s)")
        
        # Test full audio
        print(f"\n🎯 Testing full audio ({duration:.1f}s)")
        audio_data_full = AudioData(samples=audio_samples, sample_rate=sample_rate, duration=duration)
        
        print("🚀 Transcribing full audio...")
        start_time = asyncio.get_event_loop().time()
        result_full = await stt._transcribe_audio(audio_data_full)
        end_time = asyncio.get_event_loop().time()
        
        processing_time_full = end_time - start_time
        print(f"✅ Full audio result: '{result_full.text}' (confidence: {result_full.confidence:.3f}, time: {processing_time_full:.3f}s)")
        
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
    success = asyncio.run(test_direct_stt())
    if success:
        print("\n✅ Direct STT test completed!")
        sys.exit(0)
    else:
        print("\n❌ Direct STT test failed!")
        sys.exit(1)