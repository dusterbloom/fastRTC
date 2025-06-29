#!/usr/bin/env python3
"""
Test threading pipeline with real audio file and resampling
"""

import os
import sys
import numpy as np
import asyncio
import time
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent / "backend"))

# Set environment for GPU STT and threading
os.environ['STT_BACKEND'] = 'faster_gpu'
os.environ['USE_THREADING_PIPELINE'] = 'true'
os.environ['LOG_LEVEL'] = 'INFO'

from backend.src.utils.logging import get_logger
from backend.src.core.stt_worker import STTStreamingWorker
from backend.src.core.pipeline_manager import AudioPipelineManager, AudioChunk
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

async def test_threading_real_audio():
    """Test threading STT worker with real audio file."""
    audio_file = "/mnt/c/Users/PC/Dev/fastRTC/backend/tests/samples/audio_en.wav"
    
    print(f"🧪 Testing threading pipeline with real audio: {audio_file}")
    
    try:
        # Load audio file
        audio_samples, sample_rate = load_wav_file(audio_file)
        if audio_samples is None:
            return False
        
        duration = len(audio_samples) / sample_rate
        print(f"🎵 Original audio: {duration:.2f}s at {sample_rate}Hz")
        
        # Initialize STT engine
        print("📝 Initializing STT engine...")
        stt_engine = STTEngine()
        
        if not stt_engine.is_available():
            print("❌ STT engine not available")
            return False
        
        # Initialize pipeline manager
        print("📝 Initializing pipeline manager...")
        pipeline_manager = AudioPipelineManager()
        
        # Initialize STT worker
        print("📝 Initializing STT worker...")
        stt_worker = STTStreamingWorker(
            pipeline_manager=pipeline_manager,
            stt_engine=stt_engine,
            confidence_threshold=0.1,  # Lower threshold for test
            min_audio_length=0.1  # Shorter minimum for test
        )
        
        # Create 2D audio array like threading pipeline (simulate WebRTC format)
        audio_2d = audio_samples.reshape(1, -1)
        print(f"🔧 Created 2D audio: {audio_2d.shape}")
        
        # Create generation state in pipeline manager
        generation_id = pipeline_manager.create_generation()
        print(f"🆕 Created generation: {generation_id}")
        
        # Create audio chunk
        audio_chunk = AudioChunk(
            generation_id=generation_id,
            audio_data=audio_2d,
            sample_rate=sample_rate,  # Original sample rate (will be resampled by worker)
            timestamp=time.time(),
            is_final=True
        )
        
        # Test STT worker processing
        print("🚀 Testing STT worker with real audio...")
        start_time = asyncio.get_event_loop().time()
        result = await stt_worker.process_item_async(audio_chunk)
        end_time = asyncio.get_event_loop().time()
        
        processing_time = end_time - start_time
        
        if result:
            print(f"✅ Threading STT result: '{result.text}' (confidence: {result.confidence:.3f}, time: {processing_time:.3f}s)")
            print(f"📊 Performance: {duration/processing_time:.1f}x real-time")
            
            # Check if we got the expected result
            expected_phrases = ["stale smell", "old beer", "hot cross bun", "tacos", "pickle"]
            found_phrases = sum(1 for phrase in expected_phrases if phrase.lower() in result.text.lower())
            
            if found_phrases >= 3:
                print(f"✅ Transcription looks correct! Found {found_phrases}/{len(expected_phrases)} expected phrases")
                success = True
            else:
                print(f"⚠️ Transcription may be incorrect. Only found {found_phrases}/{len(expected_phrases)} expected phrases")
                success = False
        else:
            print("❌ STT worker returned None")
            success = False
        
        # Cleanup
        stt_engine.shutdown()
        print("🧹 Cleanup complete")
        
        return success
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        print(f"❌ Traceback: {traceback.format_exc()}")
        return False

if __name__ == "__main__":
    success = asyncio.run(test_threading_real_audio())
    if success:
        print("\n✅ Threading real audio test passed!")
        sys.exit(0)
    else:
        print("\n❌ Threading real audio test failed!")
        sys.exit(1)