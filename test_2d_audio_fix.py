#!/usr/bin/env python3
"""
Test 2D audio handling fix for threading pipeline
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
from backend.src.core.interfaces import AudioData

logger = get_logger(__name__)

async def test_2d_audio():
    """Test GPU STT with 2D audio like threading pipeline sends."""
    logger.info("🧪 Testing GPU STT with 2D audio...")
    
    try:
        # Initialize STT engine
        logger.info(f"📝 Initializing {STTEngine.__name__}...")
        stt = STTEngine()
        
        if not stt.is_available():
            logger.error("❌ STT engine not available")
            return False
        
        logger.info("✅ STT engine initialized successfully")
        
        # Create test audio (2D like threading pipeline - 4 seconds at 48kHz)
        sample_rate = 48000
        duration = 4.0
        samples = int(sample_rate * duration)
        
        # Create 2D audio array (1, samples) like the threading pipeline
        t = np.linspace(0, duration, samples, False)
        frequency = 440  # A4 note
        audio_1d = 0.3 * np.sin(2 * np.pi * frequency * t).astype(np.float32)
        audio_2d = audio_1d.reshape(1, -1)  # Make it 2D like threading pipeline
        
        logger.info(f"🎵 Created 2D test audio: {audio_2d.shape}, {audio_2d.dtype}")
        logger.info(f"🔍 Audio stats: min={audio_2d.min():.6f}, max={audio_2d.max():.6f}, rms={np.sqrt(np.mean(audio_2d**2)):.6f}")
        
        # Test transcription with 2D audio
        audio_data = AudioData(samples=audio_2d, sample_rate=sample_rate, duration=duration)
        
        logger.info("🚀 Starting transcription with 2D audio...")
        start_time = asyncio.get_event_loop().time()
        result = await stt.transcribe(audio_data)
        end_time = asyncio.get_event_loop().time()
        
        processing_time = end_time - start_time
        logger.info(f"✅ Transcription result: '{result.text}' (confidence: {result.confidence:.3f}, time: {processing_time:.3f}s)")
        
        # Cleanup
        stt.shutdown()
        logger.info("🧹 STT engine shutdown complete")
        
        # Check if we got a reasonable result (even if empty, it should not crash)
        if result is not None:
            logger.info("✅ STT handled 2D audio without crashing")
            return True
        else:
            logger.error("❌ STT returned None")
            return False
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        logger.error(f"❌ Traceback: {traceback.format_exc()}")
        return False

if __name__ == "__main__":
    success = asyncio.run(test_2d_audio())
    if success:
        print("✅ 2D audio test passed!")
        sys.exit(0)
    else:
        print("❌ 2D audio test failed!")
        sys.exit(1)