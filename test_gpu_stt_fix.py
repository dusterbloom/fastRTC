#!/usr/bin/env python3
"""
Test GPU STT fix for threading pipeline
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

async def test_gpu_stt():
    """Test GPU STT functionality."""
    logger.info("🧪 Testing GPU STT fix...")
    
    try:
        # Initialize STT engine
        logger.info(f"📝 Initializing {STTEngine.__name__}...")
        stt = STTEngine()
        
        if not stt.is_available():
            logger.error("❌ STT engine not available")
            return False
        
        logger.info("✅ STT engine initialized successfully")
        
        # Create test audio (1 second of sine wave at 440Hz)
        sample_rate = 16000
        duration = 1.0
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        frequency = 440  # A4 note
        audio_samples = 0.3 * np.sin(2 * np.pi * frequency * t).astype(np.float32)
        
        logger.info(f"🎵 Created test audio: {audio_samples.shape}, {audio_samples.dtype}")
        
        # Test transcription
        audio_data = AudioData(samples=audio_samples, sample_rate=sample_rate, duration=duration)
        
        logger.info("🚀 Starting transcription...")
        result = await stt.transcribe(audio_data)
        
        logger.info(f"✅ Transcription result: '{result.text}' (confidence: {result.confidence:.3f})")
        
        # Cleanup
        stt.shutdown()
        logger.info("🧹 STT engine shutdown complete")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    success = asyncio.run(test_gpu_stt())
    if success:
        print("✅ GPU STT test passed!")
        sys.exit(0)
    else:
        print("❌ GPU STT test failed!")
        sys.exit(1)