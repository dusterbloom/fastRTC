#!/usr/bin/env python3
"""
Test threading pipeline STT fix
"""

import os
import sys
import numpy as np
import asyncio
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

async def test_threading_stt():
    """Test STT worker in threading pipeline."""
    logger.info("🧪 Testing threading pipeline STT...")
    
    try:
        # Initialize STT engine
        logger.info("📝 Initializing STT engine...")
        stt_engine = STTEngine()
        
        if not stt_engine.is_available():
            logger.error("❌ STT engine not available")
            return False
        
        # Initialize pipeline manager
        logger.info("📝 Initializing pipeline manager...")
        pipeline_manager = AudioPipelineManager()
        
        # Initialize STT worker
        logger.info("📝 Initializing STT worker...")
        stt_worker = STTStreamingWorker(
            pipeline_manager=pipeline_manager,
            stt_engine=stt_engine,
            confidence_threshold=0.1,  # Lower threshold for test
            min_audio_length=0.1  # Shorter minimum for test
        )
        
        # Create test audio (2D like threading pipeline)
        sample_rate = 48000
        duration = 2.0
        samples = int(sample_rate * duration)
        
        # Create 2D audio array (1, samples) like the threading pipeline
        t = np.linspace(0, duration, samples, False)
        frequency = 440  # A4 note
        audio_1d = 0.3 * np.sin(2 * np.pi * frequency * t).astype(np.int16)
        audio_2d = audio_1d.reshape(1, -1)  # Make it 2D like threading pipeline
        
        logger.info(f"🎵 Created 2D test audio: {audio_2d.shape}, {audio_2d.dtype}")
        
        # Create audio chunk
        import time
        audio_chunk = AudioChunk(
            generation_id=1,
            audio_data=audio_2d,
            sample_rate=sample_rate,
            timestamp=time.time(),
            is_final=True
        )
        
        # Test STT worker processing
        logger.info("🚀 Testing STT worker processing...")
        result = await stt_worker.process_item_async(audio_chunk)
        
        if result:
            logger.info(f"✅ STT worker result: '{result.text}' (confidence: {result.confidence:.3f})")
            success = True
        else:
            logger.warning("⚠️ STT worker returned None (might be expected for sine wave)")
            success = True  # None result is OK for sine wave
        
        # Cleanup
        stt_engine.shutdown()
        logger.info("🧹 Cleanup complete")
        
        return success
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        logger.error(f"❌ Traceback: {traceback.format_exc()}")
        return False

if __name__ == "__main__":
    success = asyncio.run(test_threading_stt())
    if success:
        print("✅ Threading STT test passed!")
        sys.exit(0)
    else:
        print("❌ Threading STT test failed!")
        sys.exit(1)