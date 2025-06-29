#!/usr/bin/env python3
"""
Test Real Threading Callback Handler Integration
Verify all our STT fixes work in the actual system
"""

import os
import sys
import numpy as np
import asyncio
import time
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent / "backend"))

# Set environment for threading pipeline
os.environ['STT_BACKEND'] = 'faster_gpu'
os.environ['USE_THREADING_PIPELINE'] = 'true'
os.environ['LOG_LEVEL'] = 'INFO'

from backend.src.utils.logging import get_logger
from backend.src.integration.threading_callback_handler import ThreadingCallbackHandler
from backend.src.audio.engines.stt import STTEngine
from backend.src.audio.engines.tts.kokoro_tts import KokoroTTSEngine
from backend.src.audio import VoiceMapper

logger = get_logger(__name__)

class MockVoiceAssistant:
    """Mock voice assistant for testing."""
    def __init__(self):
        self.current_generation_id = 1
        self.voice_print_manager = None  # No voice auth for test
        
    def get_current_generation_id(self):
        return self.current_generation_id

def test_real_threading_integration():
    """Test the real threading callback handler with our fixes."""
    logger.info("🚀 Testing real threading callback handler integration...")
    
    try:
        # Initialize engines
        logger.info("📝 Initializing engines...")
        stt_engine = STTEngine()
        tts_engine = KokoroTTSEngine()
        voice_mapper = VoiceMapper()
        mock_voice_assistant = MockVoiceAssistant()
        
        if not stt_engine.is_available():
            logger.error("❌ STT engine not available")
            return False
        
        # Initialize threading callback handler
        logger.info("📝 Initializing threading callback handler...")
        handler = ThreadingCallbackHandler(
            voice_assistant=mock_voice_assistant,
            stt_engine=stt_engine,
            tts_engine=tts_engine,
            voice_mapper=voice_mapper
        )
        
        # Start the pipeline
        logger.info("🚀 Starting threading pipeline...")
        handler.start()
        
        # Load test audio
        logger.info("📁 Loading test audio...")
        import librosa
        audio_file = "/mnt/c/Users/PC/Dev/fastRTC/backend/tests/samples/audio_en.wav"
        audio_data, sr = librosa.load(audio_file, sr=None)
        
        # Convert to format expected by threading callback (2D)
        if audio_data.ndim == 1:
            audio_data = audio_data.reshape(1, -1)
        
        logger.info(f"🎵 Loaded audio: {audio_data.shape}, {audio_data.dtype}, {sr}Hz")
        
        # Create audio tuple format expected by callback
        # FastRTC typically sends (sample_rate, audio_data)
        audio_data_tuple = (sr, audio_data)
        
        # Process audio through the real callback handler
        logger.info("🎤 Processing audio through real threading callback...")
        
        output_chunks = []
        start_time = time.time()
        
        # Process the audio stream
        for output_chunk in handler.process_audio_stream(audio_data_tuple):
            output_chunks.append(output_chunk)
            logger.info(f"📢 Received output chunk: {type(output_chunk)}")
            
            # Break after reasonable time to avoid infinite loop
            if time.time() - start_time > 10.0:
                logger.info("⏰ Breaking after 10 seconds")
                break
        
        processing_time = time.time() - start_time
        
        # Stop the pipeline
        logger.info("🛑 Stopping threading pipeline...")
        handler.stop()
        
        # Results
        logger.info("📊 Integration Test Results:")
        logger.info(f"   Processing time: {processing_time:.3f}s")
        logger.info(f"   Output chunks: {len(output_chunks)}")
        logger.info(f"   Total callbacks: {handler.total_callbacks}")
        logger.info(f"   Successful callbacks: {handler.successful_callbacks}")
        
        # Cleanup
        stt_engine.shutdown()
        logger.info("🧹 Cleanup complete")
        
        success = len(output_chunks) > 0
        
        if success:
            logger.info("🎉 Real threading integration test PASSED!")
        else:
            logger.error("❌ Real threading integration test FAILED!")
            
        return success
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        logger.error(f"❌ Traceback: {traceback.format_exc()}")
        return False

if __name__ == "__main__":
    success = test_real_threading_integration()
    if success:
        print("✅ Real threading integration test passed!")
        sys.exit(0)
    else:
        print("❌ Real threading integration test failed!")
        sys.exit(1)