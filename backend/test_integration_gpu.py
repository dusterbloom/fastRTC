#!/usr/bin/env python3
"""
Test GPU STT integration with the main application
"""

import os
import sys
import time
import logging
import numpy as np
from pathlib import Path

# Set environment to use GPU STT
os.environ['STT_BACKEND'] = 'faster_gpu'

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_integration():
    """Test GPU STT integration with main application components."""
    logger.info("🧪 Testing GPU STT integration...")
    
    try:
        # Import main audio module
        from src.audio import STTEngine
        
        # Initialize STT engine through main interface
        logger.info("📝 Initializing STT engine through main interface...")
        stt_engine = STTEngine()
        
        if not stt_engine.is_available():
            logger.error("❌ STT engine is not available")
            return False
        
        logger.info("✅ STT engine initialized successfully")
        logger.info(f"🔧 Engine type: {type(stt_engine).__name__}")
        logger.info(f"🔧 Device: {getattr(stt_engine, 'device', 'unknown')}")
        logger.info(f"🔧 Compute type: {getattr(stt_engine, 'compute_type', 'unknown')}")
        
        # Test with voice assistant
        logger.info("🎯 Testing with VoiceAssistant...")
        from src.core.voice_assistant import VoiceAssistant
        
        voice_assistant = VoiceAssistant()
        logger.info(f"🔧 VoiceAssistant STT engine type: {type(voice_assistant.stt_engine).__name__}")
        
        # Create test audio
        sample_rate = 16000
        duration = 1.0
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        audio_samples = np.sin(2 * np.pi * 440 * t).astype(np.float32) * 0.1
        
        logger.info("🔊 Testing transcription through VoiceAssistant...")
        start_time = time.time()
        
        result = await voice_assistant.stt_engine.transcribe(audio_samples)
        
        elapsed = time.time() - start_time
        
        logger.info(f"✅ Transcription completed in {elapsed:.3f}s")
        logger.info(f"📝 Result: '{result.text if hasattr(result, 'text') else result}'")
        
        # Test callback handler integration
        logger.info("🎯 Testing callback handler integration...")
        from src.integration.callback_handler import CallbackHandler
        from src.audio import KokoroTTSEngine, VoiceMapper
        
        # Initialize other components
        tts_engine = KokoroTTSEngine()
        voice_mapper = VoiceMapper()
        
        callback_handler = CallbackHandler(
            voice_assistant=voice_assistant,
            stt_engine=voice_assistant.stt_engine,
            tts_engine=tts_engine,
            voice_mapper=voice_mapper
        )
        
        logger.info(f"🔧 CallbackHandler STT engine type: {type(callback_handler.stt_engine).__name__}")
        
        # Cleanup
        if hasattr(voice_assistant.stt_engine, 'shutdown'):
            voice_assistant.stt_engine.shutdown()
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error testing integration: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Main async function"""
    logger.info("=" * 60)
    logger.info("🧪 FastRTC GPU STT Integration Test")
    logger.info("=" * 60)
    
    # Test integration
    success = await test_integration()
    
    logger.info("")
    logger.info("=" * 60)
    if success:
        logger.info("✅ GPU STT integration test completed successfully!")
    else:
        logger.error("❌ GPU STT integration test failed!")
    logger.info("=" * 60)
    
    return success

if __name__ == "__main__":
    import asyncio
    success = asyncio.run(main())
    sys.exit(0 if success else 1)