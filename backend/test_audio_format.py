#!/usr/bin/env python3
"""
Test audio format handling with GPU STT engine
"""

import os
import sys
import time
import logging
import numpy as np
from pathlib import Path

# Set environment to use GPU STT and CPU Resemblyzer
os.environ['STT_BACKEND'] = 'faster_gpu'
os.environ['RESEMBLYZER_USE_GPU'] = 'false'  # Use CPU for Resemblyzer

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_audio_format():
    """Test audio format handling with the actual FastRTC format."""
    logger.info("🧪 Testing audio format handling...")
    
    try:
        # Import the components
        from src.audio import STTEngine
        from src.core.voice_assistant import VoiceAssistant
        from src.integration.streaming_callback_handler import StreamCallbackHandler
        from src.audio import KokoroTTSEngine, VoiceMapper
        
        # Initialize components
        logger.info("📝 Initializing components...")
        voice_assistant = VoiceAssistant()
        stt_engine = voice_assistant.stt_engine
        tts_engine = KokoroTTSEngine()
        voice_mapper = VoiceMapper()
        
        logger.info(f"✅ STT Engine: {type(stt_engine).__name__}")
        logger.info(f"🔧 STT Device: {getattr(stt_engine, 'device', 'unknown')}")
        
        # Create callback handler
        callback_handler = StreamCallbackHandler(
            voice_assistant=voice_assistant,
            stt_engine=stt_engine,
            tts_engine=tts_engine,
            voice_mapper=voice_mapper
        )
        
        # Create test audio in FastRTC format: (sample_rate, audio_array)
        sample_rate = 16000
        duration = 2.0
        
        # Create speech-like audio
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        f1, f2, f3 = 200, 800, 1200
        signal = (
            0.3 * np.sin(2 * np.pi * f1 * t) +
            0.2 * np.sin(2 * np.pi * f2 * t) +
            0.1 * np.sin(2 * np.pi * f3 * t)
        )
        envelope = 0.5 * (1 + np.sin(2 * np.pi * 3 * t))
        signal = signal * envelope
        signal = signal / np.max(np.abs(signal)) * 0.3
        audio_array = signal.astype(np.float32)
        
        # FastRTC format: (sample_rate, audio_array)
        audio_data_tuple = (sample_rate, audio_array)
        
        logger.info(f"🔊 Created test audio: {audio_array.shape} samples at {sample_rate}Hz")
        logger.info(f"🔊 Audio format: {audio_array.dtype}, range: [{audio_array.min():.3f}, {audio_array.max():.3f}]")
        
        # Test audio preprocessing
        logger.info("🎯 Testing audio preprocessing...")
        processed_audio, processed_sr = callback_handler._preprocess_audio(audio_data_tuple)
        
        if processed_audio is not None:
            logger.info(f"✅ Audio preprocessing successful")
            logger.info(f"📊 Processed: {processed_audio.shape} samples at {processed_sr}Hz")
            logger.info(f"📊 Processed format: {processed_audio.dtype}, range: [{processed_audio.min():.3f}, {processed_audio.max():.3f}]")
        else:
            logger.error("❌ Audio preprocessing failed")
            return False
        
        # Test STT transcription directly
        logger.info("🎯 Testing STT transcription...")
        start_time = time.time()
        
        result = await stt_engine._transcribe_audio(processed_audio, "en")
        
        elapsed = time.time() - start_time
        
        logger.info(f"✅ STT transcription completed in {elapsed:.3f}s")
        logger.info(f"📝 Result: '{result.text}'")
        logger.info(f"🌍 Language: {result.language}")
        logger.info(f"📊 Confidence: {result.confidence:.3f}")
        
        # Test streaming STT (the method used by callback handler)
        logger.info("🎯 Testing streaming STT...")
        start_time = time.time()
        
        stream_result = await callback_handler._stream_audio_to_stt(processed_audio)
        
        elapsed_stream = time.time() - start_time
        
        if stream_result:
            logger.info(f"✅ Streaming STT completed in {elapsed_stream:.3f}s")
            logger.info(f"📝 Stream result: '{stream_result.text}'")
        else:
            logger.warning("⚠️ Streaming STT returned None")
        
        # Cleanup
        if hasattr(stt_engine, 'shutdown'):
            stt_engine.shutdown()
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error testing audio format: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Main async function"""
    logger.info("=" * 60)
    logger.info("🧪 FastRTC Audio Format Test")
    logger.info("=" * 60)
    
    # Test audio format handling
    success = await test_audio_format()
    
    logger.info("")
    logger.info("=" * 60)
    if success:
        logger.info("✅ Audio format test completed successfully!")
    else:
        logger.error("❌ Audio format test failed!")
    logger.info("=" * 60)
    
    return success

if __name__ == "__main__":
    import asyncio
    success = asyncio.run(main())
    sys.exit(0 if success else 1)