#!/usr/bin/env python3
"""
Simple test for audio format with GPU STT and CPU Resemblyzer
"""

import os
import sys
import time
import logging
import numpy as np
from pathlib import Path

# Set environment to use GPU STT and CPU Resemblyzer
os.environ['STT_BACKEND'] = 'faster_gpu'
os.environ['RESEMBLYZER_USE_GPU'] = 'false'

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_simple_audio():
    """Test simple audio processing with GPU STT and CPU Resemblyzer."""
    logger.info("🧪 Testing simple audio processing...")
    
    try:
        # Import components
        from src.audio import STTEngine
        from src.core.voice_assistant import VoiceAssistant
        
        # Initialize VoiceAssistant (this will test both STT and Resemblyzer)
        logger.info("📝 Initializing VoiceAssistant...")
        voice_assistant = VoiceAssistant()
        
        logger.info(f"✅ VoiceAssistant initialized successfully!")
        logger.info(f"🔧 STT Engine: {type(voice_assistant.stt_engine).__name__}")
        logger.info(f"🔧 STT Device: {getattr(voice_assistant.stt_engine, 'device', 'unknown')}")
        
        # Create test audio in the format FastRTC sends
        sample_rate = 16000
        duration = 2.0
        
        # Create speech-like audio
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        
        # Create a more realistic speech signal
        # Simulate formants and speech patterns
        f1, f2, f3 = 200, 800, 1200  # Typical formant frequencies
        signal = (
            0.4 * np.sin(2 * np.pi * f1 * t) +
            0.3 * np.sin(2 * np.pi * f2 * t) +
            0.2 * np.sin(2 * np.pi * f3 * t)
        )
        
        # Add speech-like amplitude modulation
        envelope = 0.5 * (1 + np.sin(2 * np.pi * 4 * t))  # 4 Hz modulation
        signal = signal * envelope
        
        # Add some noise to make it more realistic
        noise = 0.05 * np.random.randn(len(signal))
        signal = signal + noise
        
        # Normalize to reasonable speech levels
        signal = signal / np.max(np.abs(signal)) * 0.5
        audio_array = signal.astype(np.float32)
        
        logger.info(f"🔊 Created test audio: {audio_array.shape} samples at {sample_rate}Hz")
        logger.info(f"🔊 Audio stats: min={audio_array.min():.3f}, max={audio_array.max():.3f}, rms={np.sqrt(np.mean(audio_array**2)):.3f}")
        
        # Test STT transcription with the audio format FastRTC uses
        logger.info("🎯 Testing STT transcription...")
        start_time = time.time()
        
        # Test with the _transcribe_audio method (used by callback handlers)
        result = await voice_assistant.stt_engine._transcribe_audio(audio_array, "en")
        
        elapsed = time.time() - start_time
        
        logger.info(f"✅ STT transcription completed in {elapsed:.3f}s")
        logger.info(f"📝 Result: '{result.text}'")
        logger.info(f"🌍 Language: {result.language}")
        logger.info(f"📊 Confidence: {result.confidence:.3f}")
        
        # Test with different languages
        logger.info("🎯 Testing different languages...")
        for lang in ["es", "fr", "de"]:
            start_time = time.time()
            result_lang = await voice_assistant.stt_engine._transcribe_audio(audio_array, lang)
            elapsed_lang = time.time() - start_time
            logger.info(f"🌍 {lang.upper()}: '{result_lang.text}' ({elapsed_lang:.3f}s)")
        
        # Test voice assistant's process_audio method (the main entry point)
        logger.info("🎯 Testing VoiceAssistant.process_audio...")
        start_time = time.time()
        
        # This is the method that gets called by the callback handlers
        processed_sr, processed_audio = voice_assistant.process_audio(audio_array)
        
        elapsed_process = time.time() - start_time
        
        logger.info(f"✅ VoiceAssistant.process_audio completed in {elapsed_process:.3f}s")
        logger.info(f"📊 Processed: {processed_audio.shape if processed_audio is not None else 'None'} samples at {processed_sr}Hz")
        
        # Cleanup
        if hasattr(voice_assistant.stt_engine, 'shutdown'):
            voice_assistant.stt_engine.shutdown()
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error testing simple audio: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Main async function"""
    logger.info("=" * 60)
    logger.info("🧪 Simple Audio Test (GPU STT + CPU Resemblyzer)")
    logger.info("=" * 60)
    
    # Test simple audio processing
    success = await test_simple_audio()
    
    logger.info("")
    logger.info("=" * 60)
    if success:
        logger.info("✅ Simple audio test completed successfully!")
        logger.info("🎉 Your FastRTC should now work with GPU STT and stable Resemblyzer!")
    else:
        logger.error("❌ Simple audio test failed!")
    logger.info("=" * 60)
    
    return success

if __name__ == "__main__":
    import asyncio
    success = asyncio.run(main())
    sys.exit(0 if success else 1)