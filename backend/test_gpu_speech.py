#!/usr/bin/env python3
"""
Test GPU STT with actual speech audio
"""

import os
import sys
import time
import logging
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_speech_like_audio():
    """Create audio that sounds more like speech."""
    sample_rate = 16000
    duration = 2.0  # 2 seconds
    
    # Create a more complex waveform that might trigger transcription
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    
    # Mix multiple frequencies to simulate speech formants
    f1 = 200  # Fundamental frequency
    f2 = 800  # First formant
    f3 = 1200 # Second formant
    
    # Create speech-like signal
    signal = (
        0.3 * np.sin(2 * np.pi * f1 * t) +
        0.2 * np.sin(2 * np.pi * f2 * t) +
        0.1 * np.sin(2 * np.pi * f3 * t)
    )
    
    # Add some amplitude modulation to simulate speech patterns
    envelope = 0.5 * (1 + np.sin(2 * np.pi * 3 * t))  # 3 Hz modulation
    signal = signal * envelope
    
    # Add some noise to make it more realistic
    noise = 0.01 * np.random.randn(len(signal))
    signal = signal + noise
    
    # Normalize
    signal = signal / np.max(np.abs(signal)) * 0.5
    
    return signal.astype(np.float32)

async def test_gpu_stt_with_speech():
    """Test GPU STT with speech-like audio."""
    logger.info("🧪 Testing GPU STT with speech-like audio...")
    
    try:
        # Import GPU STT engine directly
        from src.audio.engines.stt.faster_whisper_gpu_stt import FasterWhisperGPUSTT
        
        # Initialize STT engine
        logger.info("📝 Initializing FasterWhisperGPUSTT...")
        stt = FasterWhisperGPUSTT()
        
        if not stt.is_available():
            logger.error("❌ STT engine is not available")
            return False
        
        logger.info("✅ STT engine initialized successfully")
        logger.info(f"🔧 Device: {stt.device}")
        logger.info(f"🔧 Compute type: {stt.compute_type}")
        
        # Create speech-like audio
        logger.info("🔊 Creating speech-like audio...")
        audio_samples = create_speech_like_audio()
        
        logger.info(f"🔊 Created audio: {audio_samples.shape} samples")
        logger.info(f"🔊 Duration: {len(audio_samples) / 16000:.2f}s")
        logger.info(f"🔊 RMS level: {np.sqrt(np.mean(audio_samples**2)):.4f}")
        
        # Test transcription with VAD disabled to ensure processing
        logger.info("🎯 Testing transcription (VAD disabled)...")
        
        # Temporarily disable VAD for this test
        original_vad = stt.vad_filter
        stt.vad_filter = False
        
        start_time = time.time()
        result = stt._transcribe_sync(audio_samples, target_language="en")
        elapsed = time.time() - start_time
        
        # Restore VAD setting
        stt.vad_filter = original_vad
        
        logger.info(f"✅ Transcription completed in {elapsed:.3f}s")
        logger.info(f"📝 Result: '{result.text}'")
        logger.info(f"🌍 Language: {result.language}")
        logger.info(f"📊 Confidence: {result.confidence:.3f}")
        
        # Test with different languages
        logger.info("🎯 Testing with different target languages...")
        
        languages = ["en", "es", "fr", "de"]
        for lang in languages:
            start_time = time.time()
            result_lang = stt._transcribe_sync(audio_samples, target_language=lang)
            elapsed_lang = time.time() - start_time
            
            logger.info(f"🌍 {lang.upper()}: '{result_lang.text}' ({elapsed_lang:.3f}s)")
        
        # Performance test
        logger.info("🎯 Performance test (10 transcriptions)...")
        times = []
        
        for i in range(10):
            start_time = time.time()
            result_perf = stt._transcribe_sync(audio_samples, target_language="en")
            elapsed_perf = time.time() - start_time
            times.append(elapsed_perf)
            logger.info(f"   Run {i+1}: {elapsed_perf:.3f}s")
        
        avg_time = np.mean(times)
        min_time = np.min(times)
        max_time = np.max(times)
        
        logger.info(f"📊 Performance stats:")
        logger.info(f"   Average: {avg_time:.3f}s")
        logger.info(f"   Min: {min_time:.3f}s")
        logger.info(f"   Max: {max_time:.3f}s")
        logger.info(f"   Real-time factor: {avg_time / (len(audio_samples) / 16000):.2f}x")
        
        # Cleanup
        stt.shutdown()
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error testing GPU STT: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Main async function"""
    logger.info("=" * 60)
    logger.info("🧪 FastRTC GPU STT Speech Test")
    logger.info("=" * 60)
    
    # Test STT with speech-like audio
    success = await test_gpu_stt_with_speech()
    
    logger.info("")
    logger.info("=" * 60)
    if success:
        logger.info("✅ GPU STT speech test completed successfully!")
    else:
        logger.error("❌ GPU STT speech test failed!")
    logger.info("=" * 60)
    
    return success

if __name__ == "__main__":
    import asyncio
    success = asyncio.run(main())
    sys.exit(0 if success else 1)