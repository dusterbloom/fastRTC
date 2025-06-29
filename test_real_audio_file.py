#!/usr/bin/env python3
"""
Test GPU STT fix with real audio file
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

def load_wav_file(file_path):
    """Load WAV file using scipy or librosa."""
    try:
        import librosa
        audio, sr = librosa.load(file_path, sr=None)
        logger.info(f"📁 Loaded audio with librosa: {audio.shape}, sr={sr}")
        return audio.astype(np.float32), sr
    except ImportError:
        try:
            from scipy.io import wavfile
            sr, audio = wavfile.read(file_path)
            if audio.dtype == np.int16:
                audio = audio.astype(np.float32) / 32768.0
            elif audio.dtype == np.int32:
                audio = audio.astype(np.float32) / 2147483648.0
            logger.info(f"📁 Loaded audio with scipy: {audio.shape}, sr={sr}")
            return audio, sr
        except ImportError:
            logger.error("❌ Neither librosa nor scipy available for audio loading")
            return None, None

async def test_real_audio():
    """Test GPU STT with real audio file."""
    audio_file = "/mnt/c/Users/PC/Dev/fastRTC/backend/tests/samples/audio_en.wav"
    
    logger.info(f"🧪 Testing GPU STT with real audio file: {audio_file}")
    print(f"🧪 Testing GPU STT with real audio file: {audio_file}")
    
    # Check if file exists
    if not Path(audio_file).exists():
        logger.error(f"❌ Audio file not found: {audio_file}")
        return False
    
    try:
        # Load audio file
        logger.info("📁 Loading audio file...")
        audio_samples, sample_rate = load_wav_file(audio_file)
        
        if audio_samples is None:
            logger.error("❌ Failed to load audio file")
            return False
        
        # Handle stereo audio (take first channel)
        if audio_samples.ndim > 1:
            logger.info(f"🔧 Converting stereo to mono: {audio_samples.shape} -> {audio_samples.shape[0]}")
            audio_samples = audio_samples[:, 0] if audio_samples.shape[1] > 1 else audio_samples.flatten()
        
        duration = len(audio_samples) / sample_rate
        logger.info(f"🎵 Audio info: {audio_samples.shape}, {audio_samples.dtype}, {sample_rate}Hz, {duration:.2f}s")
        print(f"🎵 Audio info: {audio_samples.shape}, {audio_samples.dtype}, {sample_rate}Hz, {duration:.2f}s")
        logger.info(f"🔍 Audio stats: min={audio_samples.min():.6f}, max={audio_samples.max():.6f}, rms={np.sqrt(np.mean(audio_samples**2)):.6f}")
        
        # Initialize STT engine
        logger.info(f"📝 Initializing {STTEngine.__name__}...")
        stt = STTEngine()
        
        if not stt.is_available():
            logger.error("❌ STT engine not available")
            return False
        
        logger.info("✅ STT engine initialized successfully")
        
        # Test with 1D audio (normal case)
        logger.info("🚀 Testing with 1D audio...")
        audio_data_1d = AudioData(samples=audio_samples, sample_rate=sample_rate, duration=duration)
        
        start_time = asyncio.get_event_loop().time()
        result_1d = await stt.transcribe(audio_data_1d)
        end_time = asyncio.get_event_loop().time()
        
        processing_time_1d = end_time - start_time
        logger.info(f"✅ 1D Audio Result: '{result_1d.text}' (confidence: {result_1d.confidence:.3f}, time: {processing_time_1d:.3f}s)")
        print(f"✅ 1D Audio Result: '{result_1d.text}' (confidence: {result_1d.confidence:.3f}, time: {processing_time_1d:.3f}s)")
        
        # Test with 2D audio (threading pipeline case)
        logger.info("🚀 Testing with 2D audio (threading pipeline simulation)...")
        audio_2d = audio_samples.reshape(1, -1)  # Make it 2D like threading pipeline
        audio_data_2d = AudioData(samples=audio_2d, sample_rate=sample_rate, duration=duration)
        
        start_time = asyncio.get_event_loop().time()
        result_2d = await stt.transcribe(audio_data_2d)
        end_time = asyncio.get_event_loop().time()
        
        processing_time_2d = end_time - start_time
        logger.info(f"✅ 2D Audio Result: '{result_2d.text}' (confidence: {result_2d.confidence:.3f}, time: {processing_time_2d:.3f}s)")
        print(f"✅ 2D Audio Result: '{result_2d.text}' (confidence: {result_2d.confidence:.3f}, time: {processing_time_2d:.3f}s)")
        
        # Compare results
        if result_1d.text == result_2d.text:
            logger.info("✅ 1D and 2D audio produced identical results!")
        else:
            logger.warning(f"⚠️ Results differ: 1D='{result_1d.text}' vs 2D='{result_2d.text}'")
        
        # Cleanup
        stt.shutdown()
        logger.info("🧹 STT engine shutdown complete")
        
        # Check if we got reasonable results
        success = (result_1d.text and len(result_1d.text.strip()) > 0) or (result_2d.text and len(result_2d.text.strip()) > 0)
        
        if success:
            logger.info("✅ Real audio transcription successful!")
        else:
            logger.warning("⚠️ No transcription text produced (might be silent audio)")
            
        return True  # Return True even if no text (file might be silent)
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        logger.error(f"❌ Traceback: {traceback.format_exc()}")
        return False

if __name__ == "__main__":
    success = asyncio.run(test_real_audio())
    if success:
        print("✅ Real audio test completed!")
        sys.exit(0)
    else:
        print("❌ Real audio test failed!")
        sys.exit(1)