#!/usr/bin/env python3
"""
Test STT engine with real audio file to debug empty transcription issue
"""

import os
import sys
import time
import logging
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.audio import STTEngine

# Setup logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_stt_with_audio():
    """Test STT engine with the provided audio file"""
    
    audio_file = "/mnt/c/Users/PC/Dev/fastRTC/backend/tests/samples/audio_en.wav"
    
    if not os.path.exists(audio_file):
        logger.error(f"Audio file not found: {audio_file}")
        return False
    
    logger.info(f"Testing STT with audio file: {audio_file}")
    logger.info(f"File size: {os.path.getsize(audio_file)} bytes")
    
    try:
        # Initialize STT engine
        logger.info("Initializing STT engine...")
        stt_engine = STTEngine()
        logger.info("STT engine initialized successfully")
        
        # Load audio file using FastRTC utilities
        logger.info("Loading audio file using FastRTC utilities...")
        from tests.fixtures.audio_samples import load_audio_file
        audio_data_obj = load_audio_file(audio_file)
        
        logger.info(f"Loaded audio: {audio_data_obj.samples.shape} samples at {audio_data_obj.sample_rate}Hz")
        logger.info(f"Duration: {audio_data_obj.duration:.2f}s, dtype: {audio_data_obj.samples.dtype}")
        
        # Convert stereo to mono if needed
        audio_samples = audio_data_obj.samples
        if len(audio_samples.shape) > 1:
            audio_samples = np.mean(audio_samples, axis=1)
            logger.info(f"Converted stereo to mono: shape={audio_samples.shape}")
        
        # Note: Skipping resampling for now to avoid librosa compilation hang
        # Whisper can handle different sample rates, though 16kHz is optimal
        sample_rate = audio_data_obj.sample_rate
        logger.info(f"Using original sample rate: {sample_rate}Hz (Whisper can handle this)")
        
        logger.info(f"Final audio data: {audio_samples.shape} samples at {sample_rate}Hz")
        
        # Process audio
        logger.info("Processing audio with STT...")
        start_time = time.time()
        result = await stt_engine.transcribe(audio_samples)
        end_time = time.time()
        
        logger.info(f"STT processing took {end_time - start_time:.2f} seconds")
        logger.info(f"STT result: '{result}'")
        logger.info(f"Result type: {type(result)}")
        
        if result:
            if hasattr(result, 'text'):
                logger.info(f"STT result text: '{result.text}'")
                logger.info(f"STT result confidence: {getattr(result, 'confidence', 'N/A')}")
                if result.text and result.text.strip():
                    logger.info("✅ STT successfully transcribed audio")
                    return True
                else:
                    logger.warning("⚠️ STT returned empty text")
                    return False
            else:
                logger.info(f"STT result content: '{result}'")
                if result and str(result).strip():
                    logger.info("✅ STT successfully transcribed audio")
                    return True
                else:
                    logger.warning("⚠️ STT returned empty result")
                    return False
        else:
            logger.warning("⚠️ STT returned None")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error testing STT: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_audio_file_properties():
    """Check audio file properties"""
    audio_file = "/mnt/c/Users/PC/Dev/fastRTC/backend/tests/samples/audio_en.wav"
    
    try:
        import wave
        with wave.open(audio_file, 'rb') as wav_file:
            frames = wav_file.getnframes()
            sample_rate = wav_file.getframerate()
            channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            duration = frames / sample_rate
            
            logger.info("Audio file properties:")
            logger.info(f"  Duration: {duration:.2f} seconds")
            logger.info(f"  Sample rate: {sample_rate} Hz")
            logger.info(f"  Channels: {channels}")
            logger.info(f"  Sample width: {sample_width} bytes")
            logger.info(f"  Total frames: {frames}")
            
    except Exception as e:
        logger.error(f"Error reading audio properties: {e}")

async def main():
    """Main async function"""
    logger.info("=== STT Audio Test ===")
    
    # Test audio file properties first
    test_audio_file_properties()
    
    # Test STT processing
    success = await test_stt_with_audio()
    
    if success:
        logger.info("🎉 STT test completed successfully")
        return True
    else:
        logger.error("💥 STT test failed")
        return False

if __name__ == "__main__":
    import asyncio
    success = asyncio.run(main())
    sys.exit(0 if success else 1)