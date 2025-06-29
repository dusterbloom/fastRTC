#!/usr/bin/env python3
"""
Test GPU functionality with FasterWhisperSTT
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

def check_gpu_status():
    """Check GPU status and availability."""
    logger.info("🔍 Checking GPU status...")
    
    try:
        import torch
        logger.info(f"🐍 PyTorch version: {torch.__version__}")
        logger.info(f"🚀 CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            logger.info(f"📊 CUDA version: {torch.version.cuda}")
            logger.info(f"🔢 GPU count: {torch.cuda.device_count()}")
            
            for i in range(torch.cuda.device_count()):
                gpu_name = torch.cuda.get_device_name(i)
                logger.info(f"   GPU {i}: {gpu_name}")
                
            current_device = torch.cuda.current_device()
            logger.info(f"🎯 Current device: {current_device}")
            
            # Test GPU memory
            try:
                device = torch.device('cuda')
                x = torch.randn(100, 100).to(device)
                logger.info("✅ GPU memory test passed")
                del x
                torch.cuda.empty_cache()
                return True
            except Exception as e:
                logger.error(f"❌ GPU memory test failed: {e}")
                return False
        else:
            return False
        
    except ImportError:
        logger.error("❌ PyTorch not available")
        return False

async def test_gpu_stt():
    """Test GPU functionality with FasterWhisperSTT."""
    logger.info("🧪 Testing GPU functionality with FasterWhisperSTT...")
    
    try:
        # Import STT engine
        from src.audio.engines.stt.faster_whisper_stt import FasterWhisperSTT
        
        # Initialize STT engine
        logger.info("📝 Initializing FasterWhisperSTT...")
        stt = FasterWhisperSTT()
        
        if not stt.is_available():
            logger.error("❌ STT engine is not available")
            return False
        
        logger.info("✅ STT engine initialized successfully")
        logger.info(f"🔧 Device: {stt.device}")
        logger.info(f"🔧 Compute type: {stt.compute_type}")
        
        # Create test audio (1 second of sine wave at 440Hz)
        sample_rate = 16000
        duration = 1.0
        frequency = 440.0
        
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        audio_samples = np.sin(2 * np.pi * frequency * t).astype(np.float32) * 0.1
        
        logger.info(f"🔊 Created test audio: {audio_samples.shape} samples at {sample_rate}Hz")
        
        # Test transcription
        logger.info("🎯 Testing transcription...")
        start_time = time.time()
        
        result = stt._transcribe_sync(audio_samples, target_language="en")
        
        elapsed = time.time() - start_time
        
        logger.info(f"✅ Transcription completed in {elapsed:.3f}s")
        logger.info(f"📝 Result: '{result.text}'")
        logger.info(f"🌍 Language: {result.language}")
        logger.info(f"📊 Confidence: {result.confidence:.3f}")
        
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
    logger.info("🧪 FastRTC GPU STT Test")
    logger.info("=" * 60)
    
    # Check GPU status first
    gpu_available = check_gpu_status()
    logger.info("")
    
    # Test STT with GPU
    success = await test_gpu_stt()
    
    logger.info("")
    logger.info("=" * 60)
    logger.info("📊 Test Results:")
    logger.info(f"   GPU Available: {'✅' if gpu_available else '❌'}")
    logger.info(f"   STT GPU Test: {'✅' if success else '❌'}")
    
    if success:
        logger.info("✅ GPU STT test completed successfully!")
    else:
        logger.error("❌ GPU STT test failed!")
    logger.info("=" * 60)
    
    return success

if __name__ == "__main__":
    import asyncio
    success = asyncio.run(main())
    sys.exit(0 if success else 1)