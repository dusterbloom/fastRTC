#!/usr/bin/env python3
"""
Test script to verify GPU functionality with FasterWhisperSTT
"""

import os
import sys
import time
import numpy as np

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend', 'src'))

from audio.engines.stt.faster_whisper_stt import FasterWhisperSTT
from core.interfaces import AudioData

def test_gpu_stt():
    """Test GPU functionality with FasterWhisperSTT."""
    print("🧪 Testing GPU functionality with FasterWhisperSTT...")
    
    try:
        # Initialize STT engine
        print("📝 Initializing FasterWhisperSTT...")
        stt = FasterWhisperSTT()
        
        if not stt.is_available():
            print("❌ STT engine is not available")
            return False
        
        # Create test audio (1 second of sine wave at 440Hz)
        sample_rate = 16000
        duration = 1.0
        frequency = 440.0
        
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        audio_samples = np.sin(2 * np.pi * frequency * t).astype(np.float32) * 0.1
        
        print(f"🔊 Created test audio: {audio_samples.shape} samples at {sample_rate}Hz")
        
        # Test transcription
        print("🎯 Testing transcription...")
        start_time = time.time()
        
        result = stt._transcribe_sync(audio_samples, target_language="en")
        
        elapsed = time.time() - start_time
        
        print(f"✅ Transcription completed in {elapsed:.3f}s")
        print(f"📝 Result: '{result.text}'")
        print(f"🌍 Language: {result.language}")
        print(f"📊 Confidence: {result.confidence:.3f}")
        
        # Cleanup
        stt.shutdown()
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing GPU STT: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_gpu_status():
    """Check GPU status and availability."""
    print("🔍 Checking GPU status...")
    
    try:
        import torch
        print(f"🐍 PyTorch version: {torch.__version__}")
        print(f"🚀 CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"📊 CUDA version: {torch.version.cuda}")
            print(f"🔢 GPU count: {torch.cuda.device_count()}")
            
            for i in range(torch.cuda.device_count()):
                gpu_name = torch.cuda.get_device_name(i)
                print(f"   GPU {i}: {gpu_name}")
                
            current_device = torch.cuda.current_device()
            print(f"🎯 Current device: {current_device}")
            
            # Test GPU memory
            try:
                device = torch.device('cuda')
                x = torch.randn(100, 100).to(device)
                print(f"✅ GPU memory test passed")
                del x
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"❌ GPU memory test failed: {e}")
        
    except ImportError:
        print("❌ PyTorch not available")
    
    try:
        import ctranslate2
        print(f"🔧 CTranslate2 version: {ctranslate2.__version__}")
        cuda_devices = ctranslate2.get_cuda_device_count()
        print(f"🚀 CTranslate2 CUDA devices: {cuda_devices}")
    except ImportError:
        print("❌ CTranslate2 not available")

if __name__ == "__main__":
    print("=" * 60)
    print("🧪 FastRTC GPU STT Test")
    print("=" * 60)
    
    # Check GPU status first
    check_gpu_status()
    print()
    
    # Test STT with GPU
    success = test_gpu_stt()
    
    print()
    print("=" * 60)
    if success:
        print("✅ GPU STT test completed successfully!")
    else:
        print("❌ GPU STT test failed!")
    print("=" * 60)