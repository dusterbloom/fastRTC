#!/usr/bin/env python3
"""
Simple test script to verify GPU functionality
"""

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
                return True
            except Exception as e:
                print(f"❌ GPU memory test failed: {e}")
                return False
        else:
            return False
        
    except ImportError:
        print("❌ PyTorch not available")
        return False

def test_faster_whisper_gpu():
    """Test faster-whisper with GPU."""
    print("\n🧪 Testing faster-whisper with GPU...")
    
    try:
        from faster_whisper import WhisperModel
        import numpy as np
        import time
        
        # Test with CPU first
        print("📝 Testing CPU model...")
        start_time = time.time()
        model_cpu = WhisperModel("tiny", device="cpu", compute_type="int8")
        cpu_load_time = time.time() - start_time
        print(f"✅ CPU model loaded in {cpu_load_time:.2f}s")
        
        # Test with GPU if available
        gpu_available = check_gpu_status()
        if gpu_available:
            print("\n📝 Testing GPU model...")
            start_time = time.time()
            model_gpu = WhisperModel("tiny", device="cuda", compute_type="int8_float16")
            gpu_load_time = time.time() - start_time
            print(f"✅ GPU model loaded in {gpu_load_time:.2f}s")
            
            # Create test audio
            sample_rate = 16000
            duration = 1.0
            t = np.linspace(0, duration, int(sample_rate * duration), False)
            audio = np.sin(2 * np.pi * 440 * t).astype(np.float32) * 0.1
            
            # Test transcription on GPU
            print("🎯 Testing GPU transcription...")
            start_time = time.time()
            segments, info = model_gpu.transcribe(audio, language="en")
            text = " ".join([segment.text for segment in segments])
            gpu_transcribe_time = time.time() - start_time
            print(f"✅ GPU transcription completed in {gpu_transcribe_time:.3f}s")
            print(f"📝 Result: '{text.strip()}'")
            
            return True
        else:
            print("⚠️ GPU not available for faster-whisper test")
            return False
            
    except ImportError as e:
        print(f"❌ faster-whisper not available: {e}")
        return False
    except Exception as e:
        print(f"❌ Error testing faster-whisper: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("🧪 FastRTC GPU Test")
    print("=" * 60)
    
    # Check basic GPU status
    gpu_available = check_gpu_status()
    
    # Test faster-whisper with GPU
    whisper_gpu_success = test_faster_whisper_gpu()
    
    print("\n" + "=" * 60)
    print("📊 Test Results:")
    print(f"   GPU Available: {'✅' if gpu_available else '❌'}")
    print(f"   Faster-Whisper GPU: {'✅' if whisper_gpu_success else '❌'}")
    print("=" * 60)