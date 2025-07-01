#!/usr/bin/env python3
"""
Test WhisperLive Integration

Simple test to verify WhisperLive STT engine works with FastRTC threading pipeline.
"""

import os
import sys
import asyncio
import numpy as np
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent / "backend" / "src"))

# Set environment for testing
os.environ['STT_BACKEND'] = 'whisper_live'
os.environ['USE_THREADING_PIPELINE'] = 'true'
os.environ['WHISPER_LIVE_HOST'] = 'localhost'
os.environ['WHISPER_LIVE_PORT'] = '9090'
os.environ['WHISPER_LIVE_MODEL'] = 'small'
os.environ['WHISPER_LIVE_AUTO_START'] = 'true'

async def test_whisper_live_engine():
    """Test WhisperLive STT engine directly."""
    print("🧪 Testing WhisperLive STT engine...")
    
    try:
        from audio.engines.stt.whisper_live_stt import WhisperLiveSTTEngine
        
        # Create engine
        engine = WhisperLiveSTTEngine(
            server_host='localhost',
            server_port=9090,
            model='small',
            language='en',
            auto_start_server=True
        )
        
        # Initialize
        print("🚀 Initializing WhisperLive engine...")
        if not await engine.initialize():
            print("❌ Failed to initialize WhisperLive engine")
            return False
        
        print("✅ WhisperLive engine initialized successfully")
        
        # Create test audio (1 second of sine wave at 440Hz)
        sample_rate = 16000
        duration = 1.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio_data = np.sin(2 * np.pi * 440 * t).astype(np.float32)
        
        print("🎤 Testing transcription with synthetic audio...")
        
        # Test transcription
        result = await engine._transcribe_audio(audio_data)
        print(f"📝 Transcription result: {result}")
        
        # Cleanup
        await engine.cleanup()
        print("🧹 Engine cleaned up")
        
        return True
        
    except Exception as e:
        print(f"❌ WhisperLive engine test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_whisper_live_worker():
    """Test WhisperLive streaming worker."""
    print("🧪 Testing WhisperLive streaming worker...")
    
    try:
        from core.pipeline_manager import AudioPipelineManager, AudioChunk
        from core.whisper_live_worker import WhisperLiveStreamingWorker
        
        # Create pipeline manager
        pipeline_manager = AudioPipelineManager(max_queue_size=10)
        pipeline_manager.start()
        
        # Create worker
        worker = WhisperLiveStreamingWorker(
            pipeline_manager=pipeline_manager,
            server_host='localhost',
            server_port=9090,
            model='small',
            vad_enabled=False,  # Disable VAD for testing
            auto_start_server=True
        )
        
        # Initialize worker
        print("🚀 Initializing WhisperLive worker...")
        if not await worker.initialize():
            print("❌ Failed to initialize WhisperLive worker")
            return False
        
        print("✅ WhisperLive worker initialized successfully")
        
        # Start worker
        worker.start_worker()
        
        # Create test audio chunk
        sample_rate = 16000
        duration = 2.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio_data = np.sin(2 * np.pi * 440 * t).astype(np.float32)
        
        audio_chunk = AudioChunk(
            generation_id="test_gen_001",
            audio_data=audio_data,
            timestamp=0.0,
            sample_rate=sample_rate
        )
        
        print("🎤 Processing test audio chunk...")
        
        # Process audio chunk
        success = await worker.process_audio_chunk(audio_chunk)
        print(f"📊 Processing result: {success}")
        
        # Wait a bit for processing
        await asyncio.sleep(3)
        
        # Check for results
        try:
            while not pipeline_manager.transcription_queue.empty():
                result = pipeline_manager.transcription_queue.get_nowait()
                print(f"📝 Transcription result: {result}")
        except:
            pass
        
        # Cleanup
        await worker.cleanup()
        worker.stop_worker()
        pipeline_manager.stop()
        
        print("🧹 Worker cleaned up")
        return True
        
    except Exception as e:
        print(f"❌ WhisperLive worker test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Run all tests."""
    print("🚀 Starting WhisperLive integration tests...")
    print("=" * 50)
    
    # Test 1: Engine directly
    print("\n📋 Test 1: WhisperLive STT Engine")
    engine_success = await test_whisper_live_engine()
    
    # Test 2: Worker integration
    print("\n📋 Test 2: WhisperLive Streaming Worker")
    worker_success = await test_whisper_live_worker()
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Results:")
    print(f"   Engine Test: {'✅ PASS' if engine_success else '❌ FAIL'}")
    print(f"   Worker Test: {'✅ PASS' if worker_success else '❌ FAIL'}")
    
    if engine_success and worker_success:
        print("\n🎉 All tests passed! WhisperLive integration is working.")
        return 0
    else:
        print("\n💥 Some tests failed. Check the output above for details.")
        return 1

if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n🛑 Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Test runner failed: {e}")
        sys.exit(1)