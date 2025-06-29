#!/usr/bin/env python3
"""
Voice Authentication Hang Test
=============================

Specifically tests the voice authentication pipeline that was added recently
and is likely causing the hanging issue.
"""

import os
import sys
import time
import numpy as np
from pathlib import Path

# Set environment variables before imports
os.environ['STT_BACKEND'] = 'faster_gpu'
os.environ['PYTHONIOENCODING'] = 'utf-8'
os.environ['USE_THREADING_PIPELINE'] = 'true'
os.environ['LOG_LEVEL'] = 'DEBUG'

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.resolve()))

print("🔍 Voice Authentication Hang Test")
print("=" * 40)

def test_with_timeout(test_name, func, timeout_seconds=30):
    """Run a test with timeout to catch hangs"""
    print(f"\n🧪 Testing: {test_name}")
    start_time = time.time()
    
    try:
        import signal
        
        def timeout_handler(signum, frame):
            raise TimeoutError(f"Test timed out after {timeout_seconds}s")
        
        # Set timeout
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout_seconds)
        
        try:
            result = func()
            elapsed = time.time() - start_time
            print(f"✅ {test_name} - SUCCESS ({elapsed:.2f}s)")
            return result, True
        finally:
            signal.alarm(0)  # Cancel timeout
            
    except TimeoutError as e:
        elapsed = time.time() - start_time
        print(f"⏰ {test_name} - TIMEOUT ({elapsed:.2f}s): {e}")
        print("💥 THIS IS LIKELY WHERE THE PIPELINE HANGS!")
        return None, False
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"❌ {test_name} - FAILED ({elapsed:.2f}s): {e}")
        import traceback
        traceback.print_exc()
        return None, False

def test_resemblyzer_init():
    """Test resemblyzer initialization - known to be slow"""
    print("Initializing Resemblyzer...")
    from src.audio.voice_embeddings import VoiceEmbeddingManager
    manager = VoiceEmbeddingManager()
    print("Resemblyzer initialized")
    return manager

def test_voice_auth_manager():
    """Test voice authentication manager"""
    print("Creating SpokenUserIdentifier with voice auth...")
    from src.audio.user_identification import SpokenUserIdentifier
    
    # This is where voice auth gets enabled
    manager = SpokenUserIdentifier(enable_voice_auth=True)
    print(f"Voice auth enabled: {manager.enable_voice_auth}")
    
    if manager.enable_voice_auth:
        print("Testing audio buffer setting...")
        dummy_audio = np.random.randn(16000).astype(np.float32)
        manager.set_audio_buffer(dummy_audio)  # Call on manager itself, not voice_manager
        print("Audio buffer set successfully")
    
    return manager

def test_voice_auth_processing():
    """Test the actual voice auth processing that happens in the pipeline"""
    print("Testing voice authentication processing...")
    from src.audio.user_identification import SpokenUserIdentifier
    
    manager = SpokenUserIdentifier(enable_voice_auth=True)
    
    if not manager.enable_voice_auth:
        print("Voice auth disabled, skipping...")
        return manager
    
    # Simulate what happens in the threading callback handler
    dummy_audio = np.random.randn(16000).astype(np.float32)
    
    # This is the critical part that might hang
    print("Setting audio buffer (this might hang)...")
    if manager.enable_voice_auth:
        manager.set_audio_buffer(dummy_audio)  # Call on manager itself
        print("Audio buffer set")
        
            # Test voice authentication trigger
        print("Testing voice auth trigger...")
        test_text = "it's me"  # This should trigger voice auth
        result = manager.process_text(test_text)
        print(f"Voice auth result: {result}")
    
    return manager

def test_stt_with_voice_auth():
    """Test STT engine with voice auth context"""
    print("Testing STT engine in voice auth context...")
    from src.core.voice_assistant import VoiceAssistant
    from src.config.settings import load_config
    
    config = load_config()
    voice_assistant = VoiceAssistant(config=config)
    
    # Test STT transcription
    dummy_audio = np.random.randn(16000).astype(np.float32)
    print("Running STT transcription...")
    
    # This uses the async interface correctly
    import asyncio
    async def run_stt():
        result = await voice_assistant.stt_engine.transcribe(dummy_audio)
        return result
    
    result = asyncio.run(run_stt())
    print(f"STT result: {result}")
    return voice_assistant

def test_threading_pipeline_with_voice_auth():
    """Test the full threading pipeline with voice auth"""
    print("Testing threading pipeline with voice auth...")
    from src.integration.threading_callback_handler import ThreadingCallbackHandler
    from src.core.voice_assistant import VoiceAssistant
    from src.config.settings import load_config
    
    config = load_config()
    voice_assistant = VoiceAssistant(config=config)
    
    handler = ThreadingCallbackHandler(
        voice_assistant=voice_assistant,
        stt_engine=voice_assistant.stt_engine,
        tts_engine=voice_assistant.tts_engine,
        voice_mapper=voice_assistant.voice_mapper
    )
    
    print("Starting threading handler...")
    handler.start()
    
    try:
        # Test audio processing - this is where it likely hangs
        dummy_audio = np.random.randn(16000).astype(np.float32)
        audio_tuple = (16000, dummy_audio)
        
        print("Processing audio (this might hang)...")
        result_generator = handler.process_audio_stream(audio_tuple)
        
        # Try to get first result
        results = []
        for i, result in enumerate(result_generator):
            results.append(result)
            print(f"Got result {i+1}: {type(result)}")
            if i >= 1:  # Get a couple results then stop
                break
        
        print(f"Threading pipeline completed with {len(results)} results")
        return handler
        
    finally:
        print("Stopping threading handler...")
        handler.stop()

def main():
    """Run targeted voice auth hang tests"""
    print("🚀 Starting Voice Authentication Hang Tests")
    
    tests = [
        ("Resemblyzer Init", test_resemblyzer_init, 60),  # Longer timeout for model loading
        ("Voice Auth Manager", test_voice_auth_manager, 30),
        ("Voice Auth Processing", test_voice_auth_processing, 30),
        ("STT with Voice Auth", test_stt_with_voice_auth, 45),
        ("Threading Pipeline + Voice Auth", test_threading_pipeline_with_voice_auth, 60),
    ]
    
    results = {}
    
    for test_name, test_func, timeout in tests:
        result, success = test_with_timeout(test_name, test_func, timeout)
        results[test_name] = {"success": success, "result": result}
        
        if not success:
            print(f"\n💥 HANG DETECTED: {test_name}")
            print("This is where your pipeline is getting stuck!")
            
            if "Resemblyzer" in test_name:
                print("🔧 FIX: Resemblyzer is hanging. Try disabling voice auth or using CPU mode.")
            elif "Voice Auth" in test_name:
                print("🔧 FIX: Voice authentication is hanging. Disable voice auth temporarily.")
            elif "Threading" in test_name:
                print("🔧 FIX: Threading pipeline hangs with voice auth. Use async pipeline instead.")
            
            break
        
        time.sleep(2)  # Brief pause between tests
    
    print("\n" + "=" * 40)
    print("🏁 Voice Auth Hang Test Summary:")
    for test_name, result in results.items():
        status = "✅ PASS" if result["success"] else "❌ HANG"
        print(f"  {status} {test_name}")
    
    if all(r["success"] for r in results.values()):
        print("\n✅ All tests passed! Voice auth is working correctly.")
        print("💡 The hang might be in a different part of the pipeline.")
    else:
        print(f"\n❌ Voice authentication is causing hangs!")
        print("💡 Disable voice auth to fix the pipeline:")
        print("   1. Set enable_voice_auth=False in UserIdentificationManager")
        print("   2. Or use async pipeline: ./fastrtc.sh dev (without --threading)")

if __name__ == "__main__":
    main()