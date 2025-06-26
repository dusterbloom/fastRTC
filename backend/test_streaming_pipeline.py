#!/usr/bin/env python3
"""
Test script for the streaming pipeline implementation.
Quick validation of STT→LLM→TTS streaming functionality.
"""

import asyncio
import numpy as np
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

async def test_streaming_pipeline():
    """Test the basic streaming pipeline functionality."""
    print("🚀 Testing Streaming Pipeline...")
    
    try:
        # Import core components
        from src.core.voice_assistant import VoiceAssistant
        from src.integration.streaming_callback_handler import StreamingCallbackHandler, StreamingPipeline
        from src.audio import STTEngine, KokoroTTSEngine, VoiceMapper
        
        print("✅ Successfully imported streaming components")
        
        # Create voice assistant
        print("🧠 Creating voice assistant...")
        voice_assistant = VoiceAssistant()
        
        # Initialize async components
        print("⚡ Initializing async components...")
        await voice_assistant.initialize_async()
        
        # Create streaming pipeline components
        print("🎤 Creating streaming pipeline...")
        stt_engine = voice_assistant.stt_engine
        tts_engine = voice_assistant.tts_engine  
        voice_mapper = voice_assistant.voice_mapper
        
        # Test streaming pipeline
        streaming_pipeline = StreamingPipeline(
            voice_assistant, stt_engine, tts_engine, voice_mapper
        )
        
        print("✅ Streaming pipeline created successfully")
        
        # Test LLM streaming
        print("🤖 Testing LLM streaming...")
        test_text = "Hello, this is a test of streaming."
        token_count = 0
        
        async for token in voice_assistant.stream_llm_response_smart(test_text):
            token_count += 1
            if token_count <= 5:  # Show first few tokens
                print(f"Token {token_count}: '{token}'")
        
        print(f"✅ LLM streaming test completed. Received {token_count} tokens.")
        
        # Test TTS streaming  
        print("🔊 Testing TTS streaming...")
        if tts_engine.is_available():
            current_language = voice_assistant.current_language
            available_voices = voice_mapper.get_voices_for_language(current_language)
            voice_id = available_voices[0] if available_voices else None
            
            test_sentence = "This is a test sentence for TTS streaming."
            chunk_count = 0
            
            async for sample_rate, audio_chunk in tts_engine.stream_synthesis_async(
                test_sentence, voice_id, current_language
            ):
                chunk_count += 1
                if chunk_count <= 3:  # Show first few chunks
                    print(f"TTS Chunk {chunk_count}: {sample_rate}Hz, {audio_chunk.size} samples")
            
            print(f"✅ TTS streaming test completed. Received {chunk_count} audio chunks.")
        else:
            print("⚠️ TTS engine not available for streaming test")
        
        # Test full streaming callback handler
        print("🎛️ Testing streaming callback handler...")
        event_loop = asyncio.get_event_loop()
        
        callback_handler = StreamingCallbackHandler(
            voice_assistant, stt_engine, tts_engine, voice_mapper, event_loop
        )
        
        # Create test audio data (silence)
        sample_rate = 16000
        duration = 1.0  # 1 second
        samples = int(sample_rate * duration)
        test_audio = np.zeros(samples, dtype=np.float32)
        
        print(f"📊 Testing with {samples} samples at {sample_rate}Hz")
        
        # Test audio parsing
        parsed_sr, parsed_audio = callback_handler._parse_audio_data((sample_rate, test_audio))
        print(f"✅ Audio parsing: {parsed_sr}Hz, {parsed_audio.size if parsed_audio is not None else 0} samples")
        
        # Test audio preprocessing  
        if parsed_audio is not None:
            processed_audio = callback_handler._preprocess_audio(parsed_audio, parsed_sr)
            print(f"✅ Audio preprocessing: {processed_audio.size} samples")
        
        # Get handler stats
        stats = callback_handler.get_stats()
        print(f"📊 Handler stats: {stats}")
        
        print("🎉 All streaming pipeline tests completed successfully!")
        
        # Cleanup
        await voice_assistant.cleanup_async()
        print("🧹 Cleanup completed")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure all dependencies are installed")
        return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

async def main():
    """Main test function."""
    print("=" * 60)
    print("🧪 FastRTC Streaming Pipeline Test")
    print("=" * 60)
    
    success = await test_streaming_pipeline()
    
    if success:
        print("\n✅ All tests passed! Streaming pipeline is ready.")
        return 0
    else:
        print("\n❌ Tests failed. Check the output above for details.")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)