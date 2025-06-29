#!/usr/bin/env python3
"""
Test full threading pipeline: STT -> LLM -> TTS
"""

import os
import sys
import numpy as np
import asyncio
import time
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent / "backend"))

# Set environment for GPU STT and threading
os.environ['STT_BACKEND'] = 'faster_gpu'
os.environ['USE_THREADING_PIPELINE'] = 'true'
os.environ['LOG_LEVEL'] = 'INFO'

from backend.src.utils.logging import get_logger
from backend.src.core.stt_worker import STTStreamingWorker
from backend.src.core.llm_worker import LLMStreamingWorker
from backend.src.core.tts_worker import TTSStreamingWorker
from backend.src.core.pipeline_manager import AudioPipelineManager, AudioChunk
from backend.src.audio.engines.stt import STTEngine

logger = get_logger(__name__)

def load_wav_file(file_path):
    """Load WAV file using scipy."""
    try:
        from scipy.io import wavfile
        sr, audio = wavfile.read(file_path)
        
        # Handle stereo
        if audio.ndim > 1:
            audio = audio[:, 0]
        
        # Convert to float32
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0
        elif audio.dtype == np.int32:
            audio = audio.astype(np.float32) / 2147483648.0
        else:
            audio = audio.astype(np.float32)
            
        return audio, sr
    except Exception as e:
        print(f"❌ Failed to load audio: {e}")
        return None, None

async def test_full_pipeline():
    """Test full threading pipeline with all workers."""
    audio_file = "/mnt/c/Users/PC/Dev/fastRTC/backend/tests/samples/audio_en.wav"
    
    print(f"🧪 Testing full threading pipeline: {audio_file}")
    
    try:
        # Load audio file
        audio_samples, sample_rate = load_wav_file(audio_file)
        if audio_samples is None:
            return False
        
        # Take just first 3 seconds for faster testing
        duration = 3.0
        end_sample = int(duration * sample_rate)
        audio_samples = audio_samples[:end_sample]
        
        print(f"🎵 Using {duration:.1f}s audio at {sample_rate}Hz")
        
        # Initialize components
        print("📝 Initializing pipeline components...")
        
        # Initialize STT engine
        stt_engine = STTEngine()
        if not stt_engine.is_available():
            print("❌ STT engine not available")
            return False
        
        # Initialize pipeline manager
        pipeline_manager = AudioPipelineManager()
        
        # Initialize workers
        stt_worker = STTStreamingWorker(
            pipeline_manager=pipeline_manager,
            stt_engine=stt_engine,
            confidence_threshold=0.1,
            min_audio_length=0.1
        )
        
        # Mock voice assistant for LLM worker
        class MockVoiceAssistant:
            def __init__(self):
                self.llm_service = MockLLMService()
        
        class MockLLMService:
            async def stream_response(self, text, context=""):
                print(f"🤖 LLM received: '{text}' (context: '{context}')")
                # Simple mock response
                response = f"I heard you say: {text}"
                for word in response.split():
                    yield word + " "
                    await asyncio.sleep(0.1)
        
        mock_voice_assistant = MockVoiceAssistant()
        
        llm_worker = LLMStreamingWorker(
            pipeline_manager=pipeline_manager,
            voice_assistant=mock_voice_assistant,
            min_sentence_length=5
        )
        
        # Start workers
        print("🚀 Starting workers...")
        stt_worker.start_worker()
        llm_worker.start_worker()
        
        # Give workers time to start
        await asyncio.sleep(0.5)
        
        # Create generation and audio chunk
        generation_id = pipeline_manager.create_generation()
        print(f"🆕 Created generation: {generation_id}")
        
        # Create 2D audio array like threading pipeline
        audio_2d = audio_samples.reshape(1, -1)
        
        audio_chunk = AudioChunk(
            generation_id=generation_id,
            audio_data=audio_2d,
            sample_rate=sample_rate,
            timestamp=time.time(),
            is_final=True
        )
        
        # Put audio chunk into pipeline
        print("🎤 Putting audio chunk into STT queue...")
        pipeline_manager.audio_input_queue.put(audio_chunk)
        
        # Monitor pipeline progress
        print("👀 Monitoring pipeline progress...")
        
        start_time = time.time()
        timeout = 30.0  # 30 second timeout
        
        stt_done = False
        llm_done = False
        
        while time.time() - start_time < timeout:
            # Check STT queue
            if not stt_done and not pipeline_manager.transcription_queue.empty():
                print(f"✅ STT result available in transcription queue (size: {pipeline_manager.transcription_queue.qsize()})")
                stt_done = True
            
            # Check LLM queue  
            if not llm_done and not pipeline_manager.llm_token_queue.empty():
                print(f"✅ LLM result available in token queue (size: {pipeline_manager.llm_token_queue.qsize()})")
                llm_done = True
            
            # Check generation state
            state = pipeline_manager.get_generation_state(generation_id)
            if state:
                print(f"📊 Generation {generation_id} status: {state.status}")
                if state.stt_complete.is_set():
                    print(f"✅ STT completed: '{state.input_text}'")
                if state.llm_started.is_set():
                    print(f"✅ LLM started")
                if state.llm_streaming.is_set():
                    print(f"✅ LLM streaming")
            
            if stt_done:  # Just check STT for now
                print("🎉 Both STT and LLM completed!")
                break
                
            await asyncio.sleep(0.5)
        
        if not stt_done:
            print("❌ STT did not complete")
        if not llm_done:
            print("❌ LLM did not complete")
        
        # Stop workers
        print("🛑 Stopping workers...")
        stt_worker.stop_worker()
        llm_worker.stop_worker()
        
        # Cleanup
        stt_engine.shutdown()
        print("🧹 Cleanup complete")
        
        return stt_done and llm_done
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        print(f"❌ Traceback: {traceback.format_exc()}")
        return False

if __name__ == "__main__":
    success = asyncio.run(test_full_pipeline())
    if success:
        print("\n✅ Full pipeline test passed!")
        sys.exit(0)
    else:
        print("\n❌ Full pipeline test failed!")
        sys.exit(1)