#!/usr/bin/env python3
"""
Test Real LLM Integration and Audio Production
Verify actual LLM responses and Kokoro audio output
"""

import os
import sys
import numpy as np
import asyncio
import time
import wave
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent / "backend"))

# Set environment for complete pipeline
os.environ['STT_BACKEND'] = 'faster_gpu'
os.environ['USE_THREADING_PIPELINE'] = 'true'
os.environ['LOG_LEVEL'] = 'INFO'

from backend.src.utils.logging import get_logger
from backend.src.core.stt_worker import STTStreamingWorker
from backend.src.core.llm_worker import LLMStreamingWorker
from backend.src.core.tts_worker import TTSStreamingWorker, TTSOutputWorker
from backend.src.core.pipeline_manager import AudioPipelineManager, AudioChunk
from backend.src.audio.engines.stt import STTEngine
from backend.src.audio.engines.tts.kokoro_tts import KokoroTTSEngine
from backend.src.audio import VoiceMapper

logger = get_logger(__name__)

class MockVoiceAssistant:
    """Mock voice assistant for testing."""
    def __init__(self, generation_id):
        self.current_generation_id = generation_id
        self.voice_print_manager = None
        
    def get_current_generation_id(self):
        return self.current_generation_id

async def test_real_llm_and_audio():
    """Test real LLM responses and actual audio production."""
    logger.info("🚀 Testing real LLM integration and audio production...")
    
    try:
        # Initialize pipeline manager
        logger.info("📝 Initializing pipeline manager...")
        pipeline_manager = AudioPipelineManager()
        generation_id = pipeline_manager.create_generation()
        logger.info(f"📝 Created generation {generation_id}")
        
        # Initialize engines
        logger.info("📝 Initializing engines...")
        stt_engine = STTEngine()
        tts_engine = KokoroTTSEngine()
        voice_mapper = VoiceMapper()
        mock_voice_assistant = MockVoiceAssistant(generation_id)
        
        if not stt_engine.is_available():
            logger.error("❌ STT engine not available")
            return False
        
        # Initialize workers
        logger.info("📝 Initializing workers...")
        
        stt_worker = STTStreamingWorker(
            pipeline_manager=pipeline_manager,
            stt_engine=stt_engine,
            confidence_threshold=0.1,
            min_audio_length=0.1
        )
        
        llm_worker = LLMStreamingWorker(
            pipeline_manager=pipeline_manager,
            voice_assistant=mock_voice_assistant
        )
        
        tts_worker = TTSStreamingWorker(
            pipeline_manager=pipeline_manager,
            tts_engine=tts_engine,
            voice_mapper=voice_mapper,
            voice_assistant=mock_voice_assistant
        )
        
        # Collect audio output
        output_audio_chunks = []
        def audio_callback(audio_chunk):
            output_audio_chunks.append(audio_chunk)
            logger.info(f"📢 Received audio chunk: {type(audio_chunk)}")
        
        tts_output_worker = TTSOutputWorker(
            pipeline_manager=pipeline_manager,
            output_callback=audio_callback
        )
        
        # Load test audio
        logger.info("📁 Loading test audio...")
        import librosa
        audio_file = "/mnt/c/Users/PC/Dev/fastRTC/backend/tests/samples/audio_en.wav"
        audio_data, sr = librosa.load(audio_file, sr=None)
        
        if audio_data.ndim == 1:
            audio_data = audio_data.reshape(1, -1)
        
        logger.info(f"🎵 Loaded audio: {audio_data.shape}, {audio_data.dtype}, {sr}Hz")
        
        # Step 1: STT Processing
        logger.info("🎤 Step 1: STT Processing...")
        audio_chunk = AudioChunk(
            generation_id=generation_id,
            audio_data=audio_data,
            sample_rate=sr,
            timestamp=time.time(),
            is_final=True
        )
        
        stt_result = await stt_worker.process_item_async(audio_chunk)
        if not stt_result:
            logger.error("❌ STT failed")
            return False
            
        logger.info(f"✅ STT Result: '{stt_result.text}' (confidence: {stt_result.confidence:.3f})")
        
        # Step 2: Real LLM Processing
        logger.info("🧠 Step 2: Real LLM Processing...")
        
        # Start LLM worker in background
        llm_worker.start_worker()
        
        # Put STT result into transcription queue for LLM
        pipeline_manager.transcription_queue.put(stt_result)
        logger.info("📤 STT result sent to LLM worker")
        
        # Wait for LLM tokens to appear
        logger.info("⏳ Waiting for LLM tokens...")
        llm_tokens = []
        start_time = time.time()
        
        while time.time() - start_time < 30.0:  # Wait up to 30 seconds
            if not pipeline_manager.llm_token_queue.empty():
                try:
                    token_chunk = pipeline_manager.llm_token_queue.get(timeout=1.0)
                    llm_tokens.append(token_chunk)
                    logger.info(f"📥 Received LLM token: '{token_chunk.text.strip()}'")
                    
                    # Process through TTS immediately
                    await tts_worker.process_item_async(token_chunk)
                    
                    if token_chunk.is_sentence_complete:
                        logger.info("✅ Complete sentence received from LLM")
                        break
                        
                except Exception as e:
                    logger.warning(f"Error getting LLM token: {e}")
                    
            await asyncio.sleep(0.1)
        
        # Step 3: Process TTS Output
        logger.info("🔊 Step 3: Processing TTS Output...")
        
        # Process any TTS chunks in output queue
        processed_outputs = 0
        while not pipeline_manager.output_queue.empty() and processed_outputs < 100:
            try:
                tts_chunk = pipeline_manager.output_queue.get(timeout=1.0)
                tts_output_worker.process_item(tts_chunk)
                processed_outputs += 1
            except Exception as e:
                logger.warning(f"Error processing TTS output: {e}")
                break
        
        # Stop LLM worker
        llm_worker.stop_worker()
        
        # Step 4: Save Audio Output
        logger.info("💾 Step 4: Saving Audio Output...")
        
        if output_audio_chunks:
            # Combine all audio chunks
            all_audio_data = []
            sample_rate = 22050  # Kokoro default
            
            for chunk in output_audio_chunks:
                if hasattr(chunk, 'audio_data'):
                    all_audio_data.append(chunk.audio_data)
                    if hasattr(chunk, 'sample_rate'):
                        sample_rate = chunk.sample_rate
                elif isinstance(chunk, np.ndarray):
                    all_audio_data.append(chunk)
            
            if all_audio_data:
                # Concatenate all audio
                combined_audio = np.concatenate(all_audio_data)
                
                # Save as WAV file
                output_file = "/mnt/c/Users/PC/Dev/fastRTC/test_kokoro_output.wav"
                
                # Ensure audio is in correct format for WAV
                if combined_audio.dtype != np.int16:
                    # Convert float to int16
                    combined_audio = (combined_audio * 32767).astype(np.int16)
                
                with wave.open(output_file, 'wb') as wav_file:
                    wav_file.setnchannels(1)  # Mono
                    wav_file.setsampwidth(2)  # 16-bit
                    wav_file.setframerate(sample_rate)
                    wav_file.writeframes(combined_audio.tobytes())
                
                logger.info(f"💾 Saved audio to: {output_file}")
                logger.info(f"   Duration: {len(combined_audio) / sample_rate:.2f}s")
                logger.info(f"   Sample rate: {sample_rate}Hz")
                logger.info(f"   Samples: {len(combined_audio)}")
        
        # Results summary
        logger.info("📊 Real LLM and Audio Test Results:")
        logger.info(f"   STT: '{stt_result.text}' (confidence: {stt_result.confidence:.3f})")
        logger.info(f"   LLM tokens: {len(llm_tokens)}")
        logger.info(f"   TTS outputs: {processed_outputs}")
        logger.info(f"   Audio chunks: {len(output_audio_chunks)}")
        
        # Cleanup
        stt_engine.shutdown()
        logger.info("🧹 Cleanup complete")
        
        success = (
            stt_result is not None and
            len(llm_tokens) > 0 and
            len(output_audio_chunks) > 0
        )
        
        if success:
            logger.info("🎉 Real LLM and audio test PASSED!")
        else:
            logger.error("❌ Real LLM and audio test FAILED!")
            
        return success
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        logger.error(f"❌ Traceback: {traceback.format_exc()}")
        return False

if __name__ == "__main__":
    success = asyncio.run(test_real_llm_and_audio())
    if success:
        print("✅ Real LLM and audio test passed!")
        sys.exit(0)
    else:
        print("❌ Real LLM and audio test failed!")
        sys.exit(1)