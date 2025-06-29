#!/usr/bin/env python3
"""
Complete End-to-End Threading Pipeline Test
STT → LLM → TTS → Audio Output
"""

import os
import sys
import numpy as np
import asyncio
import time
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent / "backend"))

# Set environment for complete pipeline
os.environ['STT_BACKEND'] = 'faster_gpu'
os.environ['USE_THREADING_PIPELINE'] = 'true'
os.environ['LOG_LEVEL'] = 'INFO'

from backend.src.utils.logging import get_logger
from backend.src.core.stt_worker import STTStreamingWorker
from backend.src.core.tts_worker import TTSStreamingWorker, TTSOutputWorker
from backend.src.core.pipeline_manager import AudioPipelineManager, AudioChunk, LLMTokenChunk
from backend.src.audio.engines.stt import STTEngine
from backend.src.audio.engines.tts.kokoro_tts import KokoroTTSEngine
from backend.src.audio import VoiceMapper

logger = get_logger(__name__)

class MockVoiceAssistant:
    """Mock voice assistant for testing."""
    def __init__(self, generation_id):
        self.current_generation_id = generation_id
        
    def get_current_generation_id(self):
        return self.current_generation_id

async def test_complete_pipeline():
    """Test complete STT → LLM → TTS → Audio pipeline."""
    logger.info("🚀 Testing complete threading pipeline...")
    
    try:
        # Initialize pipeline manager
        logger.info("📝 Initializing pipeline manager...")
        pipeline_manager = AudioPipelineManager()
        
        # Create generation state for our test
        from backend.src.core.pipeline_manager import GenerationStatus
        generation_id = pipeline_manager.create_generation()
        logger.info(f"📝 Created generation {generation_id}")
        
        # Initialize STT engine
        logger.info("📝 Initializing STT engine...")
        stt_engine = STTEngine()
        
        if not stt_engine.is_available():
            logger.error("❌ STT engine not available")
            return False
        
        # Initialize TTS engine
        logger.info("📝 Initializing TTS engine...")
        tts_engine = KokoroTTSEngine()
        voice_mapper = VoiceMapper()
        mock_voice_assistant = MockVoiceAssistant(generation_id)
        
        # Initialize workers
        logger.info("📝 Initializing workers...")
        
        stt_worker = STTStreamingWorker(
            pipeline_manager=pipeline_manager,
            stt_engine=stt_engine,
            confidence_threshold=0.1,
            min_audio_length=0.1
        )
        
        tts_worker = TTSStreamingWorker(
            pipeline_manager=pipeline_manager,
            tts_engine=tts_engine,
            voice_mapper=voice_mapper,
            voice_assistant=mock_voice_assistant
        )
        
        # Create a simple callback to capture output
        output_audio_chunks = []
        def test_output_callback(audio_data):
            output_audio_chunks.append(audio_data)
            if hasattr(audio_data, 'audio_data'):
                logger.info(f"📢 Received audio output: {len(audio_data.audio_data)} samples")
            else:
                logger.info(f"📢 Received audio output: {len(audio_data)} samples")
        
        tts_output_worker = TTSOutputWorker(
            pipeline_manager=pipeline_manager,
            output_callback=test_output_callback
        )
        
        # Load test audio file
        logger.info("📁 Loading test audio...")
        audio_file = "/mnt/c/Users/PC/Dev/fastRTC/backend/tests/samples/audio_en.wav"
        
        import librosa
        audio_data, sr = librosa.load(audio_file, sr=None)
        
        # Convert to 2D format like threading pipeline
        if audio_data.ndim == 1:
            audio_data = audio_data.reshape(1, -1)
        
        logger.info(f"🎵 Loaded audio: {audio_data.shape}, {audio_data.dtype}, {sr}Hz")
        
        # Create audio chunk
        audio_chunk = AudioChunk(
            generation_id=generation_id,
            audio_data=audio_data,
            sample_rate=sr,
            timestamp=time.time(),
            is_final=True
        )
        
        # Step 1: STT Processing
        logger.info("🎤 Step 1: STT Processing...")
        stt_result = await stt_worker.process_item_async(audio_chunk)
        
        if not stt_result:
            logger.error("❌ STT failed")
            return False
            
        logger.info(f"✅ STT Result: '{stt_result.text}' (confidence: {stt_result.confidence:.3f})")
        
        # Step 2: Mock LLM Response (simulate LLM tokens)
        logger.info("🧠 Step 2: Simulating LLM response...")
        
        # Create mock LLM token chunks (send complete sentences)
        response_text = "Hello! I heard you say something interesting."
        sentences = [s.strip() + "." for s in response_text.split(".") if s.strip()]
        
        for i, sentence in enumerate(sentences):
            token_chunk = LLMTokenChunk(
                generation_id=generation_id,
                text=sentence,
                is_sentence_complete=True,  # Always complete sentences for TTS
                timestamp=time.time()
            )
            
            # Put token in LLM queue for TTS to consume
            pipeline_manager.llm_token_queue.put(token_chunk)
            logger.info(f"📤 Added sentence: '{sentence}'")
        
        # Step 3: TTS Processing
        logger.info("🔊 Step 3: TTS Processing...")
        logger.info(f"🔍 LLM queue size: {pipeline_manager.llm_token_queue.qsize()}")
        
        # Process sentences through TTS worker
        processed_tokens = 0
        while not pipeline_manager.llm_token_queue.empty():
            try:
                token_chunk = pipeline_manager.llm_token_queue.get(timeout=1.0)
                processed_tokens += 1
                logger.info(f"🔍 Processing token {processed_tokens}: '{token_chunk.text.strip()}'")
                
                # TTS worker streams directly to output queue, doesn't return chunks
                await tts_worker.process_item_async(token_chunk)
                logger.info(f"✅ TTS processed sentence: '{token_chunk.text.strip()}'")
                    
            except Exception as e:
                logger.error(f"❌ TTS processing error: {e}")
                break
        
        logger.info(f"🔍 Processed {processed_tokens} sentences through TTS")
        
        # Step 4: TTS Output Processing
        logger.info("📢 Step 4: TTS Output Processing...")
        logger.info(f"🔍 Output queue size: {pipeline_manager.output_queue.qsize()}")
        
        output_results = []
        processed_outputs = 0
        while not pipeline_manager.output_queue.empty():
            try:
                tts_chunk = pipeline_manager.output_queue.get(timeout=1.0)
                processed_outputs += 1
                logger.info(f"🔍 Processing output {processed_outputs}: {type(tts_chunk)}")
                
                # Output worker calls callback directly, doesn't return data
                tts_output_worker.process_item(tts_chunk)
                logger.info(f"✅ Output chunk processed")
                    
            except Exception as e:
                logger.error(f"❌ Output processing error: {e}")
                break
        
        logger.info(f"🔍 Processed {processed_outputs} output chunks, got {len(output_audio_chunks)} audio callbacks")
        
        # Results summary
        logger.info("📊 Pipeline Results Summary:")
        logger.info(f"   STT: '{stt_result.text}' (confidence: {stt_result.confidence:.3f})")
        logger.info(f"   LLM: {len(sentences)} sentences processed")
        logger.info(f"   TTS: {processed_tokens} sentences processed")
        logger.info(f"   Output: {len(output_audio_chunks)} final audio chunks")
        
        # Cleanup
        stt_engine.shutdown()
        logger.info("🧹 Cleanup complete")
        
        success = (
            stt_result is not None and
            processed_tokens > 0 and
            len(output_audio_chunks) > 0
        )
        
        if success:
            logger.info("🎉 Complete pipeline test PASSED!")
        else:
            logger.error("❌ Complete pipeline test FAILED!")
            
        return success
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        logger.error(f"❌ Traceback: {traceback.format_exc()}")
        return False

if __name__ == "__main__":
    success = asyncio.run(test_complete_pipeline())
    if success:
        print("✅ Complete pipeline test passed!")
        sys.exit(0)
    else:
        print("❌ Complete pipeline test failed!")
        sys.exit(1)