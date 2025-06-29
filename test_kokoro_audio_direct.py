#!/usr/bin/env python3
"""
Test Kokoro TTS Audio Production Directly
Verify Kokoro is producing real, playable audio
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

# Set environment
os.environ['STT_BACKEND'] = 'faster_gpu'
os.environ['LOG_LEVEL'] = 'INFO'

from backend.src.utils.logging import get_logger
from backend.src.core.tts_worker import TTSStreamingWorker, TTSOutputWorker
from backend.src.core.pipeline_manager import AudioPipelineManager, LLMTokenChunk
from backend.src.audio.engines.tts.kokoro_tts import KokoroTTSEngine
from backend.src.audio import VoiceMapper

logger = get_logger(__name__)

class MockVoiceAssistant:
    """Mock voice assistant for testing."""
    def __init__(self, generation_id):
        self.current_generation_id = generation_id
        
    def get_current_generation_id(self):
        return self.current_generation_id

async def test_kokoro_audio_direct():
    """Test Kokoro TTS audio production directly."""
    logger.info("🔊 Testing Kokoro TTS audio production directly...")
    
    try:
        # Initialize pipeline manager
        logger.info("📝 Initializing pipeline manager...")
        pipeline_manager = AudioPipelineManager()
        generation_id = pipeline_manager.create_generation()
        logger.info(f"📝 Created generation {generation_id}")
        
        # Initialize TTS components
        logger.info("📝 Initializing TTS components...")
        tts_engine = KokoroTTSEngine()
        voice_mapper = VoiceMapper()
        mock_voice_assistant = MockVoiceAssistant(generation_id)
        
        # Initialize TTS workers
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
            if hasattr(audio_chunk, 'audio_data'):
                logger.info(f"📢 Received audio chunk: {len(audio_chunk.audio_data)} samples")
            else:
                logger.info(f"📢 Received audio chunk: {type(audio_chunk)}")
        
        tts_output_worker = TTSOutputWorker(
            pipeline_manager=pipeline_manager,
            output_callback=audio_callback
        )
        
        # Test sentences
        test_sentences = [
            "Hello! This is a test of the Kokoro text-to-speech system.",
            "The quick brown fox jumps over the lazy dog.",
            "FastRTC voice assistant is working perfectly with threading pipeline."
        ]
        
        logger.info(f"🎤 Testing {len(test_sentences)} sentences...")
        
        for i, sentence in enumerate(test_sentences):
            logger.info(f"🔊 Processing sentence {i+1}: '{sentence}'")
            
            # Create LLM token chunk
            token_chunk = LLMTokenChunk(
                generation_id=generation_id,
                text=sentence,
                is_sentence_complete=True,
                timestamp=time.time()
            )
            
            # Process through TTS worker
            await tts_worker.process_item_async(token_chunk)
            logger.info(f"✅ TTS processed sentence {i+1}")
            
            # Process any output chunks
            while not pipeline_manager.output_queue.empty():
                try:
                    tts_chunk = pipeline_manager.output_queue.get(timeout=0.1)
                    tts_output_worker.process_item(tts_chunk)
                except Exception:
                    break
        
        # Wait a bit for any remaining processing
        await asyncio.sleep(2.0)
        
        # Process any remaining output chunks
        while not pipeline_manager.output_queue.empty():
            try:
                tts_chunk = pipeline_manager.output_queue.get(timeout=0.1)
                tts_output_worker.process_item(tts_chunk)
            except Exception:
                break
        
        logger.info(f"📊 Total audio chunks received: {len(output_audio_chunks)}")
        
        # Save audio output if we got any
        if output_audio_chunks:
            logger.info("💾 Saving audio output...")
            
            # Combine all audio chunks
            all_audio_data = []
            sample_rate = 22050  # Kokoro default
            
            for chunk in output_audio_chunks:
                if hasattr(chunk, 'audio_data'):
                    audio_data = chunk.audio_data
                    if hasattr(chunk, 'sample_rate'):
                        sample_rate = chunk.sample_rate
                elif isinstance(chunk, np.ndarray):
                    audio_data = chunk
                else:
                    continue
                    
                # Ensure audio is numpy array
                if not isinstance(audio_data, np.ndarray):
                    continue
                    
                # Flatten if needed
                if audio_data.ndim > 1:
                    audio_data = audio_data.flatten()
                    
                all_audio_data.append(audio_data)
            
            if all_audio_data:
                # Concatenate all audio
                combined_audio = np.concatenate(all_audio_data)
                
                logger.info(f"📊 Combined audio stats:")
                logger.info(f"   Total samples: {len(combined_audio)}")
                logger.info(f"   Duration: {len(combined_audio) / sample_rate:.2f}s")
                logger.info(f"   Sample rate: {sample_rate}Hz")
                logger.info(f"   Data type: {combined_audio.dtype}")
                logger.info(f"   Min/Max: {combined_audio.min():.3f} / {combined_audio.max():.3f}")
                
                # Save as WAV file
                output_file = "/mnt/c/Users/PC/Dev/fastRTC/kokoro_test_output.wav"
                
                # Convert to int16 if needed
                if combined_audio.dtype == np.float32 or combined_audio.dtype == np.float64:
                    # Normalize and convert to int16
                    max_val = np.abs(combined_audio).max()
                    if max_val > 0:
                        combined_audio = combined_audio / max_val
                    combined_audio = (combined_audio * 32767).astype(np.int16)
                elif combined_audio.dtype != np.int16:
                    combined_audio = combined_audio.astype(np.int16)
                
                # Save WAV file
                with wave.open(output_file, 'wb') as wav_file:
                    wav_file.setnchannels(1)  # Mono
                    wav_file.setsampwidth(2)  # 16-bit
                    wav_file.setframerate(sample_rate)
                    wav_file.writeframes(combined_audio.tobytes())
                
                logger.info(f"💾 ✅ Audio saved to: {output_file}")
                
                # Test if file was created successfully
                if os.path.exists(output_file):
                    file_size = os.path.getsize(output_file)
                    logger.info(f"📁 File size: {file_size} bytes")
                    success = True
                else:
                    logger.error("❌ Audio file was not created")
                    success = False
            else:
                logger.error("❌ No valid audio data found")
                success = False
        else:
            logger.error("❌ No audio chunks received")
            success = False
        
        # Results summary
        logger.info("📊 Kokoro Audio Test Results:")
        logger.info(f"   Sentences processed: {len(test_sentences)}")
        logger.info(f"   Audio chunks: {len(output_audio_chunks)}")
        logger.info(f"   Audio file created: {success}")
        
        if success:
            logger.info("🎉 Kokoro audio test PASSED!")
        else:
            logger.error("❌ Kokoro audio test FAILED!")
            
        return success
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        logger.error(f"❌ Traceback: {traceback.format_exc()}")
        return False

if __name__ == "__main__":
    success = asyncio.run(test_kokoro_audio_direct())
    if success:
        print("✅ Kokoro audio test passed!")
        sys.exit(0)
    else:
        print("❌ Kokoro audio test failed!")
        sys.exit(1)