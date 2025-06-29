#!/usr/bin/env python3
"""
Test LLM Service in Threading Context
Check if LLM service works properly in threading pipeline
"""

import os
import sys
import asyncio
import time
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent / "backend"))

# Set environment
os.environ['STT_BACKEND'] = 'faster_gpu'
os.environ['USE_THREADING_PIPELINE'] = 'true'
os.environ['LOG_LEVEL'] = 'INFO'

from backend.src.utils.logging import get_logger
from backend.src.core.llm_worker import LLMStreamingWorker
from backend.src.core.pipeline_manager import AudioPipelineManager, TranscriptionChunk
from backend.src.core.voice_assistant import VoiceAssistant

logger = get_logger(__name__)

class MockVoiceAssistant:
    """Mock voice assistant that bypasses voice authentication."""
    def __init__(self):
        self.user_id = "test_user"
        self.voice_print_manager = None  # Disable voice auth
        
        # Initialize LLM service properly
        from backend.src.services.llm_service import LLMService
        self.llm_service = LLMService()
        
        # Initialize memory manager
        from backend.src.memory.manager import AMemMemoryManager
        self.memory_manager = AMemMemoryManager(user_id="test_user")

async def test_llm_in_threading_context():
    """Test LLM service in threading pipeline context."""
    logger.info("🧠 Testing LLM service in threading context...")
    
    try:
        # Step 1: Create pipeline components
        logger.info("📝 Step 1: Creating pipeline components...")
        pipeline_manager = AudioPipelineManager()
        generation_id = pipeline_manager.create_generation()
        
        # Create mock voice assistant without voice auth
        mock_voice_assistant = MockVoiceAssistant()
        
        # Step 2: Initialize LLM service with HTTP session
        logger.info("📝 Step 2: Initializing LLM service...")
        
        import aiohttp
        async with aiohttp.ClientSession() as session:
            # Initialize LLM service with HTTP session
            await mock_voice_assistant.llm_service.initialize(
                http_session=session,
                response_cache=None,
                conversation_buffer=None,
                memory_manager=mock_voice_assistant.memory_manager
            )
            
            logger.info("✅ LLM service initialized with HTTP session")
            
            # Step 3: Create LLM worker
            logger.info("📝 Step 3: Creating LLM worker...")
            llm_worker = LLMStreamingWorker(
                pipeline_manager=pipeline_manager,
                voice_assistant=mock_voice_assistant
            )
            
            # Step 4: Test LLM processing
            logger.info("📝 Step 4: Testing LLM processing...")
            
            # Create transcription chunk
            transcription = TranscriptionChunk(
                generation_id=generation_id,
                text="Hello Echo, can you hear me?",
                confidence=0.95,
                is_partial=False,
                timestamp=time.time()
            )
            
            logger.info(f"🎤 Processing transcription: '{transcription.text}'")
            
            # Process through LLM worker
            result = await llm_worker.process_item_async(transcription)
            
            if result:
                logger.info(f"✅ LLM worker returned result: {type(result)}")
                if hasattr(result, 'text'):
                    logger.info(f"✅ LLM response: '{result.text}'")
                elif hasattr(result, 'tokens'):
                    logger.info(f"✅ LLM tokens: {result.tokens}")
                else:
                    logger.info(f"✅ LLM result: {result}")
                
                success = True
            else:
                logger.error("❌ LLM worker returned None")
                success = False
            
            # Step 5: Check if tokens were queued
            logger.info("📝 Step 5: Checking LLM token queue...")
            token_count = 0
            while not pipeline_manager.llm_token_queue.empty():
                try:
                    token_chunk = pipeline_manager.llm_token_queue.get(timeout=0.1)
                    token_count += 1
                    logger.info(f"📥 Token {token_count}: '{token_chunk.text}'")
                    if token_count > 10:  # Limit for test
                        break
                except Exception:
                    break
            
            logger.info(f"✅ Found {token_count} tokens in queue")
            
            if token_count > 0:
                success = True
            
        # Results
        logger.info("📊 LLM Threading Context Test Results:")
        logger.info(f"   LLM worker result: {'✅' if result else '❌'}")
        logger.info(f"   Tokens in queue: {token_count}")
        logger.info(f"   Overall success: {'✅' if success else '❌'}")
        
        if success:
            logger.info("🎉 LLM threading context test PASSED!")
        else:
            logger.error("❌ LLM threading context test FAILED!")
            
        return success
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        logger.error(f"❌ Traceback: {traceback.format_exc()}")
        return False

if __name__ == "__main__":
    success = asyncio.run(test_llm_in_threading_context())
    if success:
        print("✅ LLM threading context test passed!")
        sys.exit(0)
    else:
        print("❌ LLM threading context test failed!")
        sys.exit(1)