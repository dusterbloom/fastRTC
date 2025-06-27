#!/usr/bin/env python3
"""
Debug Threading Audio Pipeline

Quick debug script to test if audio is reaching the STT worker
"""

import os
import sys
import time
import logging

# Set threading environment
os.environ['USE_THREADING_PIPELINE'] = 'true'
os.environ['DEBUG_THREADING'] = 'true'

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def main():
    """Test threading pipeline initialization and check components."""
    
    print("🔍 Debugging Threading Audio Pipeline...")
    
    try:
        # Import components
        from src.integration.unified_callback_handler import UnifiedCallbackHandler
        from src.config.threading_config import is_threading_enabled
        from src.core.voice_assistant import VoiceAssistant
        
        print(f"Threading enabled: {is_threading_enabled()}")
        
        # Create voice assistant
        print("Creating voice assistant...")
        voice_assistant = VoiceAssistant()
        
        # Check components
        print(f"STT Engine available: {voice_assistant.stt_engine.is_available()}")
        print(f"TTS Engine available: {voice_assistant.tts_engine.is_available()}")
        print(f"Voice mapper: {voice_assistant.voice_mapper is not None}")
        
        # Create unified callback handler
        print("Creating unified callback handler...")
        callback_handler = UnifiedCallbackHandler(
            voice_assistant=voice_assistant,
            stt_engine=voice_assistant.stt_engine,
            tts_engine=voice_assistant.tts_engine,
            voice_mapper=voice_assistant.voice_mapper,
            event_loop=None
        )
        
        print(f"Active handler type: {callback_handler.handler_type}")
        
        # Get handler stats
        stats = callback_handler.get_handler_stats()
        print("Handler stats:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        # If threading handler is active, check pipeline stats
        if callback_handler.handler_type == "threading":
            threading_handler = callback_handler.threading_handler
            if threading_handler and hasattr(threading_handler, 'pipeline_manager'):
                pipeline_stats = threading_handler.pipeline_manager.get_pipeline_stats()
                print("Pipeline stats:")
                for key, value in pipeline_stats.items():
                    print(f"  {key}: {value}")
                    
                # Check worker stats
                if 'workers' in pipeline_stats:
                    print("Worker stats:")
                    for worker_name, worker_stats in pipeline_stats['workers'].items():
                        print(f"  {worker_name}:")
                        for stat_key, stat_value in worker_stats.items():
                            print(f"    {stat_key}: {stat_value}")
        
        print("✅ Threading pipeline debug complete")
        
        # Cleanup
        callback_handler.stop()
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()