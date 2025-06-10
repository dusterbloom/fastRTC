#!/usr/bin/env python3
"""
FastRTC Voice Assistant - Debug Version using VoiceAssistant directly

This version bypasses the VoiceAssistantApplication wrapper to debug startup issues.
"""

import sys
import asyncio
import threading
import time
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, 'src')

from src.core.voice_assistant import VoiceAssistant
from src.integration.callback_handler import StreamCallbackHandler
from src.integration.fastrtc_bridge import FastRTCBridge
from src.utils.async_utils import AsyncEnvironmentManager
from src.utils.logging import setup_logging, get_logger

# Setup logging
setup_logging()
logger = get_logger(__name__)

# Configuration
DEFAULT_SERVER_NAME = "0.0.0.0"
DEFAULT_SERVER_PORT = 7860
DEFAULT_SHARE = False

# Global variables (similar to gradio2.py pattern)
voice_assistant = None
main_event_loop = None
async_worker_thread = None

def setup_async_environment():
    """Setup async environment similar to gradio2.py pattern"""
    global main_event_loop, voice_assistant, async_worker_thread
    
    logger.info("🧠 Creating VoiceAssistant instance...")
    voice_assistant = VoiceAssistant()

    def run_async_loop_in_thread():
        global main_event_loop, voice_assistant
        main_event_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(main_event_loop)
        
        if voice_assistant:
            logger.info("⚡ Initializing async components...")
            main_event_loop.run_until_complete(voice_assistant.initialize_async())
        else:
            logger.error("🚨 Voice assistant instance is None in async thread. Cannot initialize.")
            return

        try:
            main_event_loop.run_forever()
        except KeyboardInterrupt:
            logger.info("Async loop interrupted in thread.")
        finally:
            if voice_assistant and main_event_loop and not main_event_loop.is_closed():
                logger.info("Cleaning up assistant resources in async thread...")
                main_event_loop.run_until_complete(voice_assistant.cleanup_async())
            if main_event_loop and not main_event_loop.is_closed():
                 main_event_loop.close()
            logger.info("Async event loop closed.")

    async_worker_thread = threading.Thread(target=run_async_loop_in_thread, daemon=True, name="AsyncWorkerThread")
    async_worker_thread.start()

    # Wait for initialization
    for _ in range(100):
        if main_event_loop and main_event_loop.is_running() and voice_assistant:
            logger.info("✅ Async environment and refactored components are ready.")
            return True
        time.sleep(0.1)
    
    logger.error("⚠️ Async environment or components did not confirm readiness in time.")
    return False

def print_startup_banner():
    """Print the startup banner with system information."""
    print("=" * 70)
    print("🚀 FastRTC Voice Assistant - Debug Version")
    print("🎤 Using modular STT, TTS, Memory, and LLM services")
    print("=" * 70)
    print("💡 Test Commands:")
    print("   • 'My name is [Your Name]'")
    print("   • 'What is my name?' / 'Who am I?'")
    print("   • 'I like [something interesting]'")
    print("   • Ask questions in supported languages.")
    print("\n🛑 To stop: Press Ctrl+C in the terminal")
    print("=" * 70)

def run_voice_assistant_direct(
    server_name: str = DEFAULT_SERVER_NAME,
    server_port: int = DEFAULT_SERVER_PORT,
    share: bool = DEFAULT_SHARE
):
    """
    Run the voice assistant using direct integration like gradio2.py
    """
    global voice_assistant, main_event_loop
    
    # Setup async environment
    logger.info("🔧 Setting up async environment...")
    if not setup_async_environment():
        logger.error("❌ Failed to setup async environment")
        return False
    
    # Create FastRTC components
    logger.info("🌐 Creating FastRTC bridge...")
    fastrtc_bridge = FastRTCBridge()
    
    logger.info("🎤 Creating stream callback handler...")
    callback_handler = StreamCallbackHandler(
        voice_assistant=voice_assistant,
        stt_engine=voice_assistant.stt_engine,
        tts_engine=voice_assistant.tts_engine,
        language_detector=voice_assistant.language_detector,
        voice_mapper=voice_assistant.voice_mapper,
        event_loop=main_event_loop
    )
    
    try:
        # Determine speech threshold
        speech_threshold = 0.15  # Default
        if (voice_assistant and 
            hasattr(voice_assistant.audio_processor, 'noise_floor') and
            voice_assistant.audio_processor.noise_floor):
            speech_threshold = voice_assistant.audio_processor.noise_floor * 15
        
        logger.info(f"🎯 Using speech threshold: {speech_threshold}")
        
        # Create FastRTC stream
        logger.info("🌐 Creating FastRTC stream...")
        stream = fastrtc_bridge.create_stream(
            callback_function=callback_handler.process_audio_stream,
            speech_threshold=speech_threshold
        )
        
        # Print startup banner
        print_startup_banner()
        
        # Launch the stream
        logger.info("🚀 Launching FastRTC voice assistant...")
        fastrtc_bridge.launch_stream(
            server_name=server_name,
            server_port=server_port,
            share=share,
            quiet=False
        )
        
        return True
        
    except KeyboardInterrupt:
        logger.info("🛑 KeyboardInterrupt received. Shutting down...")
        return True
    except Exception as e:
        logger.error(f"❌ Application error: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Cleanup
        logger.info("🏁 Initiating shutdown sequence...")
        if main_event_loop and not main_event_loop.is_closed():
            logger.info("Requesting async event loop to stop...")
            main_event_loop.call_soon_threadsafe(main_event_loop.stop)

        if async_worker_thread and async_worker_thread.is_alive():
            logger.info("Waiting for async worker thread to join...")
            async_worker_thread.join(timeout=15)
            if async_worker_thread.is_alive():
                logger.info("⚠️ Async worker thread did not join in time.")

        logger.info("👋 Debug voice assistant shutdown process complete.")

if __name__ == "__main__":
    # Parse command line arguments if needed
    import argparse
    
    parser = argparse.ArgumentParser(description="FastRTC Voice Assistant - Debug Version")
    parser.add_argument("--host", default=DEFAULT_SERVER_NAME, 
                       help=f"Server hostname (default: {DEFAULT_SERVER_NAME})")
    parser.add_argument("--port", type=int, default=DEFAULT_SERVER_PORT,
                       help=f"Server port (default: {DEFAULT_SERVER_PORT})")
    parser.add_argument("--share", action="store_true", default=DEFAULT_SHARE,
                       help="Create a public share link")
    
    args = parser.parse_args()
    
    # Run the application
    try:
        run_voice_assistant_direct(
            server_name=args.host,
            server_port=args.port,
            share=args.share
        )
    except KeyboardInterrupt:
        logger.info("🛑 Application interrupted by user")
    except Exception as e:
        logger.error(f"❌ Fatal application error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
