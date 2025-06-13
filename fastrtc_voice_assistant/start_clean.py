#!/usr/bin/env python3
"""
Simplified FastRTC Voice Assistant Entry Point
============================================

A minimal FastAPI entry point that leverages the refactored architecture.
This replaces the 413-line start.py with a clean, maintainable version.

Launch with:
    python start_clean.py
    or
    fastRTC/python -m uvicorn fastrtc_voice_assistant.start_clean:app --reload --port 8000
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# Make local packages importable
sys.path.insert(0, str(Path(__file__).parent.resolve()))

from src.core.voice_assistant import VoiceAssistant
from src.integration.fastrtc_bridge import FastRTCBridge
from src.integration.callback_handler import StreamCallbackHandler
from src.utils.async_utils import AsyncEnvironmentManager
from src.config.settings import load_config
from src.utils.logging import get_logger, setup_logging

# Initial setup
setup_logging()
logger = get_logger(__name__)
#logger.critical("🚨 TOP LEVEL LOGGER TEST IN START_CLEAN.PY 🚨") # New test log

# Global components
voice_assistant: Optional[VoiceAssistant] = None
fastrtc_bridge: Optional[FastRTCBridge] = None
callback_handler: Optional[StreamCallbackHandler] = None
async_env_manager: Optional[AsyncEnvironmentManager] = None

async def initialize_voice_assistant():
    """Initialize the voice assistant and all its components."""
    global voice_assistant, fastrtc_bridge, callback_handler, async_env_manager
    
    if voice_assistant is not None:
        return  # Already initialized
    
    try:
        logger.info("🚀 Initializing Voice Assistant...")
        
        # Create voice assistant
        voice_assistant = VoiceAssistant(config=load_config())
        logger.critical(f"[DEBUG] VoiceAssistant created: {voice_assistant}")
        logger.critical(f"[DEBUG] STT engine: {getattr(voice_assistant, 'stt_engine', None)}")
        logger.critical(f"[DEBUG] TTS engine: {getattr(voice_assistant, 'tts_engine', None)}")
        
        # Initialize async environment
        async_env_manager = AsyncEnvironmentManager() # Corrected class name
        success = async_env_manager.setup_async_environment(voice_assistant)
        if not success:
            raise RuntimeError("Failed to setup async environment")
        
        # Create FastRTC bridge
        fastrtc_bridge = FastRTCBridge()
        logger.critical(f"[DEBUG] FastRTCBridge created: {fastrtc_bridge}")
        
        # Create callback handler with all dependencies
        callback_handler = StreamCallbackHandler(
            voice_assistant=voice_assistant,
            stt_engine=voice_assistant.stt_engine,
            tts_engine=voice_assistant.tts_engine,
            language_detector=voice_assistant.language_detector,
            voice_mapper=voice_assistant.voice_mapper,
            event_loop=async_env_manager.get_event_loop()
        )
        logger.critical(f"[DEBUG] StreamCallbackHandler created: {callback_handler}")
        logger.critical(f"[DEBUG] CallbackHandler STT: {getattr(callback_handler, 'stt_engine', None)}")
        
        logger.info("✅ Voice Assistant initialization complete")

        # --- FORCED TTS TEST: Synthesize a test phrase at startup to validate TTS engine ---
        try:
            from src.config.language_config import KOKORO_TTS_LANG_MAP
            kokoro_lang = 'a'
            test_voice = 'af_heart'
            test_text = "This is a test of the TTS engine at startup."
            tts_lang = KOKORO_TTS_LANG_MAP.get(kokoro_lang, 'en-us')
            from src.audio.engines.tts.kokoro_tts import KokoroTTSOptions
            tts_options = KokoroTTSOptions(voice=test_voice, speed=1.05, lang=tts_lang)
            logger.critical(f"[TTS STARTUP TEST] Attempting TTS with voice='{test_voice}', lang='{tts_lang}'")
            chunk_count = 0
            total_samples = 0
            for tts_output_item in voice_assistant.tts_engine.stream_tts_sync(test_text, tts_options):
                if isinstance(tts_output_item, tuple) and len(tts_output_item) == 2:
                    sample_rate, audio_array = tts_output_item
                    if hasattr(audio_array, "size") and audio_array.size > 0:
                        chunk_count += 1
                        total_samples += audio_array.size
                elif hasattr(tts_output_item, "size") and tts_output_item.size > 0:
                    chunk_count += 1
                    total_samples += tts_output_item.size
            if chunk_count == 0:
                logger.critical("[TTS STARTUP TEST] ❌ No audio chunks produced by TTS engine at startup.")
            else:
                logger.critical(f"[TTS STARTUP TEST] ✅ Audio produced: {chunk_count} chunks, {total_samples} samples.")
        except Exception as e:
            logger.critical(f"[TTS STARTUP TEST] Exception during forced TTS test: {e}")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize voice assistant: {e}")
        raise

async def create_fastrtc_stream():
    """Create and configure the FastRTC stream."""
    if not all([voice_assistant, fastrtc_bridge, callback_handler]):
        await initialize_voice_assistant()
    
    # Create the FastRTC stream with our callback
    from fastrtc import SileroVadOptions
    stream = fastrtc_bridge.create_stream(
        callback_function=callback_handler.process_audio_stream,
        speech_threshold=0.05,  # Sensitive speech detection
        server_name="0.0.0.0",
        server_port=8000,
        share=False,
 
    )
    
    logger.info("🎤 FastRTC stream created successfully")
    return stream

# ────────────────────────── FASTAPI APP  ────────────────────────────
app = FastAPI(title="FastRTC Voice Assistant", version="1.0.0")

# CORS setup for front-end integration
origins = [
    "http://localhost:5173",               # Original Vite dev-server port
    "http://localhost:3001",               # Next.js dev-server port
    "http://127.0.0.1:3001",              # Explicit IP for Next.js dev-server
    os.getenv("FRONTEND_URL", ""),         # prod domain, if set
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o for o in origins if o],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    """Initialize the voice assistant and mount FastRTC routes."""
    print(">>> FastAPI startup_event triggered")
    try:
        # Initialize voice assistant components
        await initialize_voice_assistant()
        
        # Create and mount FastRTC stream
        stream = await create_fastrtc_stream()
        stream.mount(app, path="/assistant")
        
        logger.info("🌐 FastRTC routes mounted at /assistant")
        logger.info("✅ FastRTC Voice Assistant is ready!")
        
    except Exception as e:
        logger.error(f"❌ Failed to start voice assistant: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on FastAPI shutdown."""
    global voice_assistant, fastrtc_bridge, async_env_manager
    
    try:
        logger.info("🛑 Shutting down voice assistant...")
        
        if fastrtc_bridge:
            fastrtc_bridge.stop_stream()
            
        if async_env_manager:
            async_env_manager.shutdown(timeout=10)
            
        logger.info("👋 Voice assistant shutdown complete")
        
    except Exception as e:
        logger.error(f"❌ Error during shutdown: {e}")

@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "message": "FastRTC Voice Assistant is running",
        "status": "healthy",
        "version": "1.0.0"
    }

@app.get("/health")
async def health_check():
    """Detailed health check endpoint."""
    try:
        if not voice_assistant:
            return {"status": "unhealthy", "error": "Voice assistant not initialized"}
            
        return {
            "status": "healthy",
            "components": {
                "voice_assistant": voice_assistant is not None,
                "fastrtc_bridge": fastrtc_bridge is not None,
                "callback_handler": callback_handler is not None,
                "async_env_manager": async_env_manager is not None,
            }
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }

# Optional: serve a built SPA from /frontend/dist
_frontend_dist = Path(__file__).parent / "frontend" / "dist"
if _frontend_dist.exists():
    app.mount("/", StaticFiles(directory=_frontend_dist, html=True), name="spa")

# ────────────────────────── ENTRY POINT  ────────────────────────────
if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "fastrtc_voice_assistant.start_clean:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        reload=True,
        log_config=None  
    )
