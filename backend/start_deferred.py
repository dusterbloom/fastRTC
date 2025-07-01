#!/usr/bin/env python3
"""
FastRTC Voice Assistant - Non-Blocking Startup Solution
=====================================================

Core Problem: Voice assistant initialization blocks FastAPI startup
Solution: Deferred initialization with background task execution
"""

from __future__ import annotations

import os
os.environ['STT_BACKEND'] = 'faster'
os.environ['PYTHONIOENCODING'] = 'utf-8'

import sys
import asyncio
import threading
import argparse
from pathlib import Path
from typing import Optional
from contextlib import asynccontextmanager

# Early argument parsing to set threading environment variables before imports
def parse_early_args():
    """Parse only threading arguments early to set environment variables before imports."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--threading', action='store_true')
    parser.add_argument('--no-fallback', action='store_true')
    parser.add_argument('--whisper-live', action='store_true')
    args, _ = parser.parse_known_args()
    
    if args.threading:
        os.environ['USE_THREADING_PIPELINE'] = 'true'
        print(f"🧵 Threading pipeline enabled via command line")
    
    if args.no_fallback:
        os.environ['THREADING_FALLBACK_TO_ASYNC'] = 'false'
        print(f"🚫 Threading fallback disabled via command line")
    
    if args.whisper_live:
        os.environ['USE_WHISPER_LIVE'] = 'true'
        os.environ['STT_BACKEND'] = 'whisper_live'
        print(f"🎤 WhisperLive STT backend enabled via command line")
    
    # Enable GPU acceleration for Kokoro TTS
    os.environ['ONNX_PROVIDER'] = 'CUDAExecutionProvider'
    print(f"🚀 GPU acceleration enabled for Kokoro TTS")
    
    # Check for WhisperLive configuration
    if os.environ.get('USE_WHISPER_LIVE', 'false').lower() == 'true':
        os.environ['STT_BACKEND'] = 'whisper_live'
        print(f"🎤 WhisperLive STT backend enabled")
    
    return args

# Parse threading arguments before any imports
early_args = parse_early_args()

from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# Make local packages importable
sys.path.insert(0, str(Path(__file__).parent.resolve()))

from src.core.voice_assistant import VoiceAssistant
from src.integration.fastrtc_bridge import FastRTCBridge
from src.integration.unified_callback_handler import UnifiedCallbackHandler
from src.utils.async_utils import AsyncEnvironmentManager
from src.config.settings import load_config
from src.utils.logging import get_logger, setup_logging

# Parse command line arguments for log level and threading options
def parse_args():
    parser = argparse.ArgumentParser(description='FastRTC Voice Assistant Server')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], 
                       help='Set logging level (overrides LOG_LEVEL environment variable)')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=8000, help='Port to bind to')
    parser.add_argument('--threading', action='store_true',
                       help='Enable threading pipeline (experimental)')
    parser.add_argument('--no-fallback', action='store_true',
                       help='Disable fallback to async pipeline')
    parser.add_argument('--whisper-live', action='store_true',
                       help='Enable WhisperLive STT backend (requires threading)')
    return parser.parse_args()

# Initial setup - read log level from environment (command line args parsed in main)
log_level = os.getenv("LOG_LEVEL", "INFO")
setup_logging(log_level)
logger = get_logger(__name__)
logger.debug("🚨 TOP LEVEL LOGGER TEST IN START_CLEAN.PY 🚨") # New test log

# Global components with thread-safe initialization
_components_lock = threading.Lock()
_initialization_task: Optional[asyncio.Task] = None

class GlobalComponents:
    """Thread-safe container for voice assistant components"""
    def __init__(self):
        self.voice_assistant: Optional[VoiceAssistant] = None
        self.fastrtc_bridge: Optional[FastRTCBridge] = None
        self.callback_handler: Optional[UnifiedCallbackHandler] = None
        self.async_env_manager: Optional[AsyncEnvironmentManager] = None
        self.stream = None
        self.is_initialized = False
        self.initialization_error: Optional[str] = None

components = GlobalComponents()

# Pydantic models for API requests
class LanguageRequest(BaseModel):
    language: str

async def initialize_voice_assistant_deferred(app: FastAPI):
    """
    Initialize voice assistant in background without blocking server startup.
    This runs AFTER the server is listening on port 8000.
    """
    global components
    
    try:
        logger.critical("🔥 [DEFERRED] initialize_voice_assistant_deferred() called!")
        logger.info("🚀 Starting deferred voice assistant initialization...")
        
        # Small delay to ensure server is fully ready
        await asyncio.sleep(1.0)
        
        # Create voice assistant
        logger.info("Creating VoiceAssistant...")
        voice_assistant = VoiceAssistant(config=load_config())
        
        # Create async environment manager
        async_env_manager = AsyncEnvironmentManager()
        success = async_env_manager.setup_async_environment(voice_assistant)
        if not success:
            raise RuntimeError("Failed to setup async environment")
        
        # Create FastRTC bridge
        fastrtc_bridge = FastRTCBridge()
        
        # Create callback handler with debug logging
        logger.info("🎯 Creating callback handler...")
        logger.info(f"🧵 Threading pipeline environment: USE_THREADING_PIPELINE={os.getenv('USE_THREADING_PIPELINE', 'false')}")
        logger.info(f"🔄 Threading fallback environment: THREADING_FALLBACK_TO_ASYNC={os.getenv('THREADING_FALLBACK_TO_ASYNC', 'true')}")
        
        # Check if we should use threading directly
        print(f"🔍 DEBUG: Environment: USE_THREADING_PIPELINE={os.getenv('USE_THREADING_PIPELINE')}")
        logger.info(f"🔍 Environment: USE_THREADING_PIPELINE={os.getenv('USE_THREADING_PIPELINE')}")
        from src.config.threading_config import is_threading_enabled
        threading_enabled = is_threading_enabled()
        print(f"🔍 DEBUG: is_threading_enabled() = {threading_enabled}")
        logger.info(f"🔍 is_threading_enabled() = {threading_enabled}")
        if threading_enabled:
            print("🧵 DEBUG: Using ThreadingCallbackHandler directly")
            logger.info("🧵 Using ThreadingCallbackHandler directly")
            from src.integration.threading_callback_handler import ThreadingCallbackHandler
            callback_handler = ThreadingCallbackHandler(
                voice_assistant=voice_assistant,
                stt_engine=voice_assistant.stt_engine,
                tts_engine=voice_assistant.tts_engine,
                voice_mapper=voice_assistant.voice_mapper,
                event_loop=async_env_manager.get_event_loop()
            )
            print("🧵 DEBUG: Starting threading callback handler...")
            callback_handler.start()
            print("🧵 DEBUG: Threading callback handler started!")
            handler_type = "threading"
        else:
            print("🔄 DEBUG: Using UnifiedCallbackHandler")
            logger.info("🔄 Using UnifiedCallbackHandler")
            callback_handler = UnifiedCallbackHandler(
                voice_assistant=voice_assistant,
                stt_engine=voice_assistant.stt_engine,
                tts_engine=voice_assistant.tts_engine,
                voice_mapper=voice_assistant.voice_mapper,
                event_loop=async_env_manager.get_event_loop()
            )
            handler_type = "unified"
        
        # Log which handler was actually initialized
        logger.info(f"✅ Callback handler initialized with: {handler_type} handler")
        if hasattr(callback_handler, 'get_handler_stats'):
            handler_stats = callback_handler.get_handler_stats()
            logger.info(f"📊 Handler stats: {handler_stats}")
        
        # Create FastRTC stream with proper network configuration
        logger.info("Creating FastRTC stream...")
        from fastrtc import SileroVadOptions
        
        # Get server configuration from environment
        server_host = os.getenv("HOST", "0.0.0.0")
        server_port = int(os.getenv("PORT", "8000"))
        external_ip = os.getenv("EXTERNAL_IP", "localhost")
        
        logger.info(f"Server config - Host: {server_host}, Port: {server_port}, External IP: {external_ip}")
        
        stream = fastrtc_bridge.create_stream(
            callback_function=callback_handler.process_audio_stream,
            speech_threshold=0.05,
            server_name=server_host,
            server_port=server_port,
            share=False,
        )
        
        # Mount stream to FastAPI app
        logger.info("Mounting FastRTC stream to /assistant...")
        logger.debug(f"Stream to mount: {stream}")
        logger.debug(f"App to mount to: {app}")
        stream.mount(app, path="/assistant")
        logger.debug("FastRTC stream mounted successfully")
        
        # Store components
        with _components_lock:
            components.voice_assistant = voice_assistant
            components.fastrtc_bridge = fastrtc_bridge
            components.callback_handler = callback_handler
            components.async_env_manager = async_env_manager
            components.stream = stream
            components.is_initialized = True
        
        logger.info("✅ Voice assistant initialization complete! WebRTC endpoint available at /assistant")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize voice assistant: {e}")
        import traceback
        traceback.print_exc()
        with _components_lock:
            components.initialization_error = str(e)

# Create FastAPI app WITHOUT lifespan context manager
app = FastAPI(
    title="FastRTC Voice Assistant",
    version="1.0.0",
    description="Real-time voice assistant with WebRTC support"
)

# CORS setup
origins = [
    "http://localhost:5173",
    "http://localhost:3001",
    "http://127.0.0.1:3001",
    os.getenv("FRONTEND_URL", ""),
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
    """
    Start voice assistant initialization in background.
    This doesn't block the server from starting.
    """
    global _initialization_task
    logger.critical("🔥 [STARTUP] startup_event() called!")
    # Create background task for initialization
    _initialization_task = asyncio.create_task(initialize_voice_assistant_deferred(app))
    logger.critical("🔥 [STARTUP] Background task created, voice assistant initializing...")
    logger.info("🚀 Server started, voice assistant initializing in background...")
    
    # Set WhisperLive API callback handler reference after initialization
    async def setup_whisperlive():
        if _initialization_task:
            await _initialization_task
        if components.callback_handler:
            set_callback_handler(components.callback_handler)
            logger.info("✅ WhisperLive API callback handler set")
    
    # Create another background task for WhisperLive setup
    asyncio.create_task(setup_whisperlive())

@app.get("/")
async def root():
    """Health check endpoint with initialization status."""
    return {
        "message": "FastRTC Voice Assistant",
        "status": "running",
        "version": "1.0.0",
        "voice_assistant": {
            "initialized": components.is_initialized,
            "error": components.initialization_error
        }
    }

@app.get("/health")
async def health_check():
    """Detailed health check endpoint."""
    return {
        "status": "healthy" if components.is_initialized else "initializing",
        "components": {
            "voice_assistant": components.voice_assistant is not None,
            "fastrtc_bridge": components.fastrtc_bridge is not None,
            "callback_handler": components.callback_handler is not None,
            "async_env_manager": components.async_env_manager is not None,
            "stream_mounted": components.stream is not None,
        },
        "initialization_error": components.initialization_error
    }

@app.get("/initialization/status")
async def initialization_status():
    """Check voice assistant initialization progress."""
    if components.is_initialized:
        return JSONResponse({
            "status": "complete",
            "message": "Voice assistant ready",
            "webrtc_endpoint": "/assistant"
        })
    elif components.initialization_error:
        return JSONResponse({
            "status": "failed",
            "error": components.initialization_error
        }, status_code=500)
    else:
        return JSONResponse({
            "status": "in_progress",
            "message": "Voice assistant initializing..."
        }, status_code=202)

@app.post("/debug/reinitialize")
async def debug_reinitialize(background_tasks: BackgroundTasks):
    """Force re-initialization of voice assistant (for debugging)."""
    if _initialization_task and not _initialization_task.done():
        return JSONResponse({
            "status": "error",
            "message": "Initialization already in progress"
        }, status_code=409)
    
    # Reset components
    with _components_lock:
        components.is_initialized = False
        components.initialization_error = None
    
    # Start new initialization
    background_tasks.add_task(initialize_voice_assistant_deferred, app)
    return {"status": "started", "message": "Re-initialization started in background"}

@app.post("/api/set-language")
async def set_language(request: LanguageRequest):
    """Set the current language for voice assistant responses.
    
    This endpoint allows the frontend to change the language used for
    TTS synthesis, immediately switching to the appropriate Kokoro voice.
    """
    if not components.is_initialized:
        raise HTTPException(
            status_code=503,
            detail="Voice assistant not initialized yet. Please wait."
        )
    
    language_code = request.language
    
    # Validate language code
    from src.config.language_config import KOKORO_VOICE_MAP, LANGUAGE_NAMES
    
    if language_code not in KOKORO_VOICE_MAP:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported language code: {language_code}. Supported: {list(KOKORO_VOICE_MAP.keys())}"
        )
    
    try:
        # Update the voice assistant's current language
        components.voice_assistant.current_language = language_code
        
        # Get available voices for this language
        available_voices = components.voice_assistant.get_voices_for_language(language_code)
        language_name = LANGUAGE_NAMES.get(language_code, f"Language {language_code}")
        
        logger.info(f"🌍 Language switched to: {language_name} ({language_code}) with {len(available_voices)} voices")
        
        return {
            "status": "success",
            "language": {
                "code": language_code,
                "name": language_name,
                "available_voices": available_voices
            },
            "message": f"Language switched to {language_name}"
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to set language: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to set language: {str(e)}"
        )

@app.get("/api/current-language")
async def get_current_language():
    """Get the current language setting."""
    if not components.is_initialized:
        raise HTTPException(
            status_code=503,
            detail="Voice assistant not initialized yet. Please wait."
        )
    
    try:
        from src.config.language_config import LANGUAGE_NAMES
        
        current_lang = components.voice_assistant.current_language
        available_voices = components.voice_assistant.get_voices_for_language(current_lang)
        language_name = LANGUAGE_NAMES.get(current_lang, f"Language {current_lang}")
        
        return {
            "language": {
                "code": current_lang,
                "name": language_name,
                "available_voices": available_voices
            }
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to get current language: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get current language: {str(e)}"
        )

@app.get("/api/supported-languages")
async def get_supported_languages():
    """Get all supported languages and their available voices."""
    try:
        from src.config.language_config import KOKORO_VOICE_MAP, LANGUAGE_NAMES
        
        languages = []
        for code, voices in KOKORO_VOICE_MAP.items():
            languages.append({
                "code": code,
                "name": LANGUAGE_NAMES.get(code, f"Language {code}"),
                "available_voices": voices,
                "voice_count": len(voices)
            })
        
        return {
            "supported_languages": languages,
            "total_languages": len(languages)
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to get supported languages: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get supported languages: {str(e)}"
        )

# WhisperLive API routes
from src.api.whisperlive import router as whisperlive_router, set_callback_handler
app.include_router(whisperlive_router)


# Optional: serve frontend
_frontend_dist = Path(__file__).parent / "frontend" / "dist"
if _frontend_dist.exists():
    app.mount("/", StaticFiles(directory=_frontend_dist, html=True), name="spa")

# Entry point
if __name__ == "__main__":
    import uvicorn
    
    # Parse command line arguments for direct script usage
    args = parse_args()
    if args.log_level:
        # Re-setup logging with command line level (overrides environment)
        setup_logging(args.log_level)
        logger.info(f"Log level updated to {args.log_level} from command line")
    
    print("🚀 Starting FastRTC Voice Assistant Server...")
    print(f"📡 Server will be available at http://{args.host}:{args.port}")
    print("🎤 WebRTC endpoint will be mounted at /assistant after initialization")
    print("💡 Check /health for component status")
    print(f"📊 Log level: {os.getenv('LOG_LEVEL', 'INFO')}")
    
    # Show threading configuration
    if os.getenv('USE_THREADING_PIPELINE', 'false').lower() == 'true':
        print("🧵 Threading pipeline: ENABLED")
        print(f"🔄 Async fallback: {'DISABLED' if os.getenv('THREADING_FALLBACK_TO_ASYNC', 'true').lower() == 'false' else 'ENABLED'}")
    else:
        print("🔄 Using async pipeline (threading disabled)")
    
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level=log_level.lower(), # Use the parsed log level
        access_log=True
    )