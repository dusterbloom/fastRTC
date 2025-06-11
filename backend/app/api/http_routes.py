import io
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, Query
from fastapi.responses import StreamingResponse

from backend.app.schemas.common_schemas import (
    STTResponse,
    TTSRequest,
    ConverseRequest,
    ConverseResponse,
    LanguageDetectionRequest,
    LanguageDetectionResponse,
    SessionCreateRequest,
    SessionResponse,
    SessionDeleteResponse,
    MemoryQueryRequest,
    MemoryQueryResponse,
    MemoryAddRequest,
    MemoryAddResponse,
    UserSession, # For /memory/summary query params
    MemorySummaryResponse
)
from backend.app.services.assistant_service import AssistantService
from backend.app.core.auth import get_api_key
# Remove Settings import if no longer directly used here, keep if other parts of the file need it.
# from backend.app.core.config import Settings
from backend.app.main import assistant_service_instance # Import the global instance

router = APIRouter(prefix="/api/v1", tags=["HTTP Endpoints"])

# Dependency for AssistantService
def get_assistant_service() -> AssistantService:
    """
    Dependency to get the global instance of AssistantService.
    """
    if assistant_service_instance is None:
        # This should ideally not happen if lifespan event ran correctly
        raise HTTPException(status_code=503, detail="Assistant service not available.")
    return assistant_service_instance


@router.post("/stt", response_model=STTResponse, dependencies=[Depends(get_api_key)])
async def speech_to_text(
    audio_file: UploadFile = File(...),
    language: Optional[str] = Form(None),
    service: AssistantService = Depends(get_assistant_service)
):
    """
    Converts speech audio to text.
    """
    try:
        audio_data = await audio_file.read()
        # Assuming service.process_stt is an async method
        result = await service.process_stt(audio_data=audio_data, language_hint=language)
        return result
    except Exception as e:
        # Log the exception e
        raise HTTPException(status_code=500, detail=f"STT processing failed: {str(e)}")

@router.post("/tts", response_class=StreamingResponse, dependencies=[Depends(get_api_key)])
async def text_to_speech(
    request: TTSRequest,
    service: AssistantService = Depends(get_assistant_service) # Added service dependency
):
    """
    Converts text to speech audio stream.
    """
    # TODO: Integrate with AssistantService for actual TTS.
    # This would call assistant_service.process_tts(...)
    # which would then call voice_assistant.tts_service.synthesize_speech_async_stream(...)
    
    # Placeholder TTS logic: A minimal valid WAV header for silent audio
    # RIFF header (12 bytes), Format chunk (24 bytes), Data chunk header (8 bytes)
    # Minimal WAV: 1 channel, 1 Hz sample rate, 16-bit, 0 data bytes
    dummy_audio_content = (
        b"RIFF" + (36).to_bytes(4, 'little') + b"WAVE" +  # ChunkSize (36 + data size)
        b"fmt " + (16).to_bytes(4, 'little') +             # Subchunk1Size (16 for PCM)
        (1).to_bytes(2, 'little') +                       # AudioFormat (1 for PCM)
        (1).to_bytes(2, 'little') +                       # NumChannels (1)
        (1).to_bytes(4, 'little') +                       # SampleRate (1 Hz)
        (2).to_bytes(4, 'little') +                       # ByteRate (SampleRate * NumChannels * BitsPerSample/8) = 1*1*16/8 = 2
        (2).to_bytes(2, 'little') +                       # BlockAlign (NumChannels * BitsPerSample/8) = 1*16/8 = 2
        (16).to_bytes(2, 'little') +                      # BitsPerSample (16)
        b"data" + (0).to_bytes(4, 'little')               # Subchunk2Size (0 data bytes)
    )
    return StreamingResponse(io.BytesIO(dummy_audio_content), media_type="audio/wav")

@router.post("/converse", response_model=ConverseResponse, dependencies=[Depends(get_api_key)])
async def converse_with_assistant(
    request: ConverseRequest,
    service: AssistantService = Depends(get_assistant_service)
):
    """
    Handles conversational interaction with the assistant.
    """
    try:
        # Assuming service.process_converse is an async method
        response = await service.process_converse(
            user_id=request.user_id,
            session_id=request.session_id,
            text=request.text
        )
        return response
    except Exception as e:
        # Log the exception e
        raise HTTPException(status_code=500, detail=f"Converse processing failed: {str(e)}")

@router.post("/detect_language", response_model=LanguageDetectionResponse, dependencies=[Depends(get_api_key)])
async def detect_text_language(
    request: LanguageDetectionRequest,
    service: AssistantService = Depends(get_assistant_service) # Added service dependency
):
    """
    Detects the language of the provided text.
    """
    # TODO: Integrate with AssistantService for actual language detection.
    # result = await service.detect_language(text=request.text)
    # return result
    return LanguageDetectionResponse(detected_language_code="en", kokoro_language_code="a", confidence=0.9)

# --- Session Management ---
@router.post("/session", response_model=SessionResponse, dependencies=[Depends(get_api_key)])
async def create_session_endpoint(
    request: SessionCreateRequest,
    service: AssistantService = Depends(get_assistant_service)
):
    """
    Creates a new user session.
    """
    try:
        # Assuming service.create_session is an async method
        session_info = await service.create_session(user_id=request.user_id)
        return session_info
    except Exception as e:
        # Log the exception e
        raise HTTPException(status_code=500, detail=f"Session creation failed: {str(e)}")

@router.get("/session/{session_id}", response_model=SessionResponse, dependencies=[Depends(get_api_key)])
async def get_session_info_endpoint(
    session_id: str,
    service: AssistantService = Depends(get_assistant_service) # Added service dependency
):
    """
    Retrieves information about a specific session.
    """
    # TODO: Integrate with AssistantService.get_session_info(...)
    # session_info = await service.get_session_info(session_id=session_id)
    # if not session_info:
    #     raise HTTPException(status_code=404, detail="Session not found")
    # return session_info
    return SessionResponse(session_id=session_id, user_id="default_user_http_get_placeholder")

@router.delete("/session/{session_id}", response_model=SessionDeleteResponse, dependencies=[Depends(get_api_key)]) # Or use status_code=204
async def delete_session_endpoint(
    session_id: str,
    service: AssistantService = Depends(get_assistant_service)
):
    """
    Deletes a specific session.
    """
    try:
        # Assuming service.delete_session is an async method
        deleted = await service.delete_session(session_id=session_id)
        if deleted:
            return SessionDeleteResponse(message="Session deleted successfully", session_id=session_id)
        else:
            # This case might be handled by service raising an exception for not found
            raise HTTPException(status_code=404, detail="Session not found or already deleted")
    except Exception as e: # Catch specific exceptions if service defines them
        # Log the exception e
        raise HTTPException(status_code=500, detail=f"Session deletion failed: {str(e)}")

# --- Agentic Memory Access ---
@router.post("/memory/query", response_model=MemoryQueryResponse, dependencies=[Depends(get_api_key)])
async def query_memory_endpoint(
    request: MemoryQueryRequest,
    service: AssistantService = Depends(get_assistant_service) # Added service dependency
):
    """
    Queries the agentic memory.
    """
    # TODO: Integrate with AssistantService for memory query.
    # results = await service.query_memory(user_id=request.user_id, session_id=request.session_id, query_text=request.query_text)
    # return results
    return MemoryQueryResponse(results=[])

@router.post("/memory/add", response_model=MemoryAddResponse, dependencies=[Depends(get_api_key)])
async def add_to_memory_endpoint(
    request: MemoryAddRequest,
    service: AssistantService = Depends(get_assistant_service) # Added service dependency
):
    """
    Adds an entry to the agentic memory.
    """
    # TODO: Integrate with AssistantService for adding to memory.
    # response = await service.add_to_memory(user_id=request.user_id, session_id=request.session_id, user_text=request.user_text, assistant_text=request.assistant_text)
    # return response
    return MemoryAddResponse(message="Entry added placeholder")

@router.get("/memory/summary", response_model=MemorySummaryResponse, dependencies=[Depends(get_api_key)])
async def get_memory_summary_endpoint(
    user_id: Optional[str] = Query(None), # From UserSession
    session_id: Optional[str] = Query(None), # From UserSession
    service: AssistantService = Depends(get_assistant_service) # Added service dependency
):
    """
    Retrieves a summary of the agentic memory for a user/session.
    """
    # TODO: Integrate with AssistantService for memory summary.
    # summary = await service.get_memory_summary(user_id=user_id, session_id=session_id)
    # return summary
    return MemorySummaryResponse(total_entries=0, user_id=user_id, session_id=session_id)