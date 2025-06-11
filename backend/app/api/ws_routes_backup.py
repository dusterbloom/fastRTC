import asyncio
import io
import json
import uuid
from typing import List, Optional

import numpy as np
from fastapi import (APIRouter, Depends, HTTPException, WebSocket,
                     WebSocketDisconnect, status) # Added status for WebSocket close codes

from app.core.config import Settings
# Assuming get_settings_instance is defined in backend.app.core.config
# If not, Settings() can be instantiated directly or via a local helper.
from app.core.config import get_settings_instance
from app.schemas.ws_schemas import (WSError, WSAuthRequest,
                                            WSLLMResponse, WSSSTTFinal,
                                            WSSessionStarted, WSTTSComplete,
                                            WSTextInput)
from app.services.assistant_service import AssistantService
# Import the global instance from main.py
from app.main import assistant_service_instance
from fastrtc_voice_assistant.src.core.interfaces import TranscriptionResult
from fastrtc_voice_assistant.src.core.voice_assistant import VoiceAssistant

router = APIRouter(prefix="/ws", tags=["WebSocket Endpoints"])

# Dependency for AssistantService
def get_assistant_service() -> AssistantService:
    """
    Dependency to get the global instance of AssistantService.
    """
    if assistant_service_instance is None:
        # This will be caught by FastAPI's dependency injection system
        # if the service isn't available, leading to a connection error.
        raise HTTPException(status_code=503, detail="Assistant service not available.")
    return assistant_service_instance

@router.websocket("/converse")
async def websocket_converse_endpoint(
    websocket: WebSocket,
    service: AssistantService = Depends(get_assistant_service),
    settings: Settings = Depends(get_settings_instance),
):
    await websocket.accept()
    assistant: Optional[VoiceAssistant] = None
    session_id: Optional[str] = None
    user_id: Optional[str] = None # Assuming WSAuthRequest will provide this
    authenticated = False
    audio_buffer = bytearray()
    client_audio_format = "float32"  # Default, to be updated from auth
    client_sample_rate = 16000  # Default, to be updated from auth

    try:
        # 1. Authentication and Initialization
        try:
            auth_payload = await websocket.receive_json()
            auth_request = WSAuthRequest(**auth_payload)

            # Validate API key
            if not settings.API_KEY or auth_request.api_key != settings.API_KEY:
                error_msg = "Authentication failed: Invalid API key."
                await websocket.send_json(WSError(message=error_msg).model_dump())
                await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
                return

            user_id = auth_request.user_id # Assuming user_id is part of WSAuthRequest
            session_id = str(uuid.uuid4())
            
            client_audio_format = auth_request.audio_format
            client_sample_rate = auth_request.sample_rate

            assistant = await service.create_assistant_for_ws(session_id=session_id, user_id=user_id)
            
            # TODO: Get these values from assistant/TTS engine configuration
            expected_tts_audio_format = "float32" 
            expected_tts_sample_rate = 24000 # Common TTS output rate

            await websocket.send_json(
                WSSessionStarted(
                    session_id=session_id,
                    user_id=user_id,
                    expected_tts_audio_format=expected_tts_audio_format,
                    expected_tts_sample_rate=expected_tts_sample_rate,
                ).model_dump()
            )
            authenticated = True
            print(f"WebSocket session {session_id} started for user {user_id}")

        except json.JSONDecodeError:
            error_msg = "Authentication failed: Invalid JSON format."
            await websocket.send_json(WSError(message=error_msg).model_dump())
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
            return
        except Exception as e: # Catch other auth/init errors (e.g., Pydantic validation)
            error_msg = f"Authentication failed: {str(e)}"
            await websocket.send_json(WSError(message=error_msg).model_dump())
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
            return

        # 2. Main Message Handling Loop (if authenticated)
        if authenticated and assistant:
            while True:
                received_message = await websocket.receive()

                if received_message["type"] == "websocket.disconnect":
                    # This will be caught by the outer WebSocketDisconnect exception
                    break 
                
                text_data = received_message.get("text")
                bytes_data = received_message.get("bytes")

                if text_data:
                    message_json = json.loads(text_data)
                    message_type = message_json.get("type")

                    if message_type == "text_input":
                        parsed_input = WSTextInput(**message_json)
                        llm_response_text = await assistant.get_llm_response_smart(parsed_input.text)
                        await websocket.send_json(WSLLMResponse(text=llm_response_text).model_dump())

                        # TTS Streaming
                        async for sr, chunk_array in assistant.stream_tts_synthesis(
                            llm_response_text, language_code=assistant.current_language
                        ):
                            # Convert numpy array (float32) to bytes for WebSocket
                            audio_bytes = chunk_array.astype(np.float32).tobytes()
                            await websocket.send_bytes(audio_bytes)
                        await websocket.send_json(WSTTSComplete().model_dump())

                    elif message_type == "audio_complete":
                        if audio_buffer:
                            # TODO: Dynamically determine dtype based on client_audio_format
                            # For now, assuming float32 as per instructions
                            if client_audio_format != "float32":
                                print(f"Warning: Client audio format is {client_audio_format}, but processing as float32.")
                            
                            audio_np = np.frombuffer(audio_buffer, dtype=np.float32)
                            
                            transcription_result: TranscriptionResult = await assistant.stt_engine.transcribe_with_sample_rate(
                                audio_np, client_sample_rate
                            )
                            await websocket.send_json(
                                WSSTTFinal(
                                    text=transcription_result.text,
                                    detected_language=transcription_result.detected_language,
                                    kokoro_language=transcription_result.kokoro_language,
                                ).model_dump()
                            )
                            audio_buffer.clear()
                        else:
                            print(f"Session {session_id}: Audio complete received with empty buffer.")
                    else:
                        await websocket.send_json(WSError(message="Unknown JSON message type").model_dump())
                
                elif bytes_data:
                    audio_buffer.extend(bytes_data)
                    # Optional: Log receipt
                    # print(f"Session {session_id}: Received {len(bytes_data)} audio bytes, buffer size: {len(audio_buffer)}")

    except WebSocketDisconnect:
        print(f"WebSocket session {session_id} disconnected by client.")
    except Exception as e:
        # Generic error handling for unexpected issues during the session
        error_msg = f"An unexpected error occurred in session {session_id}: {str(e)}"
        print(error_msg)
        if websocket.client_state != WebSocketDisconnect: # Check if websocket is still openable
            try:
                await websocket.send_json(WSError(message="An unexpected server error occurred.").model_dump())
                await websocket.close(code=status.WS_1011_INTERNAL_ERROR)
            except Exception as close_exc:
                print(f"Error trying to gracefully close WebSocket for session {session_id}: {close_exc}")
    finally:
        if assistant:
            await service.cleanup_assistant_for_ws(assistant)
        print(f"Cleaned up resources for WebSocket session {session_id}")