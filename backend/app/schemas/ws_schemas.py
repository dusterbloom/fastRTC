from typing import Optional, Union, Literal
from pydantic import BaseModel, Field

# --- WebSocket General ---
class WSBaseMessage(BaseModel):
    type: str

# --- Client to Server WebSocket Messages ---
class WSAuthRequest(WSBaseMessage):
    type: Literal["auth"] = "auth"
    api_key: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None # To resume a session
    audio_format: Optional[str] = Field("float32", description="Format of upcoming binary audio frames.")
    sample_rate: Optional[int] = Field(16000, description="Sample rate of upcoming binary audio.")

class WSAudioComplete(WSBaseMessage):
    type: Literal["audio_complete"] = "audio_complete"

class WSTextInput(WSBaseMessage):
    type: Literal["text_input"] = "text_input"
    text: str

# Union type for C2S JSON messages
ClientToServerJSONMessage = Union[WSAuthRequest, WSAudioComplete, WSTextInput]


# --- Server to Client WebSocket Messages ---
class WSSessionStarted(WSBaseMessage):
    type: Literal["session_started"] = "session_started"
    session_id: str
    user_id: str
    expected_tts_audio_format: Optional[str] = Field("float32", description="Format of upcoming binary TTS audio.")
    expected_tts_sample_rate: Optional[int] = Field(16000, description="Sample rate of upcoming binary TTS audio.")

class WSSTTPartial(WSBaseMessage):
    type: Literal["stt_partial"] = "stt_partial"
    text: str

class WSSTTFinal(WSBaseMessage):
    type: Literal["stt_final"] = "stt_final"
    text: str
    detected_language: Optional[str] = None
    kokoro_language: Optional[str] = None

class WSLLMResponse(WSBaseMessage):
    type: Literal["llm_response"] = "llm_response"
    text: str

class WSTTSComplete(WSBaseMessage):
    type: Literal["tts_complete"] = "tts_complete"

class WSError(WSBaseMessage):
    type: Literal["error"] = "error"
    message: str
    details: Optional[str] = None

# Union type for S2C JSON messages
ServerToClientJSONMessage = Union[
    WSSessionStarted, WSSTTPartial, WSSTTFinal, WSLLMResponse, WSTTSComplete, WSError
]