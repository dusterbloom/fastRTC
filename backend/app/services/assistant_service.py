import uuid
from typing import Optional

from cachetools import LRUCache
from fastrtc_voice_assistant.src.core.voice_assistant import VoiceAssistant
from fastrtc_voice_assistant.src.config.settings import load_config

from app.core.config import Settings
from app.schemas.common_schemas import STTResponse, ConverseResponse, SessionResponse


class AssistantService:
    def __init__(self, settings: Settings):
        self.settings: Settings = settings
        self.http_cache: LRUCache[str, VoiceAssistant] = LRUCache(
            maxsize=settings.ASSISTANT_SERVICE_CACHE_SIZE
        )
        # For WebSocket assistants, they are typically created per connection
        # and not stored in a shared cache like http_cache.
        # If specific tracking is needed, it could be added here.
        # self.ws_assistants: Dict[str, VoiceAssistant] = {}

        # Load VoiceAssistant global settings
        # Assuming VoiceAssistantSettings can be instantiated directly
        # or has a method to load from its own .env or config files.
        # This might need adjustment based on how VoiceAssistantSettings is designed.
        self.voice_assistant_global_settings = load_config()

    async def get_assistant_for_http(
        self, session_id: str, user_id: Optional[str]
    ) -> VoiceAssistant:
        if session_id in self.http_cache:
            return self.http_cache[session_id]
        else:
            # Pass the loaded global settings for VoiceAssistant
            assistant = VoiceAssistant(
                session_id=session_id,
                user_id=user_id,
                settings_override=self.voice_assistant_global_settings,
            )
            await assistant.initialize_async()
            self.http_cache[session_id] = assistant
            return assistant

    async def create_assistant_for_ws(
        self, session_id: str, user_id: str
    ) -> VoiceAssistant:
        # Pass the loaded global settings for VoiceAssistant
        assistant = VoiceAssistant(
            session_id=session_id,
            user_id=user_id,
            settings_override=self.voice_assistant_global_settings,
        )
        await assistant.initialize_async()
        # These are not cached in the LRU cache as their lifecycle
        # is tied to the WebSocket connection.
        return assistant

    async def cleanup_assistant_for_ws(self, assistant: VoiceAssistant):
        await assistant.cleanup_async()

    async def startup(self):
        # Placeholder for any startup logic, e.g., pre-loading models if shared
        pass

    async def shutdown(self):
        for assistant in list(self.http_cache.values()): # Iterate over a copy
            await assistant.cleanup_async()
        self.http_cache.clear()

    async def process_stt(
        self,
        audio_data: bytes,
        language: Optional[str],
        session_id: str,
        user_id: Optional[str],
    ) -> STTResponse:
        assistant = await self.get_assistant_for_http(session_id, user_id)
        # TODO: Call assistant.stt_service.transcribe(...) or similar
        # This will likely involve:
        # transcription_result = await assistant.stt_service.transcribe_audio_bytes(
        #     audio_data, language_code=language
        # )
        # return STTResponse(text=transcription_result.text, language=transcription_result.language)
        return STTResponse(text="STT placeholder", language=language or "en")

    async def process_converse(
        self, text: str, session_id: str, user_id: str
    ) -> ConverseResponse:
        assistant = await self.get_assistant_for_http(session_id, user_id)
        # TODO: Call assistant.get_llm_response_async(...)
        # This will likely involve:
        # llm_response = await assistant.get_llm_response_async(text)
        # return ConverseResponse(
        #     response_text=llm_response.text_response, # Adjust based on actual LLM response structure
        #     audio_response_url=llm_response.audio_response_url, # If applicable
        #     session_id=session_id,
        #     user_id=user_id
        # )
        return ConverseResponse(
            response_text="Converse placeholder",
            session_id=session_id,
            user_id=user_id,
            audio_response_url=None # Placeholder
        )

    async def create_session(self, user_id: Optional[str]) -> SessionResponse:
        new_session_id = str(uuid.uuid4())
        # Optionally pre-cache an assistant instance here if desired,
        # for example, to warm it up.
        # assistant = await self.get_assistant_for_http(new_session_id, user_id)
        return SessionResponse(session_id=new_session_id, user_id=user_id)

    async def delete_session(
        self, session_id: str, user_id: Optional[str] # user_id might be used for auth/logging
    ) -> bool:
        if session_id in self.http_cache:
            assistant = self.http_cache.pop(session_id)
            await assistant.cleanup_async()
            return True
        return False