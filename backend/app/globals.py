"""Global instances for the FastAPI application."""

from typing import Optional
from app.services.assistant_service import AssistantService

# Global instance of AssistantService
assistant_service_instance: Optional[AssistantService] = None


def set_assistant_service(service: AssistantService) -> None:
    """Set the global assistant service instance."""
    global assistant_service_instance
    assistant_service_instance = service


def get_assistant_service() -> AssistantService:
    """Get the global assistant service instance."""
    if assistant_service_instance is None:
        raise RuntimeError("Assistant service not initialized. Make sure the app started properly.")
    return assistant_service_instance
