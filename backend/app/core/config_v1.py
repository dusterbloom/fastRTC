from pydantic import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    """
    Application settings.
    These settings are loaded from environment variables or a .env file.
    """
    API_KEY: str
    LOG_LEVEL: str = "INFO"
    ASSISTANT_SERVICE_CACHE_SIZE: int = 10

    # Pydantic v1 settings configuration
    class Config:
        env_file = ".env"
        env_file_encoding = 'utf-8'
        extra = 'ignore'  # Ignore extra fields not defined in the model


def get_settings_instance() -> Settings:
    """Get settings instance."""
    return Settings()
