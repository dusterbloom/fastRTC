"""Configuration management that works with any Pydantic version."""
from typing import Optional

try:
    # Try Pydantic v2 first
    from pydantic_settings import BaseSettings
    PYDANTIC_V2 = True
except ImportError:
    # Fall back to Pydantic v1
    from pydantic import BaseSettings
    PYDANTIC_V2 = False


class Settings(BaseSettings):
    """
    Application settings.
    These settings are loaded from environment variables or a .env file.
    """
    API_KEY: str = "dev-key-123"  # Default for development
    LOG_LEVEL: str = "INFO"
    ASSISTANT_SERVICE_CACHE_SIZE: int = 10

    if PYDANTIC_V2:
        # Pydantic v2 configuration
        model_config = {
            'env_file': '.env',
            'env_file_encoding': 'utf-8',
            'extra': 'ignore'
        }
    else:
        # Pydantic v1 configuration
        class Config:
            env_file = ".env"
            env_file_encoding = 'utf-8'
            extra = 'ignore'


def get_settings_instance() -> Settings:
    """Get settings instance."""
    return Settings()
