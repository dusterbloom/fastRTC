from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional

class Settings(BaseSettings):
    """
    Application settings.
    These settings are loaded from environment variables or a .env file.
    """
    API_KEY: str
    LOG_LEVEL: str = "INFO"
    ASSISTANT_SERVICE_CACHE_SIZE: int = 10

    # Pydantic settings configuration
    # By default, Pydantic looks for a .env file in the current working directory.
    # If the application is run from the 'backend/' directory, '.env' is correct.
    # If run from the project root, it would be 'backend/.env'.
    # For flexibility, we specify '.env'. Ensure the .env file is in the directory
    # from which the application is started (expected to be 'backend/').
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding='utf-8',
        extra='ignore'  # Ignore extra fields not defined in the model
    )

# Global instance of settings to be imported by other modules
settings = Settings()