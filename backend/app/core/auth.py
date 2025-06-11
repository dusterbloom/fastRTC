from fastapi import Depends, HTTPException, status
from fastapi.security.api_key import APIKeyHeader

from app.core.config import Settings

API_KEY_NAME = "X-API-Key"
api_key_header_auth = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

async def get_api_key(
    api_key_header: str = Depends(api_key_header_auth),
    settings: Settings = Depends(Settings) # Assuming Settings can be injected like this
):
    """
    Dependency function to validate API key.
    Compares the API key from the X-API-Key header against the API_KEY in settings.
    Raises HTTPException 401 if the key is missing or invalid.
    """
    if not api_key_header:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated: API key header missing",
            headers={"WWW-Authenticate": "APIKey"},
        )
    
    expected_api_key = settings.API_KEY
    if not expected_api_key:
        # This case should ideally not happen if API_KEY is mandatory in Settings
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error: API_KEY not configured",
        )

    if api_key_header != expected_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API Key",
            headers={"WWW-Authenticate": "APIKey"},
        )
    return api_key_header # Or return True, or the validated key itself