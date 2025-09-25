import os
from fastapi import Security, HTTPException, status
from fastapi.security import APIKeyHeader

API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)
API_KEY = os.getenv("API_KEY")


async def get_api_key(api_key_header: str = Security(API_KEY_HEADER)):
    print(f"API_KEY: {API_KEY}")
    print(f"api_key_header: {api_key_header}")
    print(f"API_KEY == api_key_header: {API_KEY == api_key_header}")
    """
    Dependency to validate the API key from the X-API-Key header.
    """
    if not API_KEY:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="API key not configured on server",
        )

    if api_key_header == API_KEY:
        return api_key_header
    else:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Could not validate credentials",
        )
