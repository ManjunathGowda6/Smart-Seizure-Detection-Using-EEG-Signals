from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from supabase import create_client, Client
import config

security = HTTPBearer()


def get_supabase_client() -> Client:
    """
    Returns an anon-key Supabase client.
    Used only for auth operations (auth.get_user to verify JWTs).
    """
    if not config.SUPABASE_URL or not config.SUPABASE_KEY:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Supabase configuration is missing in backend/.env.",
        )
    return create_client(config.SUPABASE_URL, config.SUPABASE_KEY)


def get_service_client() -> Client:
    """
    Returns a Supabase client using the service_role key.

    Why use service_role in the backend?
    ─────────────────────────────────────
    RLS (Row Level Security) is designed for direct browser/app → Supabase access.
    When requests go through a FastAPI backend, the backend is the trusted
    intermediary that has ALREADY authenticated the user via JWT (get_current_user).

    Using service_role here lets us bypass RLS and enforce ownership ourselves
    (always filtering/inserting with the verified current_user.id).  This is the
    recommended pattern for server-side Supabase integrations.

    The service_role key is kept secret in backend/.env and never exposed to
    the frontend.
    """
    if not config.SUPABASE_URL or not config.SUPABASE_SERVICE_KEY:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="SUPABASE_SERVICE_KEY is missing in backend/.env.",
        )
    return create_client(config.SUPABASE_URL, config.SUPABASE_SERVICE_KEY)


# Alias used in route dependencies
get_user_supabase_client = get_service_client


def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    supabase: Client = Depends(get_supabase_client),
):
    """
    Validates the bearer JWT and returns the Supabase User object.
    All route handlers that depend on this will 401 if the token is missing
    or invalid.
    """
    token = credentials.credentials
    try:
        response = supabase.auth.get_user(token)
        if not response or not response.user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Could not validate credentials: user not found.",
            )
        return response.user
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid or expired authorization token: {str(e)}",
        )
