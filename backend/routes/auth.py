import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from config import SUPABASE_URL, SUPABASE_SERVICE_KEY

router = APIRouter(prefix="/auth", tags=["auth"])


def _admin_headers() -> dict:
    return {
        "apikey": SUPABASE_SERVICE_KEY,
        "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
        "Content-Type": "application/json",
    }


def _check_config():
    if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
        raise HTTPException(
            status_code=500,
            detail="Supabase service credentials are not configured on the server."
        )


class RegisterRequest(BaseModel):
    email: str
    password: str
    full_name: str
    role: str = "doctor"


@router.post("/register")
async def register_user(payload: RegisterRequest):
    """
    Register a new user via the Supabase Admin API (service_role key).
    Sets email_confirm=True so the user can log in immediately — no inbox
    verification link required.
    If the email already exists but is unconfirmed, the account is confirmed
    and the password is updated so the user can log in right away.
    """
    _check_config()
    headers = _admin_headers()

    async with httpx.AsyncClient() as client:
        # ── Attempt to create a brand-new user ──────────────────────────────
        create_resp = await client.post(
            f"{SUPABASE_URL}/auth/v1/admin/users",
            json={
                "email": payload.email,
                "password": payload.password,
                "email_confirm": True,
                "user_metadata": {
                    "full_name": payload.full_name,
                    "role": payload.role,
                },
            },
            headers=headers,
        )

        # ── If email already exists (422), find the user and confirm them ───
        if create_resp.status_code == 422:
            # List users and find by email
            list_resp = await client.get(
                f"{SUPABASE_URL}/auth/v1/admin/users",
                headers=headers,
            )
            users = list_resp.json().get("users", [])
            existing = next(
                (u for u in users if u.get("email") == payload.email), None
            )

            if not existing:
                raise HTTPException(
                    status_code=409,
                    detail="Email already registered. Please log in instead."
                )

            user_id = existing["id"]
            already_confirmed = bool(existing.get("email_confirmed_at"))

            if already_confirmed:
                raise HTTPException(
                    status_code=409,
                    detail="Email already registered and confirmed. Please log in."
                )

            # Confirm + update password for previously-unconfirmed account
            patch_resp = await client.put(
                f"{SUPABASE_URL}/auth/v1/admin/users/{user_id}",
                json={
                    "email_confirm": True,
                    "password": payload.password,
                    "user_metadata": {
                        "full_name": payload.full_name,
                        "role": payload.role,
                    },
                },
                headers=headers,
            )
            if patch_resp.status_code != 200:
                raise HTTPException(
                    status_code=patch_resp.status_code,
                    detail=patch_resp.text
                )

            return {
                "message": "Account confirmed. You can now log in.",
                "user_id": user_id,
                "email": payload.email,
            }

        # ── Any other non-2xx error ──────────────────────────────────────────
        if create_resp.status_code not in (200, 201):
            try:
                body = create_resp.json()
                detail = body.get("msg") or body.get("message") or create_resp.text
            except Exception:
                detail = create_resp.text
            raise HTTPException(status_code=create_resp.status_code, detail=detail)

    user_data = create_resp.json()
    return {
        "message": "Registration successful. You can now log in.",
        "user_id": user_data.get("id"),
        "email": user_data.get("email"),
    }


@router.post("/confirm-all")
async def confirm_all_users():
    """
    Dev-only utility: force-confirm all unconfirmed user emails.
    Useful when accounts were created before auto-confirmation was enabled.
    """
    _check_config()
    headers = _admin_headers()

    async with httpx.AsyncClient() as client:
        list_resp = await client.get(
            f"{SUPABASE_URL}/auth/v1/admin/users", headers=headers
        )
        users = list_resp.json().get("users", [])
        confirmed = []
        failed = []

        for user in users:
            if not user.get("email_confirmed_at"):
                uid = user["id"]
                patch = await client.put(
                    f"{SUPABASE_URL}/auth/v1/admin/users/{uid}",
                    json={"email_confirm": True},
                    headers=headers,
                )
                if patch.status_code == 200:
                    confirmed.append(user["email"])
                else:
                    failed.append(user["email"])

    return {
        "confirmed": confirmed,
        "failed": failed,
        "message": f"{len(confirmed)} user(s) confirmed, {len(failed)} failed."
    }
