from fastapi import APIRouter, Depends, HTTPException, status
from supabase import Client
from auth import get_user_supabase_client, get_current_user

router = APIRouter(prefix="/alerts", tags=["alerts"])


@router.get("/")
def get_alerts(
    current_user=Depends(get_current_user),
    db: Client = Depends(get_user_supabase_client),
):
    """
    Lists all alerts for patients belonging to the logged-in user, ordered by time.
    """
    try:
        response = (
            db.table("alerts")
            .select("*, patients(full_name)")
            .order("triggered_at", desc=True)
            .execute()
        )
        return response.data
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch alerts: {str(e)}",
        )


@router.put("/{alert_id}/read")
def mark_as_read(
    alert_id: str,
    current_user=Depends(get_current_user),
    db: Client = Depends(get_user_supabase_client),
):
    """
    Marks a specific alert as read.
    """
    try:
        response = (
            db.table("alerts")
            .update({"is_read": True})
            .eq("id", alert_id)
            .execute()
        )
        if not response.data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Alert not found or access denied",
            )
        return response.data[0]
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update alert: {str(e)}",
        )
