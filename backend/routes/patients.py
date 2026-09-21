from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from typing import Optional
from supabase import Client
from auth import get_user_supabase_client, get_current_user

router = APIRouter(prefix="/patients", tags=["patients"])


class PatientCreate(BaseModel):
    full_name: str
    age: Optional[int] = None
    gender: Optional[str] = None
    diagnosis: Optional[str] = None


class PatientUpdate(BaseModel):
    full_name: Optional[str] = None
    age: Optional[int] = None
    gender: Optional[str] = None
    diagnosis: Optional[str] = None


# ── Patient CRUD ─────────────────────────────────────────────────────────────

@router.get("/")
def list_patients(
    current_user=Depends(get_current_user),
    db: Client = Depends(get_user_supabase_client),
):
    try:
        response = db.table("patients").select("*").execute()
        return response.data
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch patients: {str(e)}",
        )


@router.get("/{patient_id}")
def get_patient(
    patient_id: str,
    current_user=Depends(get_current_user),
    db: Client = Depends(get_user_supabase_client),
):
    try:
        response = db.table("patients").select("*").eq("id", patient_id).execute()
        if not response.data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Patient not found or access denied",
            )
        return response.data[0]
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch patient: {str(e)}",
        )


@router.post("/")
def create_patient(
    patient: PatientCreate,
    current_user=Depends(get_current_user),
    db: Client = Depends(get_user_supabase_client),
):
    try:
        data = {
            "user_id": current_user.id,
            "full_name": patient.full_name,
            "age": patient.age,
            "gender": patient.gender,
            "diagnosis": patient.diagnosis,
        }
        response = db.table("patients").insert(data).execute()
        return response.data[0]
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create patient: {str(e)}",
        )


@router.put("/{patient_id}")
def update_patient(
    patient_id: str,
    patient: PatientUpdate,
    current_user=Depends(get_current_user),
    db: Client = Depends(get_user_supabase_client),
):
    try:
        update_data = {k: v for k, v in patient.dict(exclude_unset=True).items()}
        if not update_data:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No update parameters provided",
            )
        response = (
            db.table("patients").update(update_data).eq("id", patient_id).execute()
        )
        if not response.data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Patient not found or access denied",
            )
        return response.data[0]
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update patient: {str(e)}",
        )


@router.delete("/{patient_id}")
def delete_patient(
    patient_id: str,
    current_user=Depends(get_current_user),
    db: Client = Depends(get_user_supabase_client),
):
    try:
        response = db.table("patients").delete().eq("id", patient_id).execute()
        if not response.data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Patient not found or access denied",
            )
        return {"status": "success", "message": "Patient deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete patient: {str(e)}",
        )


# ── Clinical Notes ────────────────────────────────────────────────────────────

class NoteCreate(BaseModel):
    note_text: str


class NoteUpdate(BaseModel):
    note_text: str


@router.get("/{patient_id}/notes")
def list_patient_notes(
    patient_id: str,
    current_user=Depends(get_current_user),
    db: Client = Depends(get_user_supabase_client),
):
    try:
        pat = db.table("patients").select("id").eq("id", patient_id).execute()
        if not pat.data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Patient not found or access denied",
            )
        response = (
            db.table("notes")
            .select("*")
            .eq("patient_id", patient_id)
            .order("created_at", desc=True)
            .execute()
        )
        return response.data
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch clinical notes: {str(e)}",
        )


@router.post("/{patient_id}/notes")
def create_patient_note(
    patient_id: str,
    note: NoteCreate,
    current_user=Depends(get_current_user),
    db: Client = Depends(get_user_supabase_client),
):
    try:
        pat = db.table("patients").select("id").eq("id", patient_id).execute()
        if not pat.data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Patient not found or access denied",
            )
        data = {
            "patient_id": patient_id,
            "created_by": current_user.id,
            "note_text": note.note_text,
        }
        response = db.table("notes").insert(data).execute()
        return response.data[0]
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save note: {str(e)}",
        )


@router.put("/notes/{note_id}")
def update_patient_note(
    note_id: str,
    note: NoteUpdate,
    current_user=Depends(get_current_user),
    db: Client = Depends(get_user_supabase_client),
):
    try:
        response = (
            db.table("notes")
            .update({"note_text": note.note_text})
            .eq("id", note_id)
            .execute()
        )
        if not response.data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Note not found or access denied",
            )
        return response.data[0]
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update note: {str(e)}",
        )


@router.delete("/notes/{note_id}")
def delete_patient_note(
    note_id: str,
    current_user=Depends(get_current_user),
    db: Client = Depends(get_user_supabase_client),
):
    try:
        response = db.table("notes").delete().eq("id", note_id).execute()
        if not response.data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Note not found or access denied",
            )
        return {"status": "success", "message": "Clinical note deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete note: {str(e)}",
        )
