import os
import uuid
import shutil
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, status
from typing import Optional
from supabase import Client
from auth import get_supabase_client, get_user_supabase_client, get_current_user
from ml.predict import predict_eeg_file
import config

router = APIRouter(prefix="/uploads", tags=["uploads"])

TEMP_DIR = r"E:\Downloads\Finalyearproject\backend\temp_uploads"
os.makedirs(TEMP_DIR, exist_ok=True)


@router.post("/analyze")
async def analyze_eeg(
    file: UploadFile = File(...),
    patient_id: str = Form(...),
    model_used: str = Form("svm"),
    current_user=Depends(get_current_user),
    db: Client = Depends(get_user_supabase_client),
):
    """
    Mode A: Patient-linked EEG upload.
    Saves, runs prediction, uploads to Supabase Storage, logs to eeg_uploads DB table,
    and inserts alerts if seizure is detected.
    """
    # 1. Verify patient belongs to the user
    try:
        pat_check = db.table("patients").select("*").eq("id", patient_id).execute()
        if not pat_check.data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Patient not found or access denied"
            )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Patient validation failed: {str(e)}"
        )

    # 2. Save file locally for processing
    file_id = str(uuid.uuid4())
    file_ext = os.path.splitext(file.filename)[1]
    if not file_ext:
        file_ext = ".edf"  # Default fallback
        
    temp_file_name = f"{file_id}{file_ext}"
    temp_path = os.path.join(TEMP_DIR, temp_file_name)
    
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    try:
        # 3. Run prediction pipeline
        results = predict_eeg_file(temp_path, model_name=model_used)
        
        # 4. Upload file to Supabase Storage
        storage_path = f"uploads/{file_id}{file_ext}"
        with open(temp_path, "rb") as f:
            file_data = f.read()
            
        # Use storage API
        db_storage = get_supabase_client()  # Use system client to bypass strict postgrest authorization for storage
        db_storage.storage.from_(config.STORAGE_BUCKET_NAME).upload(
            path=storage_path,
            file=file_data,
            file_options={"content-type": "application/octet-stream"}
        )
        
        # Get public url
        file_url = db_storage.storage.from_(config.STORAGE_BUCKET_NAME).get_public_url(storage_path)
        
        # 5. Insert record into eeg_uploads DB table
        upload_record = {
            "patient_id": patient_id,
            "file_url": file_url,
            "prediction": results["prediction"],
            "confidence_score": results["confidence_score"],
            "model_used": model_used,
            "is_quick_scan": False
        }
        upload_res = db.table("eeg_uploads").insert(upload_record).execute()
        
        # 6. Insert alert if seizure detected
        if results["prediction"] == "seizure":
            alert_record = {
                "patient_id": patient_id,
                "message": f"Seizure activity detected using {model_used.upper()} model. Confidence: {results['confidence_score']:.1f}%",
                "is_read": False
            }
            db.table("alerts").insert(alert_record).execute()
            
        # Cleanup local file
        if os.path.exists(temp_path):
            os.remove(temp_path)
            
        return {
            "upload": upload_res.data[0],
            "prediction": results["prediction"],
            "confidence_score": results["confidence_score"],
            "seizure_segment": results["seizure_segment"],
            "visualization_data": results["visualization_data"]
        }
        
    except Exception as e:
        # Cleanup on failure
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Analysis failed: {str(e)}"
        )

@router.post("/quick-scan")
async def quick_scan_eeg(
    file: UploadFile = File(...),
    model_used: str = Form("svm")
):
    """
    Mode B: Quick Scan (no patient details).
    Saves file in a temporary folder, runs inference, and returns results + temp_id.
    """
    file_id = str(uuid.uuid4())
    file_ext = os.path.splitext(file.filename)[1]
    if not file_ext:
        file_ext = ".edf"
        
    temp_file_name = f"{file_id}{file_ext}"
    temp_path = os.path.join(TEMP_DIR, temp_file_name)
    
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    try:
        # Run prediction
        results = predict_eeg_file(temp_path, model_name=model_used)
        
        # Return results along with temp_id (the filename) so it can be saved later
        return {
            "temp_id": temp_file_name,
            "prediction": results["prediction"],
            "confidence_score": results["confidence_score"],
            "seizure_segment": results["seizure_segment"],
            "visualization_data": results["visualization_data"]
        }
    except Exception as e:
        # Cleanup
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Quick scan failed: {str(e)}"
        )

@router.post("/save-quick-scan")
def save_quick_scan(
    temp_id: str = Form(...),
    patient_id: str = Form(...),
    prediction: str = Form(...),
    confidence_score: float = Form(...),
    model_used: str = Form(...),
    current_user=Depends(get_current_user),
    db: Client = Depends(get_user_supabase_client),
):
    """
    Mode B Follow-up: Saves a quick scan temp file to a patient's record.
    Uploads to storage, inserts in database, adds alert if seizure, and deletes temp file.
    """
    temp_path = os.path.join(TEMP_DIR, temp_id)
    if not os.path.exists(temp_path):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Temporary scan file not found or already processed"
        )
        
    try:
        # Verify patient ownership
        pat_check = db.table("patients").select("*").eq("id", patient_id).execute()
        if not pat_check.data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Patient not found or access denied"
            )
            
        # Upload to Supabase Storage
        file_id = str(uuid.uuid4())
        file_ext = os.path.splitext(temp_id)[1]
        storage_path = f"uploads/{file_id}{file_ext}"
        
        with open(temp_path, "rb") as f:
            file_data = f.read()
            
        db_storage = get_supabase_client()
        db_storage.storage.from_(config.STORAGE_BUCKET_NAME).upload(
            path=storage_path,
            file=file_data,
            file_options={"content-type": "application/octet-stream"}
        )
        
        file_url = db_storage.storage.from_(config.STORAGE_BUCKET_NAME).get_public_url(storage_path)
        
        # Log to eeg_uploads table
        upload_record = {
            "patient_id": patient_id,
            "file_url": file_url,
            "prediction": prediction,
            "confidence_score": confidence_score,
            "model_used": model_used,
            "is_quick_scan": True
        }
        upload_res = db.table("eeg_uploads").insert(upload_record).execute()
        
        # Log alert if prediction is seizure
        if prediction == "seizure":
            alert_record = {
                "patient_id": patient_id,
                "message": f"Seizure activity logged via Quick Scan. Model: {model_used.upper()}. Confidence: {confidence_score:.1f}%",
                "is_read": False
            }
            db.table("alerts").insert(alert_record).execute()
            
        # Delete temp local file
        os.remove(temp_path)
        
        return upload_res.data[0]
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save scan record: {str(e)}"
        )
