import os
import tempfile
import matplotlib.pyplot as plt
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import FileResponse
from supabase import Client
from auth import get_user_supabase_client, get_current_user

# ReportLab flowables and style imports
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, KeepTogether
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

router = APIRouter(prefix="/reports", tags=["reports"])

TEMP_DIR = r"E:\Downloads\Finalyearproject\backend\temp_uploads"
os.makedirs(TEMP_DIR, exist_ok=True)


def generate_trend_chart(uploads):
    """
    Generates a trend chart using matplotlib and returns its path.
    """
    if not uploads:
        return None
        
    # Sort uploads by time
    sorted_uploads = sorted(uploads, key=lambda x: x.get('upload_time', ''))
    
    times = []
    labels = []
    confidence_scores = []
    predictions = []
    
    for idx, u in enumerate(sorted_uploads):
        try:
            dt = datetime.fromisoformat(u['upload_time'].replace('Z', '+00:00'))
            times.append(dt.strftime('%m/%d %H:%M'))
        except Exception:
            times.append(f"Upload {idx+1}")
            
        pred = u.get('prediction', 'no_seizure')
        conf = u.get('confidence_score', 0.0)
        
        predictions.append(1 if pred == 'seizure' else 0)
        confidence_scores.append(conf if pred == 'seizure' else (100.0 - conf))
        
    plt.figure(figsize=(7, 3))
    
    # Plot seizure occurrence
    colors_list = ['red' if p == 1 else 'blue' for p in predictions]
    plt.scatter(times, confidence_scores, c=colors_list, s=80, zorder=3, 
                label='Seizure' if 1 in predictions else 'No Seizure')
    plt.plot(times, confidence_scores, color='gray', linestyle='--', alpha=0.5, zorder=2)
    
    plt.title("EEG Analysis Timeline & Confidence", fontsize=12, fontweight='bold', pad=10)
    plt.ylabel("Seizure Probability / Confidence (%)", fontsize=10)
    plt.xlabel("Upload Date & Time", fontsize=10)
    plt.xticks(rotation=30, ha='right', fontsize=8)
    plt.ylim(0, 105)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    
    # Save to temp image
    fd, path = tempfile.mkstemp(suffix=".png", dir=TEMP_DIR)
    os.close(fd)
    plt.savefig(path, dpi=150)
    plt.close()
    
    return path

@router.get("/pdf/{patient_id}")
def download_patient_report(
    patient_id: str,
    current_user=Depends(get_current_user),
    db: Client = Depends(get_user_supabase_client),
):
    """
    Generates and returns a PDF medical report for a specific patient.
    """
    try:
        # 1. Fetch Patient Details
        pat_res = db.table("patients").select("*").eq("id", patient_id).execute()
        if not pat_res.data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Patient not found or access denied"
            )
        patient = pat_res.data[0]
        
        # 2. Fetch Patient EEG Scan Logs
        upload_res = db.table("eeg_uploads").select("*").eq("patient_id", patient_id).order("upload_time", desc=True).execute()
        uploads = upload_res.data
        
        # 3. Fetch Patient Notes
        notes_res = db.table("notes").select("*").eq("patient_id", patient_id).order("created_at", desc=True).execute()
        notes = notes_res.data
        
        # 4. Generate Trend Chart Image
        chart_path = generate_trend_chart(uploads)
        
        # 5. Create PDF
        fd, pdf_path = tempfile.mkstemp(suffix=".pdf", dir=TEMP_DIR)
        os.close(fd)
        
        # Build Document
        doc = SimpleDocTemplate(
            pdf_path,
            pagesize=letter,
            rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36
        )
        
        styles = getSampleStyleSheet()
        
        # Custom styles
        title_style = ParagraphStyle(
            'TitleStyle',
            parent=styles['Heading1'],
            fontName='Helvetica-Bold',
            fontSize=22,
            textColor=colors.HexColor('#1A365D'),
            spaceAfter=15
        )
        
        h2_style = ParagraphStyle(
            'H2Style',
            parent=styles['Heading2'],
            fontName='Helvetica-Bold',
            fontSize=14,
            textColor=colors.HexColor('#2C5282'),
            spaceBefore=15,
            spaceAfter=8
        )
        
        body_style = ParagraphStyle(
            'BodyStyle',
            parent=styles['Normal'],
            fontName='Helvetica',
            fontSize=10,
            textColor=colors.HexColor('#2D3748'),
            leading=14
        )
        
        bold_body_style = ParagraphStyle(
            'BoldBodyStyle',
            parent=body_style,
            fontName='Helvetica-Bold'
        )
        
        story = []
        
        # Header / Title block
        story.append(Paragraph("Smart Seizure Detection & Patient Monitoring Platform", ParagraphStyle('Sub', fontName='Helvetica-Oblique', fontSize=8, textColor=colors.gray, spaceAfter=5)))
        story.append(Paragraph("Patient Clinical Monitoring Summary", title_style))
        story.append(Spacer(1, 10))
        
        # Patient Details Table
        age_str = str(patient.get('age', 'N/A'))
        gender_str = patient.get('gender', 'N/A')
        diagnosis_str = patient.get('diagnosis', 'No chronic diagnosis loaded.')
        
        pat_data = [
            [Paragraph("Patient Name:", bold_body_style), Paragraph(patient.get('full_name', 'N/A'), body_style),
             Paragraph("Patient ID:", bold_body_style), Paragraph(str(patient['id'])[:8] + "...", body_style)],
            [Paragraph("Age:", bold_body_style), Paragraph(age_str, body_style),
             Paragraph("Gender:", bold_body_style), Paragraph(gender_str, body_style)],
            [Paragraph("Diagnosis:", bold_body_style), Paragraph(diagnosis_str, body_style), "", ""]
        ]
        
        pat_table = Table(pat_data, colWidths=[90, 170, 90, 170])
        pat_table.setStyle(TableStyle([
            ('SPAN', (1, 2), (3, 2)),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('LINEBELOW', (0, 0), (-1, -1), 0.5, colors.HexColor('#E2E8F0')),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
        ]))
        story.append(pat_table)
        story.append(Spacer(1, 15))
        
        # Stats summary block
        total_scans = len(uploads)
        seizures_detected = sum(1 for u in uploads if u['prediction'] == 'seizure')
        last_scan = uploads[0]['upload_time'][:10] if uploads else "N/A"
        
        stats_data = [
            [Paragraph("Clinical Statistics", ParagraphStyle('StatsTitle', fontName='Helvetica-Bold', fontSize=11, textColor=colors.HexColor('#1A365D'))), "", ""],
            [Paragraph(f"<b>Total Scans:</b> {total_scans}", body_style),
             Paragraph(f"<b>Seizures Detected:</b> {seizures_detected}", body_style),
             Paragraph(f"<b>Latest Scan:</b> {last_scan}", body_style)]
        ]
        stats_table = Table(stats_data, colWidths=[180, 180, 180])
        stats_table.setStyle(TableStyle([
            ('SPAN', (0, 0), (2, 0)),
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#F7FAFC')),
            ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#CBD5E0')),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('PADDING', (0, 0), (-1, -1), 8),
        ]))
        story.append(stats_table)
        story.append(Spacer(1, 15))
        
        # Trend Chart
        if chart_path and os.path.exists(chart_path):
            story.append(Paragraph("Seizure Occurrence Trend", h2_style))
            story.append(Image(chart_path, width=480, height=200))
            story.append(Spacer(1, 15))
            
        # Recent Scan Logs Table
        story.append(Paragraph("Recent EEG Upload Analysis Logs", h2_style))
        log_data = [[
            Paragraph("<b>Upload Date</b>", bold_body_style),
            Paragraph("<b>Model</b>", bold_body_style),
            Paragraph("<b>Prediction</b>", bold_body_style),
            Paragraph("<b>Confidence</b>", bold_body_style)
        ]]
        
        for u in uploads[:8]: # List top 8
            dt_str = u['upload_time'][:16].replace('T', ' ')
            pred_cell = Paragraph(f"<font color='red'><b>SEIZURE</b></font>" if u['prediction'] == 'seizure' else "No Seizure", body_style)
            log_data.append([
                Paragraph(dt_str, body_style),
                Paragraph(u['model_used'].upper(), body_style),
                pred_cell,
                Paragraph(f"{u['confidence_score']:.1f}%", body_style)
            ])
            
        log_table = Table(log_data, colWidths=[150, 110, 130, 130])
        log_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#EDF2F7')),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
            ('TOPPADDING', (0, 0), (-1, -1), 5),
            ('LINEBELOW', (0, 0), (-1, -1), 0.5, colors.HexColor('#E2E8F0')),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))
        story.append(log_table)
        story.append(Spacer(1, 15))
        
        # Notes Section
        if notes:
            story.append(Paragraph("Clinical Visit & Progress Notes", h2_style))
            notes_story = []
            for n in notes[:5]:
                dt_str = n['created_at'][:16].replace('T', ' ')
                note_text = n['note_text']
                notes_story.append(Paragraph(f"<b>{dt_str}</b>: {note_text}", body_style))
                notes_story.append(Spacer(1, 4))
            story.append(KeepTogether(notes_story))
            
        # Build PDF
        doc.build(story)
        
        # Cleanup chart image
        if chart_path and os.path.exists(chart_path):
            os.remove(chart_path)
            
        # Return file response
        filename = f"report_{patient.get('full_name', 'patient').replace(' ', '_')}.pdf"
        return FileResponse(
            pdf_path,
            media_type="application/pdf",
            filename=filename,
            background=None # Let FastAPI cleanup
        )
        
    except HTTPException:
        # Cleanup
        if 'chart_path' in locals() and chart_path and os.path.exists(chart_path):
            os.remove(chart_path)
        raise
    except Exception as e:
        if 'chart_path' in locals() and chart_path and os.path.exists(chart_path):
            os.remove(chart_path)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Report generation failed: {str(e)}"
        )
