import sys
import os

# Ensure backend directory is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Import routes
from routes import patients, uploads, alerts, reports, auth, model

app = FastAPI(
    title="Smart Seizure Detection and Patient Monitoring API",
    description="Backend API for seizure detection classification, clinical note management, and PDF report compilation.",
    version="1.0.0"
)

# Configure CORS
# Allow our frontend dev server and other origins to communicate with the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, replace with specific frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router)
app.include_router(patients.router)
app.include_router(uploads.router)
app.include_router(alerts.router)
app.include_router(reports.router)
app.include_router(model.router)

@app.get("/")
def read_root():
    return {
        "status": "online",
        "service": "Smart Seizure Detection API",
        "version": "1.0.0"
    }

if __name__ == "__main__":
    # Run the server locally using uvicorn
    # In development, uvicorn backend.main:app --reload can be run from the console.
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
