import os
from dotenv import load_dotenv

# Load environment variables from .env file if present
load_dotenv()

# Supabase Credentials
# The user will fill these in manually in the .env file.
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")          # anon key — used for normal DB operations
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY", "")  # service_role key — used for admin operations (e.g. auto-confirm users)
SUPABASE_JWT_SECRET = os.getenv("SUPABASE_JWT_SECRET", "")  # Used to verify user JWTs in the backend if doing manual verification

# Storage Bucket Name
STORAGE_BUCKET_NAME = "eeg-uploads"

# Local models folder
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
