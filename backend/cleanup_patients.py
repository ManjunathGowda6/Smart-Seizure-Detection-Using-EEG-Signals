"""
Utility: List and delete all test/mistaken patient records from Supabase.
Run with: python cleanup_patients.py
"""
from dotenv import load_dotenv
import os, json
from supabase import create_client

load_dotenv()
url = os.getenv('SUPABASE_URL')
service_key = os.getenv('SUPABASE_SERVICE_KEY')

db = create_client(url, service_key)

# ── List all patients ──────────────────────────────────────────────────────
res = db.table('patients').select('id, full_name, age, gender, diagnosis, created_at').execute()
patients = res.data

if not patients:
    print("No patients found in the database.")
    exit(0)

print(f"Found {len(patients)} patient(s):\n")
for i, p in enumerate(patients):
    print(f"  [{i}] id={p['id']}")
    print(f"       name={p['full_name']} | age={p['age']} | gender={p['gender']}")
    print(f"       created={p['created_at'][:16]}")
    print()

# ── Delete all of them ────────────────────────────────────────────────────
confirm = input("Delete ALL patients listed above? (yes/no): ").strip().lower()

if confirm != 'yes':
    print("Aborted. No records deleted.")
    exit(0)

deleted = 0
for p in patients:
    try:
        db.table('patients').delete().eq('id', p['id']).execute()
        print(f"  Deleted: {p['full_name']} ({p['id'][:8]}...)")
        deleted += 1
    except Exception as e:
        print(f"  FAILED to delete {p['full_name']}: {e}")

print(f"\nDone. {deleted}/{len(patients)} patient(s) deleted.")
