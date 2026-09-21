"""
Utility script to:
1. List all Supabase users and their confirmation status
2. Force-confirm any unconfirmed user emails using the Admin API
"""
from dotenv import load_dotenv
import os, httpx, json

load_dotenv()
url = os.getenv('SUPABASE_URL')
service_key = os.getenv('SUPABASE_SERVICE_KEY')

if not service_key:
    print("ERROR: SUPABASE_SERVICE_KEY is not set in backend/.env")
    exit(1)

headers = {
    'apikey': service_key,
    'Authorization': f'Bearer {service_key}',
    'Content-Type': 'application/json'
}

# List all users
resp = httpx.get(f'{url}/auth/v1/admin/users', headers=headers)
data = resp.json()
users = data.get('users', [])
print(f'Total users found: {len(users)}\n')

confirmed_count = 0
unconfirmed = []

for u in users:
    email = u.get('email', '')
    user_id = u.get('id', '')
    is_confirmed = bool(u.get('email_confirmed_at'))
    status = 'CONFIRMED' if is_confirmed else 'UNCONFIRMED'
    print(f'  [{status}] {email} (id: {user_id[:8]}...)')
    if not is_confirmed:
        unconfirmed.append((user_id, email))

print(f'\n{len(unconfirmed)} unconfirmed user(s) found.')

# Force-confirm all unconfirmed users
for user_id, email in unconfirmed:
    patch_resp = httpx.put(
        f'{url}/auth/v1/admin/users/{user_id}',
        json={'email_confirm': True},
        headers=headers
    )
    if patch_resp.status_code == 200:
        print(f'  CONFIRMED: {email}')
        confirmed_count += 1
    else:
        print(f'  FAILED to confirm {email}: {patch_resp.status_code} - {patch_resp.text}')

print(f'\nDone. {confirmed_count} user(s) newly confirmed.')
print('You can now log in with any of these accounts.')
