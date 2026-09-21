# NeuroWatch: Smart Seizure Detection and Patient Monitoring Platform

NeuroWatch is a full-stack clinical dashboard designed for doctors and caregivers to monitor pediatric epilepsy patients. It utilizes machine learning (SVM, Logistic Regression) and deep learning (PyTorch CNN + Biological LTI Layer) trained on the CHB-MIT Scalp EEG database to classify seizure activity from EDF and CSV telemetry.

---

## Technical Stack
- **Frontend**: React (Vite, CSS Design System, Plotly.js Bipolar Waveform Visualization)
- **Backend**: FastAPI (Python), MNE (EEG Processing), PyEMD (Empirical Mode Decomposition), PyTorch & Scikit-Learn (Inference)
- **Database/Auth**: Supabase (PostgreSQL with Row Level Security, Auth, and Storage bucket)
- **PDF Compilation**: ReportLab with Matplotlib timeline trend generation

---

## Supabase Database Setup

To configure your Supabase backend, run the following SQL script in your Supabase SQL Editor. This script creates the tables, enables RLS, establishes user profiles automatically via a trigger, and configures appropriate access policies:

```sql
-- 1. Profiles Table (extends auth.users)
CREATE TABLE public.profiles (
  id UUID PRIMARY KEY REFERENCES auth.users(id) ON DELETE CASCADE,
  full_name TEXT,
  role TEXT CHECK (role IN ('doctor', 'caregiver')),
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
ALTER TABLE public.profiles ENABLE ROW LEVEL SECURITY;

-- 2. Patients Table
CREATE TABLE public.patients (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID NOT NULL REFERENCES public.profiles(id) ON DELETE CASCADE,
  name TEXT NOT NULL,
  age INTEGER,
  gender TEXT CHECK (gender IN ('Male', 'Female', 'Other')),
  diagnosis TEXT,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
ALTER TABLE public.patients ENABLE ROW LEVEL SECURITY;

-- 3. EEG Uploads Table
CREATE TABLE public.eeg_uploads (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  patient_id UUID REFERENCES public.patients(id) ON DELETE CASCADE,
  file_url TEXT NOT NULL,
  upload_time TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  prediction TEXT CHECK (prediction IN ('seizure', 'no_seizure')),
  confidence_score FLOAT NOT NULL,
  model_used TEXT CHECK (model_used IN ('svm', 'lr', 'cnn_lti')),
  is_quick_scan BOOLEAN DEFAULT FALSE
);
ALTER TABLE public.eeg_uploads ENABLE ROW LEVEL SECURITY;

-- 4. Alerts Table
CREATE TABLE public.alerts (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  patient_id UUID NOT NULL REFERENCES public.patients(id) ON DELETE CASCADE,
  triggered_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  message TEXT NOT NULL,
  is_read BOOLEAN DEFAULT FALSE
);
ALTER TABLE public.alerts ENABLE ROW LEVEL SECURITY;

-- 5. Notes Table
CREATE TABLE public.notes (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  patient_id UUID NOT NULL REFERENCES public.patients(id) ON DELETE CASCADE,
  created_by UUID NOT NULL REFERENCES public.profiles(id) ON DELETE CASCADE,
  note_text TEXT NOT NULL,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
ALTER TABLE public.notes ENABLE ROW LEVEL SECURITY;

-- 6. Trigger for Automatic Profile Creation on Signup
CREATE OR REPLACE FUNCTION public.handle_new_user()
RETURNS TRIGGER AS $$
BEGIN
  INSERT INTO public.profiles (id, full_name, role)
  VALUES (
    new.id,
    COALESCE(new.raw_user_meta_data->>'full_name', ''),
    COALESCE(new.raw_user_meta_data->>'role', 'doctor')
  );
  RETURN NEW;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

CREATE OR REPLACE TRIGGER on_auth_user_created
  AFTER INSERT ON auth.users
  FOR EACH ROW EXECUTE FUNCTION public.handle_new_user();

-- 7. Row Level Security Policies
CREATE POLICY "Allow public read of profiles" ON profiles FOR SELECT USING (true);
CREATE POLICY "Allow users to update own profile" ON profiles FOR UPDATE USING (auth.uid() = id) WITH CHECK (auth.uid() = id);
CREATE POLICY "Allow users to insert own profile" ON profiles FOR INSERT WITH CHECK (auth.uid() = id);

CREATE POLICY "Allow users to read own patients" ON patients FOR SELECT USING (user_id = auth.uid());
CREATE POLICY "Allow users to insert own patients" ON patients FOR INSERT WITH CHECK (user_id = auth.uid());
CREATE POLICY "Allow users to update own patients" ON patients FOR UPDATE USING (user_id = auth.uid()) WITH CHECK (user_id = auth.uid());
CREATE POLICY "Allow users to delete own patients" ON patients FOR DELETE USING (user_id = auth.uid());

CREATE POLICY "Allow users to read own eeg_uploads" ON eeg_uploads FOR SELECT USING (
  patient_id IS NULL OR patient_id IN (SELECT id FROM patients WHERE user_id = auth.uid())
);
CREATE POLICY "Allow users to insert own eeg_uploads" ON eeg_uploads FOR INSERT WITH CHECK (
  patient_id IS NULL OR patient_id IN (SELECT id FROM patients WHERE user_id = auth.uid())
);

CREATE POLICY "Allow users to read own alerts" ON alerts FOR SELECT USING (
  patient_id IN (SELECT id FROM patients WHERE user_id = auth.uid())
);
CREATE POLICY "Allow users to update own alerts" ON alerts FOR UPDATE USING (
  patient_id IN (SELECT id FROM patients WHERE user_id = auth.uid())
) WITH CHECK (
  patient_id IN (SELECT id FROM patients WHERE user_id = auth.uid())
);

CREATE POLICY "Allow users to read own notes" ON notes FOR SELECT USING (
  patient_id IN (SELECT id FROM patients WHERE user_id = auth.uid())
);
CREATE POLICY "Allow users to insert own notes" ON notes FOR INSERT WITH CHECK (
  patient_id IN (SELECT id FROM patients WHERE user_id = auth.uid()) AND created_by = auth.uid()
);
CREATE POLICY "Allow users to update own notes" ON notes FOR UPDATE USING (
  patient_id IN (SELECT id FROM patients WHERE user_id = auth.uid()) AND created_by = auth.uid()
) WITH CHECK (
  patient_id IN (SELECT id FROM patients WHERE user_id = auth.uid()) AND created_by = auth.uid()
);
CREATE POLICY "Allow users to delete own notes" ON notes FOR DELETE USING (
  patient_id IN (SELECT id FROM patients WHERE user_id = auth.uid()) AND created_by = auth.uid()
);
```

### Storage Bucket Setup
1. Open the Supabase console, go to **Storage**, and create a new bucket named `eeg-uploads`.
2. Set it to **Public** (or configure bucket policies allowing authenticated read/write access).

---

## Configuration

Fill the `.env` configuration files manually with your project credentials:

### Backend Configuration (`backend/.env`)
```env
SUPABASE_URL=your_supabase_project_url
SUPABASE_KEY=your_supabase_service_role_key_or_anon_key
SUPABASE_JWT_SECRET=your_supabase_jwt_secret
```

### Frontend Configuration (`frontend/.env`)
```env
VITE_SUPABASE_URL=your_supabase_project_url
VITE_SUPABASE_ANON_KEY=your_supabase_anon_key
```

---

## How to Run

### 1. Model Training (Already Completed)
The models have been trained and compiled on the local CHB-MIT dataset segment. The weights, scaling structures, and metrics are stored in `backend/models/`. If you wish to retrain:
```bash
python backend/ml/train.py
```

### 2. Start the Backend API
From the root directory, navigate to `backend` and run the FastAPI server:
```bash
cd backend
python main.py
```
The documentation is available at `http://localhost:8000/docs`.

### 3. Start the Frontend Application
In another terminal, navigate to `frontend`, install packages, and boot up the Vite development server:
```bash
cd frontend
npm install
npm run dev
```
Open `http://localhost:5173` in your browser.
