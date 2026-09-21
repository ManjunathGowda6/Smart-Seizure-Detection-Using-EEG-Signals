import React, { useState, useEffect } from 'react';
import { supabase } from '../supabaseClient';
import WaveformPlot from '../components/WaveformPlot';
import { Activity, AlertTriangle, CheckCircle, Plus, Users, Save, X } from 'lucide-react';

export default function QuickScan() {
  const [selectedModel, setSelectedModel] = useState('cnn_bilstm');
  const [uploadFile, setUploadFile] = useState(null);
  const [scanning, setScanning] = useState(false);
  const [scanResult, setScanResult] = useState(null);

  // Patient assignment states
  const [patients, setPatients] = useState([]);
  const [selectedPatientId, setSelectedPatientId] = useState('');
  const [showSaveModal, setShowSaveModal] = useState(false);
  const [savingRecord, setSavingRecord] = useState(false);

  useEffect(() => {
    // Fetch user's patients list for dropdown
    const fetchPatients = async () => {
      try {
        const session = (await supabase.auth.getSession()).data.session;
        if (!session) return;
        const token = session.access_token;
        
        const res = await fetch('http://localhost:8000/patients', {
          headers: { 'Authorization': `Bearer ${token}` }
        });
        if (res.ok) {
          const list = await res.json();
          setPatients(list);
          if (list.length > 0) {
            setSelectedPatientId(list[0].id);
          }
        }
      } catch (err) {
        console.error("Error fetching patients list:", err);
      }
    };
    fetchPatients();
  }, []);

  const handleScanSubmit = async (e) => {
    e.preventDefault();
    if (!uploadFile) return;

    setScanning(true);
    setScanResult(null);

    const formData = new FormData();
    formData.append('file', uploadFile);
    formData.append('model_used', selectedModel);

    // ── Debug checkpoint 1: confirm what is being sent ──────────────────────
    console.log('[QuickScan] STEP 1 — sending to backend', {
      url:   'http://localhost:8000/uploads/quick-scan',
      model: selectedModel,
      file:  uploadFile.name,
      size:  uploadFile.size,
    });

    try {
      const res = await fetch('http://localhost:8000/uploads/quick-scan', {
        method: 'POST',
        body: formData
      });

      // ── Debug checkpoint 2: raw HTTP response ────────────────────────────
      console.log('[QuickScan] STEP 2 — HTTP response', res.status, res.ok);

      if (!res.ok) {
        const errData = await res.json();
        throw new Error(errData.detail || 'Quick scan failed.');
      }

      const data = await res.json();

      // ── Debug checkpoint 3: raw JSON from backend ────────────────────────
      // If confidence_score is 100 here the bug is in the backend, not frontend.
      // If confidence_score is 99.x here and displays as 100 the bug is .toFixed.
      console.log('[QuickScan] STEP 3 — backend JSON response:', {
        prediction:       data.prediction,
        confidence_score: data.confidence_score,   // should be 90-99.9, never 100
        seizure_segment:  data.seizure_segment,
      });

      setScanResult(data);
    } catch (err) {
      alert(err.message);
    } finally {
      setScanning(false);
    }
  };

  const handleSaveToPatient = async () => {
    if (!selectedPatientId || !scanResult) return;
    setSavingRecord(true);

    const formData = new FormData();
    formData.append('temp_id', scanResult.temp_id);
    formData.append('patient_id', selectedPatientId);
    formData.append('prediction', scanResult.prediction);
    formData.append('confidence_score', scanResult.confidence_score);
    formData.append('model_used', selectedModel);

    try {
      const session = (await supabase.auth.getSession()).data.session;
      if (!session) return;
      const token = session.access_token;

      const res = await fetch('http://localhost:8000/uploads/save-quick-scan', {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${token}`
        },
        body: formData
      });

      if (res.ok) {
        alert("Scan saved and linked to patient successfully!");
        setShowSaveModal(false);
        // Clear scan
        setScanResult(null);
        setUploadFile(null);
      } else {
        const errData = await res.json();
        alert(errData.detail || "Failed to save record.");
      }
    } catch (err) {
      alert(err.message);
    } finally {
      setSavingRecord(false);
    }
  };

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '2fr 1fr', gap: '24px', alignItems: 'start' }}>
      
      {/* Left Column: Waveform Plot */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
        <section className="glass-panel" style={{ padding: '24px' }}>
          <h3 style={{ fontSize: '1.1rem', fontWeight: 'bold', marginBottom: '16px', display: 'flex', alignItems: 'center', gap: '8px' }}>
            <Activity style={{ width: '18px', height: '18px', color: '#3B82F6' }} />
            EEG Waveform Display
          </h3>

          <div style={{ height: '420px', position: 'relative' }}>
            {scanResult ? (
              <WaveformPlot 
                visualizationData={scanResult.visualization_data}
                seizureSegment={scanResult.seizure_segment}
              />
            ) : (
              <div style={{ height: '100%', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', color: 'var(--text-muted)' }}>
                <Activity style={{ width: '60px', height: '60px', color: 'rgba(255,255,255,0.04)', marginBottom: '16px' }} />
                <h4>Diagnostic Waveform Area</h4>
                <p style={{ fontSize: '0.8rem', marginTop: '4px' }}>Upload a file in the sidebar to visualize the live EEG telemetry.</p>
              </div>
            )}
          </div>
        </section>
      </div>

      {/* Right Column: Scan Controls and Results */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
        
        {/* Upload Panel */}
        <section className="glass-panel" style={{ padding: '24px' }}>
          <h3 style={{ fontSize: '1.1rem', fontWeight: 'bold', marginBottom: '20px' }}>Instant Scanner</h3>
          
          <form onSubmit={handleScanSubmit} style={{ display: 'flex', flexDirection: 'column', gap: '18px' }}>
            <div>
              <label style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', display: 'block', marginBottom: '6px' }}>
                Classification Model
              </label>
              <select
                value={selectedModel}
                onChange={(e) => setSelectedModel(e.target.value)}
                className="input-field"
                style={{ background: '#0F172A' }}
              >
                <option value="cnn_bilstm">CNN-BiLSTM (Recommended) ★</option>
                <option value="svm">Support Vector Machine (SVM)</option>
                <option value="lr">Logistic Regression (LR)</option>
                <option value="cnn_lti">CNN + Biological LTI Layer</option>
              </select>
            </div>

            <div>
              <label style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', display: 'block', marginBottom: '6px' }}>
                EEG Data File (.edf, .csv)
              </label>
              <div 
                style={{
                  border: '2px dashed var(--border-color)',
                  borderRadius: 'var(--radius-sm)',
                  padding: '30px 16px',
                  textAlign: 'center',
                  cursor: 'pointer',
                  background: 'rgba(15, 23, 42, 0.4)'
                }}
                onDragOver={(e) => e.preventDefault()}
                onDrop={(e) => {
                  e.preventDefault();
                  if (e.dataTransfer.files[0]) setUploadFile(e.dataTransfer.files[0]);
                }}
              >
                <input
                  type="file"
                  id="quick-file-selector"
                  accept=".edf,.csv"
                  onChange={(e) => setUploadFile(e.target.files[0])}
                  style={{ display: 'none' }}
                />
                <label htmlFor="quick-file-selector" style={{ cursor: 'pointer', display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '8px' }}>
                  <Activity style={{ width: '32px', height: '32px', color: '#60A5FA' }} />
                  <span style={{ fontSize: '0.85rem', color: 'white', fontWeight: '600' }}>
                    {uploadFile ? uploadFile.name : 'Select EEG Signal File'}
                  </span>
                  <span style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>
                    EDF or multi-channel CSV telemetry
                  </span>
                </label>
              </div>
            </div>

            <button
              type="submit"
              className="btn btn-primary"
              disabled={!uploadFile || scanning}
              style={{ width: '100%', padding: '12px' }}
            >
              {scanning ? 'Running Signal Preprocessing...' : 'Scan Signal Now'}
            </button>
          </form>
        </section>

        {/* Scan Results Panel */}
        {scanResult && (
          <section 
            className="glass-panel fade-in-up" 
            style={{ 
              padding: '24px', 
              border: scanResult.prediction === 'seizure' ? '1px solid rgba(239, 68, 68, 0.2)' : '1px solid rgba(16, 185, 129, 0.2)',
              animation: scanResult.prediction === 'seizure' ? 'pulse-red 2s infinite' : 'pulse-green 2s infinite'
            }}
          >
            <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '16px' }}>
              {scanResult.prediction === 'seizure' ? (
                <AlertTriangle style={{ color: '#EF4444', width: '24px', height: '24px' }} />
              ) : (
                <CheckCircle style={{ color: '#10B981', width: '24px', height: '24px' }} />
              )}
              <h3 style={{ fontSize: '1.1rem', fontWeight: 'bold' }}>
                {scanResult.prediction === 'seizure' ? 'Seizure Activity Detected' : 'EEG Scan Clear'}
              </h3>
            </div>

            <div style={{ display: 'flex', flexDirection: 'column', gap: '10px', fontSize: '0.85rem', color: 'var(--text-secondary)', marginBottom: '20px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span>Inference Confidence:</span>
                {/* ── Debug checkpoint 4: display value before format ── */}
                {/* Open DevTools → Console to see: [QuickScan] STEP 4 raw conf */}
                { (() => { console.log('[QuickScan] STEP 4 — rendering confidence_score:', scanResult.confidence_score); return null; })() }
                <span style={{ color: 'white', fontWeight: 'bold' }}>
                  {/* Fix: cap at 99.9 so toFixed(1) can never round up to 100.0.
                      Root cause: (99.99).toFixed(1) === "100.0" in JavaScript.    */}
                  {Math.min(scanResult.confidence_score, 99.9).toFixed(1)}%
                </span>
              </div>
              {scanResult.seizure_segment.start !== null && (
                <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                  <span>Seizure Interval:</span>
                  <span style={{ color: '#FCA5A5', fontWeight: 'bold' }}>
                    {scanResult.seizure_segment.start.toFixed(1)}s - {scanResult.seizure_segment.end.toFixed(1)}s
                  </span>
                </div>
              )}
            </div>

            {/* Prompt to save */}
            <div style={{ borderTop: '1px solid var(--border-color)', paddingTop: '16px', display: 'flex', flexDirection: 'column', gap: '12px' }}>
              <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
                Do you want to save this diagnostic result to an active patient record?
              </p>
              <button 
                onClick={() => setShowSaveModal(true)} 
                className="btn btn-primary"
                style={{ width: '100%', padding: '10px', fontSize: '0.85rem' }}
              >
                <Save style={{ width: '14px', height: '14px' }} />
                <span>Save to Patient Record</span>
              </button>
            </div>
          </section>
        )}

      </div>

      {/* Save to Patient Modal Dialog */}
      {showSaveModal && (
        <div className="modal-overlay">
          <div className="modal-content glass-panel" style={{ maxWidth: '400px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px' }}>
              <h3 style={{ fontSize: '1.15rem', fontWeight: 'bold' }}>Save Scan Result</h3>
              <button onClick={() => setShowSaveModal(false)} style={{ background: 'none', border: 'none', color: 'var(--text-muted)', cursor: 'pointer' }}>
                <X style={{ width: '18px', height: '18px' }} />
              </button>
            </div>

            {patients.length === 0 ? (
              <div style={{ padding: '20px 0', textColor: 'var(--text-secondary)', fontSize: '0.9rem' }}>
                No patient records found in your registry. Please add a patient first.
              </div>
            ) : (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
                <div>
                  <label style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', display: 'block', marginBottom: '6px' }}>
                    Select Patient Record
                  </label>
                  <select
                    value={selectedPatientId}
                    onChange={(e) => setSelectedPatientId(e.target.value)}
                    className="input-field"
                    style={{ background: '#0F172A' }}
                  >
                    {patients.map(p => (
                      <option key={p.id} value={p.id}>
                        {p.full_name} (Age: {p.age || 'N/A'})
                      </option>
                    ))}
                  </select>
                </div>

                <button
                  onClick={handleSaveToPatient}
                  className="btn btn-primary"
                  disabled={savingRecord}
                  style={{ width: '100%', padding: '12px' }}
                >
                  {savingRecord ? 'Uploading telemetry file...' : 'Link & Save Record'}
                </button>
              </div>
            )}
          </div>
        </div>
      )}

    </div>
  );
}
