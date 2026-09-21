import React, { useEffect, useState, useRef, useCallback } from 'react';
import { Brain, Activity, Zap, TrendingUp, ChevronRight, RefreshCw,
         AlertTriangle, CheckCircle, Loader, BarChart2, Upload, FlaskConical } from 'lucide-react';

const API = 'http://localhost:8000';

// ── Tiny SVG EEG waveform visualiser ─────────────────────────────────────────
function EEGWave({ values = [], isSeizure = false, isAnimating = false }) {
  if (!values.length) return null;
  const W = 720, H = 120;
  const min = Math.min(...values), max = Math.max(...values);
  const range = Math.max(max - min, 1);
  const pts = values.map((v, i) => {
    const x = (i / (values.length - 1)) * W;
    const y = H - ((v - min) / range) * (H - 10) - 5;
    return `${x},${y}`;
  }).join(' ');

  const color = isSeizure ? '#F87171' : '#34D399';
  const glow  = isSeizure ? 'rgba(248,113,113,0.25)' : 'rgba(52,211,153,0.2)';

  return (
    <svg viewBox={`0 0 ${W} ${H}`} width="100%" height="100%"
         style={{ display: 'block' }}>
      <defs>
        <filter id="eeg-glow">
          <feGaussianBlur stdDeviation="2.5" result="blur" />
          <feMerge><feMergeNode in="blur" /><feMergeNode in="SourceGraphic" /></feMerge>
        </filter>
        <linearGradient id="wave-grad" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor={color} stopOpacity="0.15" />
          <stop offset="100%" stopColor={color} stopOpacity="0" />
        </linearGradient>
      </defs>
      {/* Grid lines */}
      {[0.25, 0.5, 0.75].map(f => (
        <line key={f} x1="0" y1={H * f} x2={W} y2={H * f}
              stroke="rgba(255,255,255,0.04)" strokeWidth="1" />
      ))}
      {/* Area fill */}
      <polyline points={pts + ` ${W},${H} 0,${H}`}
        fill={`url(#wave-grad)`} stroke="none" />
      {/* Signal line */}
      <polyline points={pts} fill="none" stroke={color} strokeWidth="1.8"
        filter="url(#eeg-glow)"
        style={{ transition: 'stroke 0.5s ease' }} />
      {/* Baseline */}
      <line x1="0" y1={H / 2} x2={W} y2={H / 2}
            stroke="rgba(255,255,255,0.06)" strokeWidth="1" strokeDasharray="4,6" />
    </svg>
  );
}

// ── Animated probability bar ──────────────────────────────────────────────────
function ProbBar({ value, label, color }) {
  const [width, setWidth] = useState(0);
  useEffect(() => {
    const t = setTimeout(() => setWidth(value * 100), 60);
    return () => clearTimeout(t);
  }, [value]);
  return (
    <div style={{ marginBottom: '10px' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between',
                    fontSize: '0.8rem', color: 'var(--text-secondary)',
                    marginBottom: '5px' }}>
        <span>{label}</span>
        <span style={{ color, fontWeight: 600 }}>{(value * 100).toFixed(1)}%</span>
      </div>
      <div style={{ height: '7px', background: 'rgba(255,255,255,0.06)',
                    borderRadius: '4px', overflow: 'hidden' }}>
        <div style={{ height: '100%', width: `${width}%`,
                      background: color, borderRadius: '4px',
                      transition: 'width 0.8s cubic-bezier(0.16,1,0.3,1)',
                      boxShadow: `0 0 8px ${color}55` }} />
      </div>
    </div>
  );
}

// ── Model status card ─────────────────────────────────────────────────────────
function ModelCard({ model, isNew = false }) {
  const available = model.available;
  const color = model.type === 'Deep Learning' ? '#818CF8' : '#60A5FA';

  return (
    <div className="glass-panel" style={{
      padding: '18px 20px',
      border: isNew ? '1px solid rgba(129,140,248,0.35)' : '1px solid var(--border-color)',
      background: isNew ? 'rgba(129,140,248,0.05)' : 'var(--bg-card)',
      position: 'relative',
      overflow: 'hidden',
    }}>
      {isNew && (
        <div style={{
          position: 'absolute', top: '10px', right: '10px',
          fontSize: '0.6rem', fontWeight: 700, color: '#818CF8',
          background: 'rgba(129,140,248,0.15)',
          border: '1px solid rgba(129,140,248,0.3)',
          padding: '2px 7px', borderRadius: '20px', textTransform: 'uppercase',
          letterSpacing: '0.06em',
        }}>NEW</div>
      )}
      <div style={{ display: 'flex', alignItems: 'center', gap: '10px',
                    marginBottom: '12px' }}>
        <div style={{ width: '8px', height: '8px', borderRadius: '50%',
                      background: available ? '#34D399' : '#6B7280',
                      boxShadow: available ? '0 0 6px #34D39988' : 'none' }} />
        <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)',
                       textTransform: 'uppercase', letterSpacing: '0.06em' }}>
          {model.type}
        </span>
      </div>
      <div style={{ fontWeight: 700, fontSize: '0.95rem', marginBottom: '4px' }}>
        {model.name}
      </div>
      <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)',
                    marginBottom: '12px' }}>{model.dataset}</div>
      {available && model.recall !== null && model.recall !== undefined && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '6px' }}>
          {[
            { l: 'Recall',   v: model.recall },
            { l: 'F1',       v: model.f1 },
            { l: 'Accuracy', v: model.accuracy },
            { l: 'ROC-AUC',  v: model.roc_auc },
          ].map(({ l, v }) => (
            v !== null && v !== undefined ? (
              <div key={l} style={{ background: 'rgba(255,255,255,0.03)',
                                    borderRadius: '6px', padding: '6px 8px' }}>
                <div style={{ fontSize: '0.65rem', color: 'var(--text-muted)',
                               marginBottom: '2px' }}>{l}</div>
                <div style={{ fontSize: '0.9rem', fontWeight: 700,
                               color: v < 0.3 ? '#F87171' : v < 0.6 ? '#FBBF24' : '#34D399' }}>
                  {(v * 100).toFixed(1)}%
                </div>
              </div>
            ) : null
          ))}
        </div>
      )}
      {!available && (
        <div style={{ fontSize: '0.78rem', color: 'var(--text-muted)',
                      fontStyle: 'italic' }}>Not yet trained</div>
      )}
    </div>
  );
}

// ── Main page ─────────────────────────────────────────────────────────────────
export default function ModelTester() {
  const [models, setModels]       = useState([]);
  const [eegVals, setEegVals]     = useState([]);
  const [sampleKind, setSampleKind] = useState(null);   // 'normal' | 'seizure'
  const [result, setResult]       = useState(null);
  const [loading, setLoading]     = useState(false);
  const [statusLoad, setStatusLoad] = useState(true);
  const [error, setError]         = useState('');
  const [customText, setCustomText] = useState('');
  const [activeTab, setActiveTab] = useState('generate');   // 'generate' | 'paste' | 'upload'
  const fileRef = useRef(null);

  // Load model status
  useEffect(() => {
    fetch(`${API}/model/status`)
      .then(r => r.json())
      .then(d => { setModels(d.models || []); setStatusLoad(false); })
      .catch(() => setStatusLoad(false));
  }, []);

  // Generate synthetic sample
  const generateSample = useCallback(async (kind) => {
    setLoading(true); setError(''); setResult(null);
    try {
      const r = await fetch(`${API}/model/sample/${kind}`);
      const d = await r.json();
      setEegVals(d.eeg_values);
      setSampleKind(kind);
    } catch {
      setError('Failed to generate sample.');
    } finally {
      setLoading(false);
    }
  }, []);

  // Run prediction
  const runPredict = useCallback(async (vals) => {
    if (!vals || !vals.length) { setError('No EEG values to predict.'); return; }
    setLoading(true); setError(''); setResult(null);
    try {
      const r = await fetch(`${API}/model/bilstm-predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ eeg_values: vals }),
      });
      if (!r.ok) {
        const e = await r.json();
        throw new Error(e.detail || 'Prediction failed');
      }
      const d = await r.json();
      setResult(d);
      setEegVals(d.eeg_values);   // show normalised signal
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }, []);

  // Paste EEG values
  const handlePaste = () => {
    try {
      const vals = customText.split(/[\s,;]+/).map(Number).filter(n => !isNaN(n));
      if (vals.length < 10) { setError('Need at least 10 values.'); return; }
      setEegVals(vals);
      setSampleKind(null);
      runPredict(vals);
    } catch { setError('Could not parse EEG values.'); }
  };

  // Upload CSV
  const handleFile = async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const text = await file.text();
    const rows = text.split('\n').filter(r => r.trim());
    const vals = rows[rows.length - 1]   // use last data row
      .split(',').map(Number).filter(n => !isNaN(n));
    if (vals.length < 10) { setError('Could not parse CSV row.'); return; }
    setEegVals(vals);
    setSampleKind(null);
    runPredict(vals);
  };

  const isSeizureResult = result?.prediction === 'seizure';

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '28px' }}>

      {/* ── Page header ── */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
        <div style={{ padding: '14px', background: 'rgba(129,140,248,0.15)',
                      borderRadius: '14px', color: '#818CF8' }}>
          <Brain style={{ width: '28px', height: '28px' }} />
        </div>
        <div>
          <h2 style={{ fontSize: '1.5rem', fontWeight: 800, fontFamily: 'var(--font-title)' }}>
            CNN-BiLSTM v1 — Model Tester
          </h2>
          <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', marginTop: '4px' }}>
            Test the new Phase 1 deep-learning model trained on Bonn + CHB-MIT (SMOTE balanced)
          </p>
        </div>
      </div>

      {/* ── Model status cards ── */}
      <section>
        <h3 style={{ fontSize: '0.8rem', textTransform: 'uppercase',
                     letterSpacing: '0.1em', color: 'var(--text-muted)',
                     marginBottom: '14px' }}>All Trained Models</h3>
        {statusLoad ? (
          <div style={{ color: 'var(--text-muted)', fontSize: '0.85rem' }}>
            Loading model status...
          </div>
        ) : (
          <div style={{ display: 'grid',
                        gridTemplateColumns: 'repeat(auto-fill, minmax(230px,1fr))',
                        gap: '16px' }}>
            {models.map(m => (
              <ModelCard key={m.id} model={m} isNew={m.id === 'cnn_bilstm_v1'} />
            ))}
          </div>
        )}
      </section>

      {/* ── Demo area ── */}
      <section style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px' }}>

        {/* LEFT — controls */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
          <div className="glass-panel" style={{ padding: '24px' }}>
            <h3 style={{ fontSize: '1rem', fontWeight: 700, marginBottom: '6px',
                         display: 'flex', alignItems: 'center', gap: '8px' }}>
              <FlaskConical style={{ width: '18px', height: '18px', color: '#818CF8' }} />
              Input Method
            </h3>
            <p style={{ fontSize: '0.8rem', color: 'var(--text-secondary)',
                        marginBottom: '20px' }}>
              Choose how to provide EEG data for the model
            </p>

            {/* Tabs */}
            <div style={{ display: 'flex', gap: '4px', background: 'rgba(0,0,0,0.3)',
                           padding: '4px', borderRadius: '10px', marginBottom: '20px' }}>
              {[
                { id: 'generate', label: 'Generate', icon: Zap },
                { id: 'paste',    label: 'Paste',    icon: Activity },
                { id: 'upload',   label: 'Upload',   icon: Upload },
              ].map(({ id, label, icon: Icon }) => (
                <button key={id} onClick={() => setActiveTab(id)}
                  style={{
                    flex: 1, padding: '8px', borderRadius: '7px', border: 'none',
                    cursor: 'pointer', fontSize: '0.8rem', fontWeight: 600,
                    transition: 'all 0.2s',
                    background: activeTab === id ? 'rgba(129,140,248,0.2)' : 'transparent',
                    color: activeTab === id ? '#A5B4FC' : 'var(--text-muted)',
                    display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '5px',
                  }}>
                  <Icon style={{ width: '13px', height: '13px' }} />
                  {label}
                </button>
              ))}
            </div>

            {/* Tab: Generate */}
            {activeTab === 'generate' && (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
                <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
                  Generate synthetic EEG for quick testing
                </p>
                <button id="gen-normal-btn"
                  onClick={() => { generateSample('normal'); }}
                  disabled={loading}
                  className="btn btn-secondary"
                  style={{ justifyContent: 'flex-start', gap: '10px' }}>
                  <div style={{ width: '10px', height: '10px', borderRadius: '50%',
                                background: '#34D399' }} />
                  Normal / Interictal EEG
                </button>
                <button id="gen-seizure-btn"
                  onClick={() => { generateSample('seizure'); }}
                  disabled={loading}
                  className="btn btn-secondary"
                  style={{ justifyContent: 'flex-start', gap: '10px' }}>
                  <div style={{ width: '10px', height: '10px', borderRadius: '50%',
                                background: '#F87171' }} />
                  Ictal / Seizure EEG
                </button>
                {eegVals.length > 0 && (
                  <button id="predict-btn"
                    onClick={() => runPredict(eegVals)}
                    disabled={loading}
                    className="btn btn-primary" style={{ marginTop: '6px' }}>
                    {loading
                      ? <><Loader style={{ width: '15px', height: '15px',
                                          animation: 'spin 1s linear infinite' }} /> Analysing...</>
                      : <><Brain style={{ width: '15px', height: '15px' }} /> Run CNN-BiLSTM v1</>}
                  </button>
                )}
              </div>
            )}

            {/* Tab: Paste */}
            {activeTab === 'paste' && (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
                <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
                  Paste up to 178 comma-separated EEG values
                </p>
                <textarea
                  id="eeg-paste-input"
                  placeholder="e.g.  120, -45, 67, 88, ..."
                  rows={5}
                  value={customText}
                  onChange={e => setCustomText(e.target.value)}
                  className="input-field"
                  style={{ resize: 'vertical', fontSize: '0.78rem',
                           fontFamily: 'monospace' }}
                />
                <button id="paste-predict-btn"
                  onClick={handlePaste} disabled={loading || !customText.trim()}
                  className="btn btn-primary">
                  {loading ? 'Analysing...' : 'Run CNN-BiLSTM v1'}
                </button>
              </div>
            )}

            {/* Tab: Upload */}
            {activeTab === 'upload' && (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
                <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
                  Upload a CSV row with 178 EEG values (Bonn format)
                </p>
                <div
                  id="file-drop-zone"
                  onClick={() => fileRef.current?.click()}
                  style={{
                    border: '2px dashed rgba(129,140,248,0.35)',
                    borderRadius: '10px', padding: '32px',
                    textAlign: 'center', cursor: 'pointer',
                    transition: 'all 0.2s',
                    color: 'var(--text-muted)',
                  }}
                  onMouseEnter={e => e.currentTarget.style.borderColor = 'rgba(129,140,248,0.6)'}
                  onMouseLeave={e => e.currentTarget.style.borderColor = 'rgba(129,140,248,0.35)'}
                >
                  <Upload style={{ width: '28px', height: '28px', margin: '0 auto 8px',
                                   display: 'block', color: '#818CF8' }} />
                  <div style={{ fontSize: '0.85rem', fontWeight: 600 }}>
                    Click to upload CSV
                  </div>
                  <div style={{ fontSize: '0.75rem', marginTop: '4px' }}>
                    One row, 178 columns (Bonn EEG format)
                  </div>
                </div>
                <input ref={fileRef} type="file" accept=".csv,.txt"
                       onChange={handleFile} style={{ display: 'none' }} />
              </div>
            )}
          </div>

          {/* EEG Waveform */}
          {eegVals.length > 0 && (
            <div className="glass-panel" style={{ padding: '18px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between',
                             alignItems: 'center', marginBottom: '12px' }}>
                <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)',
                               display: 'flex', alignItems: 'center', gap: '6px' }}>
                  <Activity style={{ width: '13px', height: '13px' }} />
                  EEG Waveform  ({eegVals.length} samples)
                </span>
                {sampleKind && (
                  <span style={{
                    fontSize: '0.7rem', padding: '3px 8px', borderRadius: '12px',
                    background: sampleKind === 'seizure'
                      ? 'rgba(248,113,113,0.12)' : 'rgba(52,211,153,0.1)',
                    color: sampleKind === 'seizure' ? '#F87171' : '#34D399',
                    border: `1px solid ${sampleKind === 'seizure'
                      ? 'rgba(248,113,113,0.2)' : 'rgba(52,211,153,0.2)'}`,
                    fontWeight: 600,
                  }}>
                    {sampleKind === 'seizure' ? '⚠ Synthetic Ictal' : '✓ Synthetic Normal'}
                  </span>
                )}
              </div>
              <div style={{ height: '100px' }}>
                <EEGWave values={eegVals}
                         isSeizure={result?.prediction === 'seizure' || sampleKind === 'seizure'} />
              </div>
            </div>
          )}
        </div>

        {/* RIGHT — results */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
          {error && (
            <div style={{
              padding: '14px 16px', borderRadius: '10px',
              background: 'rgba(239,68,68,0.1)', color: '#FCA5A5',
              border: '1px solid rgba(239,68,68,0.2)',
              fontSize: '0.85rem', display: 'flex', gap: '10px', alignItems: 'center',
            }}>
              <AlertTriangle style={{ width: '16px', height: '16px', flexShrink: 0 }} />
              {error}
            </div>
          )}

          {!result && !loading && (
            <div className="glass-panel" style={{
              padding: '48px 32px', textAlign: 'center',
              display: 'flex', flexDirection: 'column', alignItems: 'center',
              gap: '14px', flex: 1,
            }}>
              <div style={{ padding: '20px', background: 'rgba(129,140,248,0.08)',
                             borderRadius: '50%' }}>
                <Brain style={{ width: '40px', height: '40px', color: '#818CF8',
                                opacity: 0.5 }} />
              </div>
              <div style={{ color: 'var(--text-muted)', fontSize: '0.9rem' }}>
                Generate a sample or paste EEG values, then run the model
              </div>
            </div>
          )}

          {loading && (
            <div className="glass-panel" style={{
              padding: '48px 32px', textAlign: 'center',
              display: 'flex', flexDirection: 'column', alignItems: 'center',
              gap: '16px', flex: 1,
            }}>
              <div style={{ padding: '20px', background: 'rgba(129,140,248,0.12)',
                             borderRadius: '50%', animation: 'pulse-green 1.5s ease infinite' }}>
                <Brain style={{ width: '40px', height: '40px', color: '#818CF8' }} />
              </div>
              <div style={{ color: 'var(--text-secondary)', fontSize: '0.9rem' }}>
                Running CNN-BiLSTM v1 inference...
              </div>
              <div style={{ display: 'flex', gap: '6px' }}>
                {[0, 0.15, 0.3].map(d => (
                  <div key={d} style={{
                    width: '8px', height: '8px', borderRadius: '50%',
                    background: '#818CF8',
                    animation: `pulse-green 1.2s ${d}s ease-in-out infinite`,
                  }} />
                ))}
              </div>
            </div>
          )}

          {result && !loading && (
            <>
              {/* Prediction badge */}
              <div className="glass-panel" style={{
                padding: '28px',
                border: `1px solid ${isSeizureResult
                  ? 'rgba(248,113,113,0.3)' : 'rgba(52,211,153,0.25)'}`,
                background: isSeizureResult
                  ? 'rgba(239,68,68,0.05)' : 'rgba(16,185,129,0.04)',
                textAlign: 'center',
                animation: 'slide-in 0.4s cubic-bezier(0.16,1,0.3,1)',
              }}>
                <div style={{
                  display: 'inline-flex', alignItems: 'center', gap: '10px',
                  padding: '10px 20px', borderRadius: '50px',
                  background: isSeizureResult
                    ? 'rgba(248,113,113,0.15)' : 'rgba(52,211,153,0.12)',
                  border: `1px solid ${isSeizureResult
                    ? 'rgba(248,113,113,0.3)' : 'rgba(52,211,153,0.25)'}`,
                  marginBottom: '16px',
                  animation: isSeizureResult ? 'pulse-red 2s ease infinite' : 'none',
                }}>
                  {isSeizureResult
                    ? <AlertTriangle style={{ width: '20px', height: '20px', color: '#F87171' }} />
                    : <CheckCircle  style={{ width: '20px', height: '20px', color: '#34D399' }} />}
                  <span style={{
                    fontSize: '1.1rem', fontWeight: 800,
                    color: isSeizureResult ? '#F87171' : '#34D399',
                    fontFamily: 'var(--font-title)',
                    textTransform: 'uppercase', letterSpacing: '0.05em',
                  }}>
                    {isSeizureResult ? 'Seizure Detected' : 'No Seizure'}
                  </span>
                </div>

                <div style={{ display: 'flex', justifyContent: 'center', gap: '32px' }}>
                  <div>
                    <div style={{ fontSize: '2rem', fontWeight: 800,
                                  color: isSeizureResult ? '#F87171' : '#34D399' }}>
                      {(result.probability * 100).toFixed(1)}%
                    </div>
                    <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                      Seizure Probability
                    </div>
                  </div>
                  <div>
                    <div style={{ fontSize: '2rem', fontWeight: 800,
                                  color: result.seizure_risk === 'High'   ? '#F87171' :
                                         result.seizure_risk === 'Moderate'? '#FBBF24' : '#34D399' }}>
                      {result.seizure_risk}
                    </div>
                    <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                      Risk Level
                    </div>
                  </div>
                </div>
              </div>

              {/* Probability breakdown */}
              <div className="glass-panel" style={{ padding: '22px' }}>
                <h4 style={{ fontSize: '0.85rem', fontWeight: 700,
                              marginBottom: '16px', color: 'var(--text-secondary)' }}>
                  Probability Breakdown
                </h4>
                <ProbBar value={result.probability}
                         label="Seizure (Ictal)"  color="#F87171" />
                <ProbBar value={1 - result.probability}
                         label="Normal (Interictal)"  color="#34D399" />
              </div>

              {/* Model info */}
              <div className="glass-panel" style={{ padding: '18px' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px',
                               marginBottom: '12px' }}>
                  <BarChart2 style={{ width: '15px', height: '15px', color: '#818CF8' }} />
                  <span style={{ fontSize: '0.8rem', fontWeight: 600 }}>Model Info</span>
                </div>
                {[
                  { l: 'Model',    v: 'CNN-BiLSTM v1 (Phase 1)' },
                  { l: 'Input',    v: '178 EEG timesteps · 1 channel' },
                  { l: 'Dataset',  v: 'Bonn + CHB-MIT (SMOTE balanced)' },
                  { l: 'Fix',      v: 'SMOTE 1:1 balance → no more 100% normal bias' },
                ].map(({ l, v }) => (
                  <div key={l} style={{ display: 'flex', gap: '12px',
                                        fontSize: '0.78rem', marginBottom: '6px' }}>
                    <span style={{ color: 'var(--text-muted)', width: '70px',
                                   flexShrink: 0 }}>{l}</span>
                    <span style={{ color: 'var(--text-secondary)' }}>{v}</span>
                  </div>
                ))}
              </div>

              {/* Re-run button */}
              <button id="rerun-btn"
                onClick={() => runPredict(eegVals)}
                className="btn btn-secondary"
                style={{ width: '100%' }}>
                <RefreshCw style={{ width: '14px', height: '14px' }} />
                Re-run Analysis
              </button>
            </>
          )}
        </div>
      </section>

      {/* ── Comparison summary ── */}
      {models.length > 0 && (
        <section className="glass-panel" style={{ padding: '24px', overflowX: 'auto' }}>
          <h3 style={{ fontSize: '0.9rem', fontWeight: 700, marginBottom: '16px',
                       display: 'flex', alignItems: 'center', gap: '8px' }}>
            <TrendingUp style={{ width: '16px', height: '16px', color: '#60A5FA' }} />
            Model Comparison Table
          </h3>
          <table style={{ width: '100%', borderCollapse: 'collapse',
                          fontSize: '0.83rem', textAlign: 'left' }}>
            <thead>
              <tr style={{ color: 'var(--text-muted)', fontSize: '0.75rem',
                           textTransform: 'uppercase', letterSpacing: '0.06em' }}>
                {['Model', 'Type', 'Dataset', 'Accuracy', 'Recall', 'F1', 'ROC-AUC', 'Status'].map(h => (
                  <th key={h} style={{ padding: '10px 14px',
                                       borderBottom: '1px solid var(--border-color)' }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {models.map((m, i) => {
                const pct = v => v !== null && v !== undefined
                  ? (v * 100).toFixed(1) + '%' : '—';
                const clr = v => !v ? 'var(--text-muted)'
                  : v < 0.3 ? '#F87171' : v < 0.6 ? '#FBBF24' : '#34D399';
                return (
                  <tr key={m.id}
                    style={{ borderBottom: '1px solid var(--border-color)',
                             background: m.id === 'cnn_bilstm_v1'
                               ? 'rgba(129,140,248,0.04)' : 'transparent' }}>
                    <td style={{ padding: '12px 14px', fontWeight: 700,
                                  color: m.id === 'cnn_bilstm_v1' ? '#A5B4FC' : 'white' }}>
                      {m.name}
                      {m.id === 'cnn_bilstm_v1' && (
                        <span style={{ fontSize: '0.65rem', marginLeft: '6px',
                                       color: '#818CF8', background: 'rgba(129,140,248,0.15)',
                                       padding: '1px 5px', borderRadius: '10px' }}>NEW</span>
                      )}
                    </td>
                    <td style={{ padding: '12px 14px', color: 'var(--text-secondary)' }}>{m.type}</td>
                    <td style={{ padding: '12px 14px', color: 'var(--text-muted)',
                                  fontSize: '0.75rem' }}>{m.dataset}</td>
                    <td style={{ padding: '12px 14px', color: clr(m.accuracy) }}>{pct(m.accuracy)}</td>
                    <td style={{ padding: '12px 14px', color: clr(m.recall),
                                  fontWeight: 600 }}>{pct(m.recall)}</td>
                    <td style={{ padding: '12px 14px', color: clr(m.f1) }}>{pct(m.f1)}</td>
                    <td style={{ padding: '12px 14px', color: clr(m.roc_auc) }}>{pct(m.roc_auc)}</td>
                    <td style={{ padding: '12px 14px' }}>
                      <span style={{
                        fontSize: '0.7rem', padding: '3px 8px', borderRadius: '12px',
                        background: m.available ? 'rgba(52,211,153,0.1)' : 'rgba(107,114,128,0.15)',
                        color: m.available ? '#34D399' : '#6B7280',
                        border: `1px solid ${m.available ? 'rgba(52,211,153,0.2)' : 'rgba(107,114,128,0.2)'}`,
                        fontWeight: 600,
                      }}>
                        {m.available ? '● Ready' : '○ Not trained'}
                      </span>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </section>
      )}

      <style>{`
        @keyframes spin { to { transform: rotate(360deg); } }
      `}</style>
    </div>
  );
}
