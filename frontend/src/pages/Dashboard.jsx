import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { supabase } from '../supabaseClient';
import { Search, Plus, Users, AlertCircle, RefreshCw, FileText } from 'lucide-react';

export default function Dashboard() {
  const navigate = useNavigate();
  const [patients, setPatients] = useState([]);
  const [searchTerm, setSearchTerm] = useState('');
  const [loading, setLoading] = useState(true);
  const [isModalOpen, setIsModalOpen] = useState(false);
  
  // New Patient Form state
  const [fullName, setFullName] = useState('');
  const [age, setAge] = useState('');
  const [gender, setGender] = useState('Male');
  const [diagnosis, setDiagnosis] = useState('');
  const [formError, setFormError] = useState('');

  // Statistics
  const [stats, setStats] = useState({
    totalPatients: 0,
    activeAlerts: 0,
    totalScans: 0
  });

  const fetchPatientsAndStats = async () => {
    setLoading(true);
    try {
      const session = (await supabase.auth.getSession()).data.session;
      if (!session) return;
      
      const token = session.access_token;

      // 1. Fetch Patients
      const patRes = await fetch('http://localhost:8000/patients', {
        headers: { 'Authorization': `Bearer ${token}` }
      });
      if (!patRes.ok) throw new Error("Failed to fetch patients");
      const patientsList = await patRes.json();
      setPatients(patientsList);

      // 2. Fetch Alerts for count
      const alertsRes = await fetch('http://localhost:8000/alerts', {
        headers: { 'Authorization': `Bearer ${token}` }
      });
      let activeAlertsCount = 0;
      if (alertsRes.ok) {
        const alertsList = await alertsRes.json();
        activeAlertsCount = alertsList.filter(a => !a.is_read).length;
      }

      // Calculate stats
      setStats({
        totalPatients: patientsList.length,
        activeAlerts: activeAlertsCount,
        totalScans: 0 // Will get dynamically from uploads
      });
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchPatientsAndStats();
  }, []);

  const handleCreatePatient = async (e) => {
    e.preventDefault();
    setFormError('');
    try {
      const session = (await supabase.auth.getSession()).data.session;
      if (!session) return;
      
      const token = session.access_token;

      const response = await fetch('http://localhost:8000/patients/', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`
        },
        body: JSON.stringify({
          full_name: fullName,
          age: age ? parseInt(age) : null,
          gender,
          diagnosis
        })
      });

      if (!response.ok) {
        const errData = await response.json();
        throw new Error(errData.detail || 'Failed to create patient');
      }

      // Refresh list and close modal
      await fetchPatientsAndStats();
      setIsModalOpen(false);
      // Reset form
      setFullName('');
      setAge('');
      setGender('Male');
      setDiagnosis('');
    } catch (err) {
      setFormError(err.message);
    }
  };

  const filteredPatients = patients.filter(p =>
    (p.full_name || '').toLowerCase().includes(searchTerm.toLowerCase()) ||
    (p.diagnosis && p.diagnosis.toLowerCase().includes(searchTerm.toLowerCase()))
  );

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '30px' }}>
      {/* Statistics Cards Grid */}
      <section style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))', gap: '20px' }}>
        <div className="glass-panel" style={{ padding: '24px', display: 'flex', alignItems: 'center', gap: '20px' }}>
          <div style={{ padding: '16px', background: 'rgba(59, 130, 246, 0.15)', borderRadius: '12px', color: '#60A5FA' }}>
            <Users style={{ width: '28px', height: '28px' }} />
          </div>
          <div>
            <span style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>Monitored Patients</span>
            <h3 style={{ fontSize: '2rem', fontWeight: '800', marginTop: '4px' }}>{stats.totalPatients}</h3>
          </div>
        </div>

        <div className="glass-panel" style={{ padding: '24px', display: 'flex', alignItems: 'center', gap: '20px' }}>
          <div style={{ padding: '16px', background: 'rgba(239, 68, 68, 0.12)', borderRadius: '12px', color: '#FCA5A5' }}>
            <AlertCircle style={{ width: '28px', height: '28px' }} />
          </div>
          <div>
            <span style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>Active Seizure Alerts</span>
            <h3 style={{ fontSize: '2rem', fontWeight: '800', marginTop: '4px' }}>{stats.activeAlerts}</h3>
          </div>
        </div>

        <div className="glass-panel" style={{ padding: '24px', display: 'flex', alignItems: 'center', gap: '20px' }}>
          <div style={{ padding: '16px', background: 'rgba(16, 185, 129, 0.12)', borderRadius: '12px', color: '#6EE7B7' }}>
            <FileText style={{ width: '28px', height: '28px' }} />
          </div>
          <div>
            <span style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>EEG Reports Compiled</span>
            <h3 style={{ fontSize: '2rem', fontWeight: '800', marginTop: '4px' }}>{stats.totalPatients}</h3>
          </div>
        </div>
      </section>

      {/* Patients List Section */}
      <section className="glass-panel" style={{ padding: '30px' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '16px', marginBottom: '24px' }}>
          <h3 style={{ fontSize: '1.25rem', fontWeight: '700' }}>Active Clinical Patients</h3>
          
          <div style={{ display: 'flex', gap: '12px', alignItems: 'center', flexWrap: 'wrap' }}>
            {/* Search */}
            <div style={{ position: 'relative', minWidth: '240px' }}>
              <Search style={{ position: 'absolute', left: '12px', top: '10px', width: '16px', height: '16px', color: 'var(--text-muted)' }} />
              <input
                type="text"
                placeholder="Search by name or diagnosis..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="input-field"
                style={{ padding: '8px 12px 8px 36px', fontSize: '0.85rem' }}
              />
            </div>

            {/* Refresh */}
            <button onClick={fetchPatientsAndStats} className="btn btn-secondary" style={{ padding: '8px 12px' }}>
              <RefreshCw style={{ width: '16px', height: '16px' }} />
            </button>

            {/* Add Patient Button */}
            <button onClick={() => setIsModalOpen(true)} className="btn btn-primary" style={{ padding: '8px 16px', fontSize: '0.85rem' }}>
              <Plus style={{ width: '16px', height: '16px' }} />
              <span>Add Patient</span>
            </button>
          </div>
        </div>

        {/* Patients Table */}
        {loading ? (
          <div style={{ display: 'flex', justifyContent: 'center', padding: '40px', color: 'var(--text-secondary)' }}>
            Loading patients data...
          </div>
        ) : filteredPatients.length === 0 ? (
          <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', padding: '60px 40px', color: 'var(--text-secondary)' }}>
            <Users style={{ width: '48px', height: '48px', color: 'var(--text-muted)', marginBottom: '16px' }} />
            <h4 style={{ color: 'white', marginBottom: '4px' }}>No Patients Found</h4>
            <p style={{ fontSize: '0.85rem' }}>Add a new patient or adjust your search filter.</p>
          </div>
        ) : (
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left' }}>
              <thead>
                <tr style={{ borderBottom: '1px solid var(--border-color)', color: 'var(--text-muted)', fontSize: '0.8rem', textTransform: 'uppercase' }}>
                  <th style={{ padding: '12px 16px' }}>Name</th>
                  <th style={{ padding: '12px 16px' }}>Age</th>
                  <th style={{ padding: '12px 16px' }}>Gender</th>
                  <th style={{ padding: '12px 16px' }}>Diagnosis</th>
                  <th style={{ padding: '12px 16px', textAlign: 'right' }}>Actions</th>
                </tr>
              </thead>
              <tbody>
                {filteredPatients.map((patient) => (
                  <tr
                    key={patient.id}
                    onClick={() => navigate(`/patient/${patient.id}`)}
                    style={{
                      borderBottom: '1px solid var(--border-color)',
                      cursor: 'pointer',
                      transition: 'background 0.2s'
                    }}
                    onMouseEnter={(e) => e.currentTarget.style.background = 'rgba(255, 255, 255, 0.02)'}
                    onMouseLeave={(e) => e.currentTarget.style.background = 'transparent'}
                  >
                    <td style={{ padding: '16px', fontWeight: '600', color: 'white' }}>{patient.full_name}</td>
                    <td style={{ padding: '16px', color: 'var(--text-secondary)' }}>{patient.age || 'N/A'}</td>
                    <td style={{ padding: '16px', color: 'var(--text-secondary)' }}>{patient.gender || 'N/A'}</td>
                    <td style={{ padding: '16px', color: 'var(--text-secondary)', maxWidth: '300px', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                      {patient.diagnosis || <span style={{ color: 'var(--text-muted)', fontStyle: 'italic' }}>No notes</span>}
                    </td>
                    <td style={{ padding: '16px', textAlign: 'right' }}>
                      <span style={{ fontSize: '0.75rem', color: '#60A5FA', fontWeight: '500' }}>Monitor Dashboard →</span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>

      {/* Add Patient Modal */}
      {isModalOpen && (
        <div className="modal-overlay">
          <div className="modal-content glass-panel" style={{ border: '1px solid rgba(59, 130, 246, 0.2)' }}>
            <h3 style={{ fontSize: '1.25rem', fontWeight: 'bold', marginBottom: '20px' }}>Register New Patient</h3>
            
            {formError && (
              <div style={{
                padding: '10px 14px',
                background: 'rgba(239, 68, 68, 0.12)',
                color: '#FCA5A5',
                fontSize: '0.8rem',
                borderRadius: '6px',
                marginBottom: '16px',
                border: '1px solid rgba(239, 68, 68, 0.2)'
              }}>
                {formError}
              </div>
            )}

            <form onSubmit={handleCreatePatient} style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
              <div>
                <label style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', display: 'block', marginBottom: '6px' }}>
                  Patient Full Name *
                </label>
                <input
                  type="text"
                  required
                  placeholder="E.g. Jane Doe"
                  value={fullName}
                  onChange={(e) => setFullName(e.target.value)}
                  className="input-field"
                />
              </div>

              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px' }}>
                <div>
                  <label style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', display: 'block', marginBottom: '6px' }}>
                    Age (Years)
                  </label>
                  <input
                    type="number"
                    min="0"
                    placeholder="E.g. 14"
                    value={age}
                    onChange={(e) => setAge(e.target.value)}
                    className="input-field"
                  />
                </div>

                <div>
                  <label style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', display: 'block', marginBottom: '6px' }}>
                    Gender
                  </label>
                  <select
                    value={gender}
                    onChange={(e) => setGender(e.target.value)}
                    className="input-field"
                    style={{ background: '#0F172A' }}
                  >
                    <option value="Male">Male</option>
                    <option value="Female">Female</option>
                    <option value="Other">Other</option>
                  </select>
                </div>
              </div>

              <div>
                <label style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', display: 'block', marginBottom: '6px' }}>
                  Initial Diagnosis / Clinical Notes
                </label>
                <textarea
                  placeholder="Enter medical conditions, seizure symptoms, or epilepsy classification details..."
                  rows="3"
                  value={diagnosis}
                  onChange={(e) => setDiagnosis(e.target.value)}
                  className="input-field"
                  style={{ resize: 'none' }}
                />
              </div>

              <div style={{ display: 'flex', gap: '12px', justifyContent: 'flex-end', marginTop: '10px' }}>
                <button type="button" onClick={() => setIsModalOpen(false)} className="btn btn-secondary">
                  Cancel
                </button>
                <button type="submit" className="btn btn-primary">
                  Create Record
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  );
}
