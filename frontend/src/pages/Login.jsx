import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { supabase } from '../supabaseClient';
import { Activity, Mail, Lock, User, UserCheck } from 'lucide-react';

export default function Login({ onAuthSuccess }) {
  const navigate = useNavigate();
  const [isSignUp, setIsSignUp] = useState(false);
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [fullName, setFullName] = useState('');
  const [role, setRole] = useState('doctor'); // doctor or caregiver
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState({ type: '', text: '' });

  const BACKEND = import.meta.env.VITE_BACKEND_URL || 'http://localhost:8000';

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setMessage({ type: '', text: '' });

    try {
      if (isSignUp) {
        // Call our backend which uses the Supabase Admin API to register AND
        // auto-confirm the email — so users can log in immediately without
        // having to click a verification link in their inbox.
        const resp = await fetch(`${BACKEND}/auth/register`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            email,
            password,
            full_name: fullName,
            role,
          }),
        });

        const result = await resp.json();
        if (!resp.ok) {
          throw new Error(result.detail || 'Registration failed.');
        }

        // Auto-sign in right after registration succeeds
        const { data, error } = await supabase.auth.signInWithPassword({ email, password });
        if (error) throw error;

        if (data.session) {
          onAuthSuccess(data.session.user);
          navigate('/');
        }
      } else {
        // Log In — goes directly to Supabase
        const { data, error } = await supabase.auth.signInWithPassword({
          email,
          password,
        });

        if (error) throw error;

        if (data.session) {
          onAuthSuccess(data.session.user);
          navigate('/');
        }
      }
    } catch (err) {
      setMessage({ type: 'error', text: err.message || 'Authentication failed.' });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{
      minHeight: '100vh',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      background: 'radial-gradient(circle at center, #1E293B 0%, #0F172A 70%, #0B0F19 100%)',
      padding: '20px'
    }}>
      <div className="glass-panel" style={{
        width: '100%',
        maxWidth: '420px',
        padding: '40px 32px',
        animation: 'slide-in 0.4s ease-out'
      }}>
        {/* Logo Header */}
        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '8px', marginBottom: '36px' }}>
          <div style={{
            background: 'linear-gradient(135deg, #3B82F6, #1D4ED8)',
            padding: '12px',
            borderRadius: '12px',
            boxShadow: '0 8px 20px rgba(59, 130, 246, 0.3)'
          }}>
            <Activity style={{ width: '28px', height: '28px', color: 'white' }} />
          </div>
          <h1 style={{ fontSize: '1.75rem', fontWeight: '800', fontFamily: 'var(--font-title)', color: 'white', marginTop: '8px' }}>
            NeuroWatch
          </h1>
          <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
            Seizure Detection & Monitoring Platform
          </p>
        </div>

        {/* Message Banner */}
        {message.text && (
          <div style={{
            padding: '12px 16px',
            borderRadius: 'var(--radius-sm)',
            fontSize: '0.85rem',
            marginBottom: '20px',
            background: message.type === 'error' ? 'rgba(239, 68, 68, 0.12)' : 'rgba(16, 185, 129, 0.12)',
            color: message.type === 'error' ? '#FCA5A5' : '#6EE7B7',
            border: `1px solid ${message.type === 'error' ? 'rgba(239, 68, 68, 0.2)' : 'rgba(16, 185, 129, 0.2)'}`
          }}>
            {message.text}
          </div>
        )}

        {/* Auth Form */}
        <form onSubmit={handleSubmit} style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
          {isSignUp && (
            <>
              <div>
                <label style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', display: 'block', marginBottom: '6px' }}>
                  Full Name
                </label>
                <div style={{ position: 'relative' }}>
                  <User style={{ position: 'absolute', left: '14px', top: '12px', width: '18px', height: '18px', color: 'var(--text-muted)' }} />
                  <input
                    type="text"
                    required
                    placeholder="Dr. John Doe"
                    value={fullName}
                    onChange={(e) => setFullName(e.target.value)}
                    className="input-field"
                    style={{ paddingLeft: '44px' }}
                  />
                </div>
              </div>

              <div>
                <label style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', display: 'block', marginBottom: '6px' }}>
                  Clinical Role
                </label>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px' }}>
                  <button
                    type="button"
                    onClick={() => setRole('doctor')}
                    className="btn"
                    style={{
                      background: role === 'doctor' ? 'rgba(59, 130, 246, 0.15)' : 'rgba(255, 255, 255, 0.03)',
                      color: role === 'doctor' ? '#60A5FA' : 'var(--text-secondary)',
                      borderColor: role === 'doctor' ? '#3B82F6' : 'var(--border-color)',
                      borderWidth: '1px',
                      borderStyle: 'solid'
                    }}
                  >
                    Doctor
                  </button>
                  <button
                    type="button"
                    onClick={() => setRole('caregiver')}
                    className="btn"
                    style={{
                      background: role === 'caregiver' ? 'rgba(59, 130, 246, 0.15)' : 'rgba(255, 255, 255, 0.03)',
                      color: role === 'caregiver' ? '#60A5FA' : 'var(--text-secondary)',
                      borderColor: role === 'caregiver' ? '#3B82F6' : 'var(--border-color)',
                      borderWidth: '1px',
                      borderStyle: 'solid'
                    }}
                  >
                    Caregiver
                  </button>
                </div>
              </div>
            </>
          )}

          <div>
            <label style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', display: 'block', marginBottom: '6px' }}>
              Email Address
            </label>
            <div style={{ position: 'relative' }}>
              <Mail style={{ position: 'absolute', left: '14px', top: '12px', width: '18px', height: '18px', color: 'var(--text-muted)' }} />
              <input
                type="email"
                required
                placeholder="physician@hospital.org"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                className="input-field"
                style={{ paddingLeft: '44px' }}
              />
            </div>
          </div>

          <div>
            <label style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', display: 'block', marginBottom: '6px' }}>
              Password
            </label>
            <div style={{ position: 'relative' }}>
              <Lock style={{ position: 'absolute', left: '14px', top: '12px', width: '18px', height: '18px', color: 'var(--text-muted)' }} />
              <input
                type="password"
                required
                placeholder="••••••••"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                className="input-field"
                style={{ paddingLeft: '44px' }}
              />
            </div>
          </div>

          <button
            type="submit"
            className="btn btn-primary"
            disabled={loading}
            style={{ width: '100%', padding: '12px', marginTop: '10px' }}
          >
            {loading ? 'Processing...' : isSignUp ? 'Create Account' : 'Sign In'}
          </button>
        </form>

        {/* Toggle Mode */}
        <div style={{ marginTop: '24px', textAlign: 'center', fontSize: '0.85rem' }}>
          <span style={{ color: 'var(--text-secondary)' }}>
            {isSignUp ? 'Already have an account? ' : "Don't have an account? "}
          </span>
          <button
            onClick={() => setIsSignUp(!isSignUp)}
            style={{
              background: 'none',
              border: 'none',
              color: '#3B82F6',
              fontWeight: '600',
              cursor: 'pointer'
            }}
          >
            {isSignUp ? 'Sign In' : 'Register Now'}
          </button>
        </div>
      </div>
    </div>
  );
}
