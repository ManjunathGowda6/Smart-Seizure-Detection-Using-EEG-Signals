import React, { useEffect, useState } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { supabase } from './supabaseClient';
import Layout from './components/Layout';
import Login from './pages/Login';
import Dashboard from './pages/Dashboard';
import PatientDetail from './pages/PatientDetail';
import QuickScan from './pages/QuickScan';
import ModelTester from './pages/ModelTester';

export default function App() {
  const [session, setSession] = useState(null);
  const [userProfile, setUserProfile] = useState(null);
  const [loading, setLoading] = useState(true);

  const fetchProfile = async (user) => {
    try {
      const { data, error } = await supabase
        .from('profiles')
        .select('*')
        .eq('id', user.id)
        .single();
      
      if (error) throw error;
      setUserProfile(data);
    } catch (err) {
      console.warn("Could not fetch profile, using metadata fallback:", err);
      // Fallback: use user metadata if profile table doesn't have the row yet
      setUserProfile({
        id: user.id,
        full_name: user.user_metadata?.full_name || user.email.split('@')[0],
        role: user.user_metadata?.role || 'doctor'
      });
    }
  };

  useEffect(() => {
    // Fetch current session
    supabase.auth.getSession().then(({ data: { session } }) => {
      setSession(session);
      if (session) {
        fetchProfile(session.user);
      }
      setLoading(false);
    });

    // Listen for auth state updates
    const { data: { subscription } } = supabase.auth.onAuthStateChange((event, session) => {
      setSession(session);
      if (session) {
        fetchProfile(session.user);
      } else {
        setUserProfile(null);
      }
      setLoading(false);
    });

    return () => subscription.unsubscribe();
  }, []);

  if (loading) {
    return (
      <div style={{
        minHeight: '100vh',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        background: '#0B0F19',
        color: 'var(--text-secondary)',
        fontFamily: 'var(--font-sans)',
        fontSize: '0.9rem'
      }}>
        Initialising NeuroWatch session...
      </div>
    );
  }

  return (
    <Router>
      <Routes>
        <Route 
          path="/login" 
          element={!session ? <Login onAuthSuccess={(user) => fetchProfile(user)} /> : <Navigate to="/" />} 
        />
        
        <Route 
          path="/" 
          element={
            session ? (
              <Layout userProfile={userProfile}>
                <Dashboard />
              </Layout>
            ) : (
              <Navigate to="/login" />
            )
          } 
        />

        <Route 
          path="/patient/:patientId" 
          element={
            session ? (
              <Layout userProfile={userProfile}>
                <PatientDetail />
              </Layout>
            ) : (
              <Navigate to="/login" />
            )
          } 
        />

        <Route 
          path="/quick-scan" 
          element={
            session ? (
              <Layout userProfile={userProfile}>
                <QuickScan />
              </Layout>
            ) : (
              <Navigate to="/login" />
            )
          } 
        />

        <Route 
          path="/model-tester" 
          element={
            session ? (
              <Layout userProfile={userProfile}>
                <ModelTester />
              </Layout>
            ) : (
              <Navigate to="/login" />
            )
          } 
        />

        <Route path="*" element={<Navigate to="/" />} />
      </Routes>
    </Router>
  );
}
