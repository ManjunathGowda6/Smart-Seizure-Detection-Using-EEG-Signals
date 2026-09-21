import React, { useEffect, useState } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import { supabase } from '../supabaseClient';
import { Activity, Users, Bell, LogOut, Zap, Brain } from 'lucide-react';

export default function Layout({ children, userProfile }) {
  const navigate = useNavigate();
  const location = useLocation();
  const [unreadCount, setUnreadCount] = useState(0);

  useEffect(() => {
    if (!userProfile) return;
    
    // Fetch unread alerts count
    const fetchUnreadAlerts = async () => {
      try {
        const { token } = (await supabase.auth.getSession()).data.session || {};
        if (!token) return;
        
        const res = await fetch('http://localhost:8000/alerts', {
          headers: {
            'Authorization': `Bearer ${token}`
          }
        });
        if (res.ok) {
          const data = await res.json();
          setUnreadCount(data.filter(a => !a.is_read).length);
        }
      } catch (err) {
        console.error("Error fetching alerts count:", err);
      }
    };

    fetchUnreadAlerts();
    // Refresh alerts count every 15 seconds
    const interval = setInterval(fetchUnreadAlerts, 15000);
    return () => clearInterval(interval);
  }, [userProfile]);

  const handleLogout = async () => {
    await supabase.auth.signOut();
    navigate('/login');
  };

  const navItems = [
    { name: 'Patient Monitor', path: '/',             icon: Users  },
    { name: 'Quick Scan',      path: '/quick-scan',   icon: Zap    },
    { name: 'Model Tester',   path: '/model-tester', icon: Brain  },
  ];

  return (
    <div className="layout-container">
      {/* Sidebar */}
      <aside className="sidebar">
        <div className="flex items-center gap-3 px-2 py-3" style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <div className="p-2 rounded-lg bg-blue-600" style={{ background: '#2563EB', padding: '8px', borderRadius: '8px', display: 'flex' }}>
            <Activity className="h-6 w-6 text-white" style={{ color: 'white', width: '24px', height: '24px' }} />
          </div>
          <div>
            <h1 style={{ fontSize: '1.25rem', fontWeight: 'bold', fontFamily: 'var(--font-title)' }}>NeuroWatch</h1>
            <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Seizure Telemetry</span>
          </div>
        </div>

        <nav style={{ display: 'flex', flexDirection: 'column', gap: '8px', flex: 1 }}>
          {navItems.map((item) => {
            const Icon = item.icon;
            const isActive = location.pathname === item.path;
            return (
              <button
                key={item.path}
                onClick={() => navigate(item.path)}
                className="btn"
                style={{
                  justifyContent: 'flex-start',
                  width: '100%',
                  background: isActive ? 'rgba(59, 130, 246, 0.15)' : 'transparent',
                  color: isActive ? '#60A5FA' : 'var(--text-secondary)',
                  border: isActive ? '1px solid rgba(59, 130, 246, 0.25)' : '1px solid transparent',
                  padding: '12px 16px',
                  borderRadius: 'var(--radius-sm)'
                }}
              >
                <Icon style={{ width: '18px', height: '18px' }} />
                <span>{item.name}</span>
              </button>
            );
          })}
        </nav>

        {/* User Card */}
        {userProfile && (
          <div className="glass-panel" style={{ padding: '16px', display: 'flex', flexDirection: 'column', gap: '12px' }}>
            <div style={{ display: 'flex', flexDirection: 'column' }}>
              <span style={{ fontSize: '0.9rem', fontWeight: '600', color: 'var(--text-primary)' }}>
                {userProfile.full_name || 'Medical Staff'}
              </span>
              <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)', textTransform: 'uppercase' }}>
                {userProfile.role || 'Doctor'}
              </span>
            </div>
            <button
              onClick={handleLogout}
              className="btn btn-secondary"
              style={{ padding: '8px 12px', fontSize: '0.8rem', width: '100%', justifyContent: 'center' }}
            >
              <LogOut style={{ width: '14px', height: '14px' }} />
              <span>Log Out</span>
            </button>
          </div>
        )}
      </aside>

      {/* Main Panel */}
      <main className="main-content">
        {/* Top Header */}
        <header style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '32px' }}>
          <div>
            <span style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>Clinical Dashboard</span>
            <h2 style={{ fontSize: '1.75rem', fontWeight: '700' }}>
              {location.pathname === '/'            ? 'Patient Telemetry Dashboard' :
               location.pathname === '/quick-scan'  ? 'EEG Quick Scan' :
               location.pathname === '/model-tester'? 'CNN-BiLSTM v1 — Model Tester' :
               'Patient Profile'}
            </h2>
          </div>
          
          <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
            {/* Alerts Bell Indicator */}
            <div style={{ position: 'relative', cursor: 'pointer' }}>
              <div 
                className="glass-panel" 
                style={{ padding: '10px', borderRadius: '50%', display: 'flex', alignItems: 'center', justifyContent: 'center' }}
              >
                <Bell style={{ width: '18px', height: '18px', color: 'var(--text-primary)' }} />
              </div>
              {unreadCount > 0 && (
                <span 
                  style={{
                    position: 'absolute',
                    top: '-4px',
                    right: '-4px',
                    background: 'var(--color-danger)',
                    color: 'white',
                    fontSize: '0.65rem',
                    fontWeight: 'bold',
                    padding: '2px 6px',
                    borderRadius: '10px',
                    border: '2px solid var(--bg-main)'
                  }}
                >
                  {unreadCount}
                </span>
              )}
            </div>
            
            <div style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', textAlign: 'right' }}>
              <div>{new Date().toLocaleDateString('en-US', { weekday: 'long', month: 'short', day: 'numeric' })}</div>
              <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Telemetry Active</div>
            </div>
          </div>
        </header>

        {/* Inner Content */}
        <div className="fade-in-up">
          {children}
        </div>
      </main>
    </div>
  );
}
