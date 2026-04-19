import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../hooks/useAuth';

export default function LoginPage() {
  const { login, signup } = useAuth();
  const navigate = useNavigate();

  const [mode, setMode] = useState('login'); // 'login' | 'signup'
  const [form, setForm] = useState({ username: '', email: '', password: '' });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  function onChange(e) {
    setForm((f) => ({ ...f, [e.target.name]: e.target.value }));
    setError('');
  }

  async function onSubmit(e) {
    e.preventDefault();
    setLoading(true);
    setError('');
    try {
      if (mode === 'login') {
        await login(form.email, form.password);
      } else {
        await signup(form.username, form.email, form.password);
      }
      navigate('/upload');
    } catch (err) {
      setError(err.response?.data?.error || err.message || 'Authentication failed.');
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="auth-page" role="main">
      <div className="auth-card card" id="auth-card">
        <header className="auth-header">
          <h1>🫁 <span>TB</span>-Detect</h1>
          <p>AI-Powered Tuberculosis Screening</p>
        </header>

        {error && (
          <div className="alert alert-error mb-4" role="alert" id="auth-error">
            ⚠️ {error}
          </div>
        )}

        <form onSubmit={onSubmit} id="auth-form" noValidate>
          {mode === 'signup' && (
            <div className="form-group">
              <label className="form-label" htmlFor="username">Username</label>
              <input
                id="username"
                name="username"
                type="text"
                className="form-input"
                placeholder="johndoe"
                value={form.username}
                onChange={onChange}
                required
                minLength={3}
              />
            </div>
          )}

          <div className="form-group">
            <label className="form-label" htmlFor="email">Email Address</label>
            <input
              id="email"
              name="email"
              type="email"
              className="form-input"
              placeholder="you@hospital.com"
              value={form.email}
              onChange={onChange}
              required
            />
          </div>

          <div className="form-group">
            <label className="form-label" htmlFor="password">Password</label>
            <input
              id="password"
              name="password"
              type="password"
              className="form-input"
              placeholder="••••••••"
              value={form.password}
              onChange={onChange}
              required
              minLength={8}
            />
          </div>

          <button
            id="auth-submit"
            type="submit"
            className="btn btn-primary btn-lg"
            style={{ width: '100%', justifyContent: 'center' }}
            disabled={loading}
          >
            {loading ? '⏳ Please wait…' : mode === 'login' ? '🔐 Sign In' : '🚀 Create Account'}
          </button>
        </form>

        <div className="auth-toggle">
          {mode === 'login' ? (
            <p>
              No account?{' '}
              <button id="auth-toggle-signup" onClick={() => setMode('signup')}>
                Create one
              </button>
            </p>
          ) : (
            <p>
              Already registered?{' '}
              <button id="auth-toggle-login" onClick={() => setMode('login')}>
                Sign in
              </button>
            </p>
          )}
        </div>

        <div className="divider" />
        <p className="text-center text-sm text-muted">
          ⚠️ For research and educational use only. Not a certified medical diagnostic tool.
        </p>
      </div>
    </main>
  );
}
