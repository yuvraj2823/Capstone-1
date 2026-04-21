import React, { createContext, useContext, useState, useCallback } from 'react';
import { login as apiLogin, signup as apiSignup } from '../services/authService';

const AuthContext = createContext(null);

export function AuthProvider({ children }) {
  // Restore from sessionStorage to allow surviving refreshes, but still logout on tab close
  const [user, setUser] = useState(() => {
    try {
      const saved = sessionStorage.getItem('tb_user');
      return saved ? JSON.parse(saved) : null;
    } catch {
      return null;
    }
  });

  const login = useCallback(async (email, password) => {
    const data = await apiLogin(email, password);
    sessionStorage.setItem('tb_token', data.token);
    sessionStorage.setItem('tb_user', JSON.stringify(data.user));
    setUser(data.user);
    return data;
  }, []);

  const signup = useCallback(async (username, email, password) => {
    const data = await apiSignup(username, email, password);
    sessionStorage.setItem('tb_token', data.token);
    sessionStorage.setItem('tb_user', JSON.stringify(data.user));
    setUser(data.user);
    return data;
  }, []);

  const logout = useCallback(() => {
    sessionStorage.removeItem('tb_token');
    sessionStorage.removeItem('tb_user');
    setUser(null);
  }, []);

  return (
    <AuthContext.Provider value={{ user, login, signup, logout }}>
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error('useAuth must be used inside AuthProvider');
  return ctx;
}
