// src/App.jsx
import React from 'react'
import { Routes, Route, Navigate } from 'react-router-dom'
import { useAuth } from './contexts/AuthContext';

import Home from './pages/Home'
import SelectFarm from './pages/SelectFarm'  
import AdminPage from './pages/AdminPage'

export default function App() {
  const { user } = useAuth();

  return (
    <Routes>
      <Route path="/" element={<Home />} />

      {user?.role === 'admin' && (
        <Route path="/admin" element={<AdminPage />} />
      )}

      {user?.role && user.role !== 'admin' && (
        <Route path="/select-farm" element={<SelectFarm />} />
      )}

      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  )
}