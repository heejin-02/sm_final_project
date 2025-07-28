// src/App.jsx
import React from 'react'
import { Routes, Route, Navigate } from 'react-router-dom'
import { useAuth } from './contexts/AuthContext';

import AdminPage from './pages/AdminPage'
import Home from './pages/Home'
import SelectFarm from './pages/SelectFarm'
import MainFarm from './pages/MainFarm';
import Report from './pages/Report';
import NotiDetail from './pages/NotiDetail';


export default function App() {
  const { user } = useAuth();

  return (
    <Routes>
      <Route 
        path="/"
        element={
          user
            ? <Navigate to={user.role === 'admin' ? '/admin' : '/select-farm'} replace />
            : <Home />
        }
      />

      {user?.role === 'admin' && (
        <Route path="/admin" element={<AdminPage />} />
      )}

      {user?.role && user.role !== 'admin' && (
        <>
          <Route path="/select-farm" element={<SelectFarm />} />
          <Route path="/mainfarm/:id" element={<MainFarm />} />
          <Route path="/report/:period" element={<Report />} />
          <Route path="/notifications/:id" element={<NotiDetail />} />
        </>
      )}

      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  )
}