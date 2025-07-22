// src/App.jsx
import React, { useState } from 'react'
import { Routes, Route, Navigate } from 'react-router-dom'

import Home from './pages/Home'
import SelectFarm from './pages/SelectFarm'  
import AdminPage from './pages/AdminPage'

export default function App() {
  const [role, setRole] = useState(null)

  return (
    <Routes>
      <Route path="/" element={<Home onLogin={setRole} />} />

      {role === 'admin' && (
        <Route path="/admin" element={<AdminPage />} />
      )}

      {role && role !== 'admin' && (
        <Route path="/select-farm" element={<SelectFarm />} />
      )}

      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  )
}