import React from 'react'
import { Routes, Route, Navigate } from 'react-router-dom'
import Home   from './pages/Home'
import Detail from './pages/Detail'
import Write  from './pages/Write'

export default function App() {
  return (
    <Routes>
      <Route path="/"       element={<Home   />} />
      <Route path="detail"  element={<Detail />} />
      <Route path="write"   element={<Write  />} />
      <Route path="*"       element={<Navigate to="/" replace />} />
    </Routes>
  )
}
