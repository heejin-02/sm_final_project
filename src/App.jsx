// src/App.jsx
import React from 'react'
import { Routes, Route, Navigate } from 'react-router-dom'
import Home   from './pages/Home'
import Detail from './pages/Detail'
import Write  from './pages/Write'
//import AdminDashboard from './pages/AdminDashboard'

export default function App() {
  const role = localStorage.getItem('role')

  return (
    <Routes>
      {/* 1) 기본 로그인/홈 화면 */}
      <Route path="/" element={<Home />} />

      {/* 2) 어드민 전용 경로 */}
      {role === 'admin' && (
        <Route path="/admin/*" element={<AdminDashboard />} />
      )}

      {/* 3) 로그인된 사용자 전용 경로 */}
      {!!role && (
        <>
          <Route path="/detail" element={<Detail />} />
          <Route path="/write"  element={<Write  />} />
          <Route path="/select-farm"  element={<SelectFarm  />} />
        </>
      )}

      {/* 4) 그 외 모두 홈으로 리다이렉트 */}
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  )
}
