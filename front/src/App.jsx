// src/App.jsx
import { useEffect } from 'react'
import { Routes, Route, Navigate } from 'react-router-dom'
import { useAuth } from './contexts/AuthContext';

import AdminMain from './pages/AdminMain'
import AddUser from './pages/AddUser'
import Home from './pages/Home'
import SelectFarm from './pages/SelectFarm'
import MainFarm from './pages/MainFarm';
import Report from './pages/Report';
import NotiDetail from './pages/NotiDetail';


export default function App() {
  const { user } = useAuth();

  // 최상위 #app div에 admin 클래스 추가/제거
  useEffect(() => {
    const appDiv = document.getElementById('app');
    if (appDiv) {
      if (user?.role === 'admin') {
        appDiv.classList.add('admin');
      } else {
        appDiv.classList.remove('admin');
      }
    }
  }, [user?.role]);

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
        <>
          <Route path="/admin" element={<AdminMain />} />
          <Route path="/admin/add-user" element={<AddUser />} />       
        </>
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