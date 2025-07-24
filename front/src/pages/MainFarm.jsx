// src/pages/MainFarm.jsx
import React from 'react';
import { useAuth } from '../contexts/AuthContext';
import AlertList from '../components/AlertList';

export default function MainFarm() {
  const { user } = useAuth();

  const farm = user?.selectedFarm;

  if (!farm) return <p>선택된 농장이 없습니다. 다시 선택해주세요.</p>;

  return (
    <div className="p-4">
      <AlertList/>
    </div>
  );
}
